# `scripts/` — Pipeline Reference

Every executable in the fine-tuning → serving → evaluation pipeline, with its
purpose. **Python files do the ML work** (deterministic, testable); **shell
scripts manage the GPU lifecycle and orchestration** (serving, sequencing,
teardown).

## Lifecycle at a glance

```
setup_env.sh                         one-time: 3 venvs, base models, patches, registrations
        │
prepare_toolace.py                   ToolACE -> chat-formatted JSONL (train/val)
        │
run_pipeline.sh <7b|14b>             top-level orchestrator (per model size)
        ├─ train_sft.py              LoRA/QLoRA SFT  ->  PEFT adapter
        ├─ merge_lora.py             adapter -> merged bf16 weights (servable)
        ├─ serve_vllm.sh             start vLLM OpenAI server (bf16 | fp8)
        ├─ run_bfcl.sh               BFCL python-subset eval against the server
        ├─ benchmark_latency.py      TTFT / latency / throughput @ conc 1/16/32
        └─ kill_server.sh            tear down server + free VRAM
        │
summarize_results.py                 raw BFCL scores -> per-phase summary.json
make_report.py                       all summaries -> consolidated results/REPORT.md
```

Supporting/one-off: `serve_eval_bench.sh`, `patch_vllm_metrics.py`,
`register_bfcl_models.py`, `register_qwen35_nothink.py`.

---

## Python files (ML logic)

### Data
- **`prepare_toolace.py`** — Converts the raw `Team-ACE/ToolACE` dataset into
  chat-formatted JSONL (`{messages:[...]}`). Role normalization
  (`gpt→assistant`, `human→user`, `function→tool`), light structural hygiene
  (drop rows with no assistant turn, enforce system→user→assistant shape),
  deterministic seed-42 train/val split. Produces `data/toolace_{train,val}.jsonl`.

### Training & merging
- **`train_sft.py`** — Core trainer. LoRA / QLoRA SFT for tool-calling. Key
  design choices: **assistant-only loss masking** (prefix-delta over the chat
  template — only tool-call/answer tokens contribute loss), LoRA on **all 7
  attention + MLP projections** (q,k,v,o,gate,up,down), bf16 + gradient
  checkpointing + cosine schedule w/ warmup, **non-finite-gradient skipping**
  (a few ToolACE rows overflow to NaN loss in bf16), and **drop-don't-truncate**
  for over-length examples. `--method {lora,qlora}` is the only difference
  between the two recipes (QLoRA loads the frozen base in 4-bit NF4). Outputs a
  PEFT adapter dir.
- **`merge_lora.py`** — Merges a PEFT adapter into the base model and saves
  full-precision (bf16) weights ready for vLLM. For QLoRA adapters the base is
  reloaded in **bf16, not 4-bit**, before merging — the accuracy-safe deploy path.

### Evaluation registration
- **`register_bfcl_models.py`** — Idempotently injects `ModelConfig` entries for
  our Qwen2.5 base + ToolACE checkpoints into the installed `bfcl-eval` registry
  (it ships Qwen3 but not Qwen2.5 or our finetunes). Registered in **prompt mode**
  (`QwenHandler`) so eval format matches ToolACE's training format — apples-to-apples.
- **`register_qwen35_nothink.py`** — Registers a **no-thinking** variant of
  Qwen3.5-9B by prefilling an empty `<think>\n\n</think>` block. Qwen3.5 reasons
  by default (~6× latency, no accuracy benefit), and prompt-mode bypasses the
  chat template's `enable_thinking` flag, so the no-think behavior is forced here.

### Benchmarking
- **`benchmark_latency.py`** — Production-style latency/throughput benchmark
  against a running vLLM server. Streamed chat-completions at concurrency
  **1 / 16 / 32**, measuring **TTFT** (p50/p90/p99), **E2E latency**
  (p50/p90/p99), **TPOT**, output tokens/sec, and requests/sec. Source of all
  reported performance numbers. Prompts mimic BFCL/production tool-calling shapes.

### Reporting / aggregation
- **`summarize_results.py`** — Turns one BFCL phase's raw score CSVs into the
  fair Python-subset metric. **Deliberately ignores BFCL's `data_overall.csv`**
  (which averages in multi-turn/web/memory sections we did not run) and instead
  computes `python_macro_avg` = equal-weight mean of the 11 Python categories,
  plus the full per-category table. Writes per-phase `summary.json`.
- **`make_report.py`** — Aggregates every `results/<phase>/summary.json` +
  latency JSON into one consolidated `results/REPORT.md` — a single source of
  truth to cross-check the README/slide numbers against.

---

## Shell scripts (orchestration & infrastructure)

### Environment
- **`setup_env.sh`** — One-time bootstrap on a fresh 1×H100 box (Ubuntu 24.04,
  Python 3.12). Installs `uv`, creates the **three isolated venvs**
  (train / vllm / bfcl) from the pinned `requirements-*.txt` so dependencies
  never collide, downloads the base models from HF, applies the vLLM metrics
  patch, and runs the BFCL model registrations. This is step 0 of reproduction.

### Serving lifecycle
- **`serve_vllm.sh`** — Launches the vLLM OpenAI-compatible server for a given
  model / port / quantization (`none` | `fp8`). The **hosting** script — all
  production serving flags live here (max-model-len, max-num-seqs, prefix
  caching, gpu-memory-utilization). `fp8` = on-the-fly W8A8 FP8 on H100 (the
  "optimized" deployment).
- **`kill_server.sh`** — Cleanly stops the server **and its vLLM EngineCore
  subprocesses** (killing only the parent leaks tens of GiB of VRAM). Uses the
  `[v]llm` bracket trick so it never `pkill`s itself. This clean teardown is what
  makes the sequential single-H100 design work.

### Evaluation
- **`run_bfcl.sh`** — Runs BFCL generation + scoring against an **already-running**
  vLLM server on a given port. Handles `--skip-server-setup` + remote-base-URL
  wiring and the tokenizer path (for token counting); writes raw result+score
  under `results/<phase>/`.

### Orchestration
- **`serve_eval_bench.sh`** — Reusable **single-phase** wrapper: serve → BFCL
  eval → latency benchmark → stop, as one command. Used for one-off evals
  outside the main pipeline (e.g. the QLoRA and Qwen3.5-9B runs).
- **`run_pipeline.sh`** — Top-level **end-to-end orchestrator** for one model
  size (`7b` | `14b`). Runs 4 stages sequentially —
  **baseline (bf16) → train+merge → finetuned (bf16) → optimized (fp8)** — each
  stage serving, evaluating, benchmarking, then tearing down before the next, so
  the single H100 is never double-booked. (Inlines its own serve/eval/bench
  lifecycle rather than calling `serve_eval_bench.sh`, so it can sequence
  training between stages on a single persistent port.)

---

## Reproduce everything

```bash
# 0. one-time: venvs (pinned), base models, patches, registrations
bash scripts/setup_env.sh

# 1. data
.venv-train/bin/python scripts/prepare_toolace.py

# 2. full pipeline per size (baseline -> LoRA -> fp8)
bash scripts/run_pipeline.sh 7b
bash scripts/run_pipeline.sh 14b

# 3. recommended model variant (QLoRA = one flag), then eval
.venv-train/bin/python scripts/train_sft.py --base-model models/Qwen2.5-7B-Instruct \
  --output-dir checkpoints/Qwen2.5-7B-Instruct-qlora --method qlora \
  --epochs 2 --lr 1e-4 --warmup-ratio 0.05 --max-grad-norm 1.0 \
  --per-device-batch 8 --grad-accum 4 --max-len 4096 --grad-checkpointing 1
.venv-train/bin/python scripts/merge_lora.py --base-model models/Qwen2.5-7B-Instruct \
  --adapter checkpoints/Qwen2.5-7B-Instruct-qlora --out models/Qwen2.5-7B-Instruct-ToolACE
bash scripts/serve_eval_bench.sh models/Qwen2.5-7B-Instruct-ToolACE \
  Qwen2.5-7B-Instruct-ToolACE finetuned-7b-qlora none models/Qwen2.5-7B-Instruct

# 4. consolidated report
.venv-bfcl/bin/python scripts/make_report.py
```

> Results reproduce **within run-to-run variance, not bit-exact** (vLLM
> nondeterminism + near-greedy temp 0.001). Latency/TTFT numbers are
> H100-specific.
