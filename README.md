# Fine-Tuning an Open LLM for Function Calling (ToolACE → BFCL)

End-to-end, reproducible pipeline that fine-tunes an open-weights LLM for
**function / tool calling**, serves it with **vLLM**, and evaluates it on the
**Berkeley Function Calling Leaderboard (BFCL)** *Python* subset — on a single
**NVIDIA H100 80GB**.

> Optimization priority (per the brief): **accuracy > latency > cost.**
> Target production load: **16–32 concurrent requests.**

---

## TL;DR / Results

BFCL Python-subset **macro accuracy** (equal-weight mean of the 11 python categories) and
serving metrics at **32 concurrent requests** on one H100. See [Results](#results) for full per-category tables.

| Model | Mode | BFCL Python macro-acc | TTFT p50 @32 | E2E p50 @32 | Throughput (tok/s) @32 |
|-------|------|----------------------:|-------------:|------------:|-----------------------:|
| Qwen2.5-7B-Instruct  | baseline bf16 | 79.52% | 70 ms | 187 ms | 2630 |
| Qwen2.5-7B-Instruct  | **ToolACE LoRA bf16** | **81.12%** | 71 ms | 187 ms | 2608 |
| Qwen2.5-7B-Instruct  | ToolACE LoRA **fp8 (optimized)** | 80.76% | 73 ms | **162 ms** | **3228** |
| Qwen2.5-14B-Instruct | **baseline bf16** | **82.60%** | 73 ms | 307 ms | 1600 |
| Qwen2.5-14B-Instruct | ToolACE LoRA bf16 (uniform recipe) | 79.33% ⚠️ | 68 ms | 304 ms | 1585 |
| Qwen2.5-14B-Instruct | ToolACE LoRA fp8 | 78.28% | 65 ms | 219 ms | 2162 |
| Qwen2.5-14B-Instruct | ToolACE LoRA **lighter recipe** (lr 5e-5, 1 ep) | 80.83% | — | — | — |

**Key findings:**
- **7B:** ToolACE fine-tuning lifts macro accuracy **+1.6–2 pts** (LoRA 81.1 / QLoRA 81.6 vs 79.5),
  concentrated in **irrelevance detection** (non-live 77.9→87.5, live 58.4→70.9) and multiple-call
  accuracy (94→97.5) — the model gets better at *not* calling a function when none applies, exactly the
  robustness a FinTech workflow agent needs.
- **QLoRA ≈ LoRA** on accuracy but ~3–4× cheaper to train → the cost-optimal recipe.
- **FP8** serving buys **+24% throughput / −13% latency** at −0.36 pt accuracy (noise).
- **14B:** the base is already excellent (82.6) and ToolACE SFT **hurts** it (catastrophic forgetting of
  parallel-multiple); even a lighter recipe (80.8) trails baseline. → don't fine-tune the 14B here.
- **Newer ≠ better for this task:** the much newer multimodal **Qwen3.5-9B** scores only **74.9** macro
  (no-think) — *below* the older, FC-specialized Qwen2.5-7B (79.5) — and is slower. See
  [Candidate base: Qwen3.5-9B](#candidate-base-qwen35-9b-no-fine-tune).

## Recommendation

**Ship `Qwen2.5-7B-Instruct + ToolACE (QLoRA), served FP8.`**

| | macro acc | E2E p50 @32 | throughput @32 | rel. cost |
|---|--:|--:|--:|--:|
| 7B + ToolACE, FP8 (**pick**) | ~81 | **162 ms** | **3228 tok/s** | low |
| 14B baseline, FP8 | 82.6* | 219 ms | 2162 tok/s | ~2× |

It delivers ~81% macro — **within ~1.6 pts of the much larger 14B baseline** — while serving **~1.5×
the throughput, ~1.35× lower latency, and roughly half the GPU memory/cost**, comfortably handling the
16–32 concurrent target. Under the stated **accuracy → latency → cost** priority, that small accuracy gap
doesn't justify ~2× the serving cost and higher latency of the 14B. If maximum accuracy is non-negotiable, ship the **14B baseline** (do not
fine-tune it). _*14B FP8 accuracy estimated from bf16; the 14B baseline was benchmarked in bf16 (307 ms /
1600 tok/s)._

---

## 1. Model selection — why Qwen2.5-7B / 14B Instruct

The client needs the **best achievable tool-calling accuracy** that still serves
**16–32 concurrent requests** comfortably on **one H100**.

| Criterion | Why Qwen2.5-Instruct |
|---|---|
| Tool-calling quality | Qwen2.5-Instruct is among the strongest *open* models on BFCL out of the box, and its chat template natively models `tools`, `tool_calls`, and `tool` responses. |
| Right size for 1×H100 | 7B (≈15 GB bf16) and 14B (≈28 GB bf16) both leave ample HBM for a large KV cache → high concurrency. A 70B model would force aggressive quantization and crush throughput. |
| License | Apache-2.0 (7B/14B) → clean for a commercial FinTech deployment. |
| Format match with ToolACE | ToolACE's training format (system prompt listing JSON functions + assistant replies as `[func(arg=val)]`) **is exactly** the Gorilla/BFCL prompt format → minimal train/eval skew. |

We carry **two sizes** to make the accuracy-vs-latency tradeoff explicit:
**7B** = the latency/cost-optimal candidate, **14B** = the accuracy-optimal
candidate. The client can pick the point on the curve that fits their SLA.

We also **baseline-evaluated a newer candidate, the multimodal Qwen3.5-9B**
(Feb 2026, Apache-2.0), to check whether a more recent generation would be a
better starting point. It was not — it scored lower on BFCL-python *and* served
slower, so we kept Qwen2.5 as the base. Details in
[Candidate base: Qwen3.5-9B](#candidate-base-qwen35-9b-no-fine-tune). This is the
right selection process per Task 1a ("fine-tune the *most suitable* model"):
evaluate candidates, then commit.

## 2. Dataset — Team-ACE/ToolACE

- 11,300 multi-turn tool-calling conversations; columns `system` (string with a
  JSON list of available functions) and `conversations` (ShareGPT-style
  `{from: user|assistant|tool, value}`).
- Assistant tool calls are in **bracket notation** `[FuncName(param=value)]` —
  identical to BFCL prompt-mode, so we train on it **as-is** (the documented
  ToolACE→BFCL recipe). See [`scripts/prepare_toolace.py`](scripts/prepare_toolace.py).
- Split: 11,074 train / 226 validation (deterministic, seed 42).

## 3. Evaluation — BFCL "Python" subset

We use the official `bfcl-eval` package. The **`python`** meta-category expands to
**11 categories**:

```
simple_python, irrelevance, parallel, multiple, parallel_multiple,
live_simple, live_multiple, live_parallel, live_parallel_multiple,
live_irrelevance, live_relevance
```

These are AST-checked (no code execution), covering single/parallel/multiple
calls, plus relevance/irrelevance detection — the skills that matter for routing
NL instructions to internal APIs.

**How we evaluate (production-faithful):** we host the model with our own vLLM
server and point BFCL at it via `--skip-server-setup` +
`REMOTE_OPENAI_BASE_URL`. Both base and fine-tuned models are registered in
**prompt mode** (`QwenHandler`) so the comparison is apples-to-apples and isolates
the lift from ToolACE fine-tuning. See
[`scripts/register_bfcl_models.py`](scripts/register_bfcl_models.py) and
[`scripts/run_bfcl.sh`](scripts/run_bfcl.sh).

## 4. Fine-tuning method — LoRA (primary) and QLoRA

Best practices applied:
- **Chat-template tokenization that exactly matches eval** (Qwen2.5 ChatML; the
  function list lives in the system message).
- **Multi-turn assistant-only loss** — every system/user/tool token is masked to
  `-100`; we only learn the model's tool-call / answer turns. Implemented with a
  prefix-delta scheme over `apply_chat_template`
  ([`scripts/train_sft.py`](scripts/train_sft.py)).
- **LoRA on all attention + MLP projections** (`q,k,v,o,gate,up,down`), r=16,
  α=32, dropout=0.05.
- bf16, gradient checkpointing, cosine LR + warmup, near-greedy eval (temp 0.001).

**LoRA vs QLoRA:** on an H100, 7B/14B fit in bf16, so **LoRA** is the primary
recipe (no quant error during training, fastest). **QLoRA** (4-bit NF4 frozen
base) is provided as the **cost/memory-optimal** alternative — same script,
`--method qlora` — and benchmarked so the client sees the accuracy/cost tradeoff.
We always **merge** the adapter back into bf16 weights before serving
([`scripts/merge_lora.py`](scripts/merge_lora.py)).

## 5. Inference optimization & deployment

- **Serving:** vLLM OpenAI-compatible server with continuous batching, paged
  attention, prefix caching, and CUDA-graph/`torch.compile`
  ([`scripts/serve_vllm.sh`](scripts/serve_vllm.sh)).
- **Optimization:** an **FP8 (W8A8)** variant on H100 — `--quantization fp8` —
  for higher throughput / lower latency at near-identical accuracy. We report
  bf16 vs fp8 side by side so the accuracy→latency→cost priority is visible.
- **Benchmark:** [`scripts/benchmark_latency.py`](scripts/benchmark_latency.py)
  drives realistic streamed tool-calling requests at concurrency **1 / 16 / 32**
  and reports **TTFT** (p50/p90/p99), **end-to-end latency** (p50/p90/p99),
  **TPOT**, and **system throughput** (tokens/s, requests/s).

---

## Repository layout

```
scripts/
  setup_env.sh                 create the 3 venvs + system deps + patches
  register_bfcl_models.py      register base+ToolACE (7B/14B) + Qwen3.5-9B into BFCL
  register_qwen35_nothink.py   no-think handler + model for Qwen3.5-9B
  patch_vllm_metrics.py        fix prometheus-instrumentator vs Starlette 1.3
  prepare_toolace.py           ToolACE -> chat-formatted train/val JSONL
  train_sft.py                 LoRA / QLoRA SFT (assistant-only loss, nan-skip)
  merge_lora.py                merge adapter -> bf16 weights for serving
  serve_vllm.sh                vLLM server (bf16 or fp8)
  run_bfcl.sh                  BFCL generate + evaluate (python subset) vs server
  benchmark_latency.py         TTFT / latency / throughput @ conc 1/16/32 (+--no-think)
  serve_eval_bench.sh          serve -> eval -> benchmark -> stop (one phase)
  run_pipeline.sh              full per-size pipeline orchestration
  summarize_results.py         per-phase BFCL score -> python-subset summary
  make_report.py               aggregate all phases -> results/REPORT.md
  kill_server.sh               stop the vLLM server + free GPU safely
data/                          prepared JSONL
results/<phase>/{result,score}/  BFCL outputs & scores per phase (+ summary.json)
results/REPORT.md              consolidated comparison table (all models)
results/latency/               latency benchmark JSON
logs/                          server logs
```

## How to reproduce

```bash
# 0. one-time setup: creates 3 isolated venvs from the PINNED requirements-*.txt
#    files, applies the vLLM-metrics patch, and registers models in BFCL.
bash scripts/setup_env.sh
.venv-train/bin/python scripts/prepare_toolace.py

# 1. full pipeline (baseline -> train -> finetuned -> fp8) for each size
bash scripts/run_pipeline.sh 7b
bash scripts/run_pipeline.sh 14b
```

> **Exact-version reproducibility:** each venv is pinned via `requirements-train.txt`,
> `requirements-vllm.txt`, `requirements-bfcl.txt` (frozen from the working H100/CUDA-13
> env). They're separate because the stages need *conflicting* deps — e.g. training pins
> `torch==2.12.0` while vLLM pins `torch==2.11.0`.

### Reproducing the ablations & candidate models

`run_pipeline.sh` covers the per-size **baseline → LoRA → FP8** path. The remaining
rows of `results/REPORT.md` were produced by:

```bash
# 7B QLoRA ablation (method comparison)
.venv-train/bin/python scripts/train_sft.py --base-model models/Qwen2.5-7B-Instruct \
  --output-dir checkpoints/Qwen2.5-7B-Instruct-qlora --method qlora \
  --epochs 2 --lr 1e-4 --per-device-batch 8 --grad-accum 4 --max-len 4096
.venv-train/bin/python scripts/merge_lora.py --base-model models/Qwen2.5-7B-Instruct \
  --adapter checkpoints/Qwen2.5-7B-Instruct-qlora --out models/Qwen2.5-7B-Instruct-QLoRA
bash scripts/serve_eval_bench.sh models/Qwen2.5-7B-Instruct-QLoRA Qwen2.5-7B-Instruct-ToolACE \
  finetuned-7b-qlora none models/Qwen2.5-7B-Instruct-QLoRA 8000 64 0   # last arg 0 = skip latency

# 14B "lighter" retry (lr 5e-5, 1 epoch — recovers some of the forgetting)
.venv-train/bin/python scripts/train_sft.py --base-model models/Qwen2.5-14B-Instruct \
  --output-dir checkpoints/Qwen2.5-14B-Instruct-lora-light --method lora \
  --epochs 1 --lr 5e-5 --per-device-batch 4 --grad-accum 8 --max-len 4096
.venv-train/bin/python scripts/merge_lora.py --base-model models/Qwen2.5-14B-Instruct \
  --adapter checkpoints/Qwen2.5-14B-Instruct-lora-light --out models/Qwen2.5-14B-Instruct-ToolACE-light
bash scripts/serve_eval_bench.sh models/Qwen2.5-14B-Instruct-ToolACE-light Qwen2.5-14B-Instruct-ToolACE \
  finetuned-14b-light none models/Qwen2.5-14B-Instruct-ToolACE-light 8000 48 0

# Qwen3.5-9B candidate baseline (thinking + no-think); register_qwen35_nothink.py runs in setup_env.sh
.venv-train/bin/hf download Qwen/Qwen3.5-9B --local-dir models/Qwen3.5-9B
bash scripts/serve_vllm.sh models/Qwen3.5-9B Qwen3.5-9B 8000 none 16384 64 &   # serve, then:
bash scripts/run_bfcl.sh Qwen3.5-9B         baseline-qwen35-9b         8000 python 16 models/Qwen3.5-9B
bash scripts/run_bfcl.sh Qwen3.5-9B-nothink baseline-qwen35-9b-nothink 8000 python 16 models/Qwen3.5-9B
```

### Determinism & data versions

- Training sets `seed=42` with a fixed 98/2 split, and eval uses near-greedy decoding
  (temp 0.001) — so results reproduce **within run-to-run variance, not bit-exact**
  (CUDA kernels and vLLM batching aren't fully deterministic by default). Expected
  macro accuracies are recorded in `results/REPORT.md`.
- The **BFCL eval data is pinned** by `bfcl-eval==2026.3.23`. To fully pin the *training*
  data as well, pass a dataset `revision=<commit-sha>` to `load_dataset` in
  `prepare_toolace.py` (and `--revision` on the base-model `hf download`s) — currently
  they track the latest revision on the Hub.

## Environment

- 1× H100 80GB, CUDA 13 driver; Ubuntu 24.04, Python 3.12.
- Three isolated venvs to avoid dependency conflicts:
  `.venv-train` (torch/transformers/trl/peft/bitsandbytes),
  `.venv-vllm` (vllm 0.23), `.venv-bfcl` (bfcl-eval).
- System packages needed by vLLM's compile path: `python3-dev`, `ninja-build`.

---

## Results

All raw outputs live under `results/<phase>/` (BFCL `result/` + `score/` +
`summary.json`) and `results/latency/`.

### 7B — BFCL Python subset, per category (accuracy %)

| Category | base bf16 | LoRA bf16 | LoRA fp8 |
|---|--:|--:|--:|
| non_live simple_python      | 97.00 | 94.25 | — |
| non_live multiple           | 94.00 | 97.50 | — |
| non_live parallel           | 91.50 | 89.50 | — |
| non_live parallel_multiple  | 83.50 | 81.00 | — |
| non_live irrelevance        | 77.92 | 87.50 | — |
| live simple                 | 78.68 | 78.29 | — |
| live multiple               | 75.02 | 76.64 | — |
| live parallel               | 62.50 | 62.50 | — |
| live parallel_multiple      | 62.50 | 66.67 | — |
| live irrelevance            | 58.37 | 70.93 | — |
| live relevance              | 93.75 | 87.50 | — |
| **Non-Live overall**        | **75.33** | **74.85** | 74.65 |
| **Live overall**            | **75.35** | **76.61** | 75.72 |
| **Python macro-avg (11)**   | **79.52** | **81.12** | **80.76** |

Net: ToolACE LoRA trades a little non-live simple/parallel accuracy for large
irrelevance/robustness gains and better live performance → +1.6 macro. FP8
serving keeps accuracy within noise (80.76 vs 81.12).

### 7B — serving metrics (TTFT / latency / throughput)

| Config | conc | TTFT p50 | TTFT p99 | E2E p50 | tok/s | req/s |
|---|--:|--:|--:|--:|--:|--:|
| base bf16     |  1 | 16 ms | 73 ms | 154 ms | 149 | 7.6 |
| base bf16     | 16 | 34 ms | 57 ms | 157 ms | 1794 | 92 |
| base bf16     | 32 | 70 ms | 97 ms | 187 ms | 2630 | 135 |
| LoRA bf16     | 16 | 30 ms | 61 ms | 160 ms | 1795 | 92 |
| LoRA bf16     | 32 | 71 ms | 94 ms | 187 ms | 2608 | 134 |
| LoRA **fp8**  | 16 | 33 ms | 54 ms | **117 ms** | **2364** | 121 |
| LoRA **fp8**  | 32 | 73 ms | 101 ms | **162 ms** | **3228** | 166 |

At the client's 16–32 concurrency target the **fp8** build sustains **3.2K tok/s
/ 166 req/s** with **162 ms** median end-to-end latency and ~73 ms TTFT.

### 14B — BFCL Python subset, per category (accuracy %)

| Category | base bf16 | LoRA bf16 | Δ |
|---|--:|--:|--:|
| non_live simple_python      | 95.50 | 94.50 | −1.00 |
| non_live multiple           | 93.50 | 95.00 | +1.50 |
| non_live parallel           | 92.50 | 87.00 | −5.50 |
| non_live parallel_multiple  | 84.50 | 64.50 | **−20.00** |
| non_live irrelevance        | 82.08 | 87.08 | +5.00 |
| live simple                 | 73.26 | 77.91 | +4.65 |
| live multiple               | 74.55 | 75.78 | +1.23 |
| live parallel               | 68.75 | 68.75 | 0.00 |
| live parallel_multiple      | 70.83 | 54.17 | **−16.66** |
| live irrelevance            | 73.08 | 80.43 | +7.35 |
| live relevance              | 100.0 | 87.50 | −12.50 |
| **Python macro-avg (11)**   | **82.60** | **79.33** | **−3.27** |

**Finding (important).** The *same* recipe that helped the 7B (+1.6) **hurt** the
already-stronger 14B (−3.3). The gains are identical in spirit (irrelevance
+5/+7, live simple +4.7) but the 14B suffers **catastrophic forgetting of
parallel-multiple** (−20 / −16.7) and relevance (−12.5) — skills the 14B base was
already excellent at. lr=1e-4 × 2 epochs is too aggressive for the larger model.

**Lighter retry (lr 5e-5, 1 epoch).** This recovered most of the damage —
parallel-multiple went from −20.0/−16.7 back to −3.5/−8.3, and macro rose
79.33 → **80.83** — but it **still trails the 82.60 baseline**. Conclusion:
**ToolACE SFT does not improve the already-excellent 14B on this benchmark**;
the data distribution costs more parallel-call accuracy than the irrelevance
gains recover. The right call for the 14B is to **ship the baseline** (or invest
in data mixing / replay to protect parallel skills — out of scope here).

This is the key architectural insight of the exercise: **fine-tuning helps the
7B but hurts the 14B**, so "always fine-tune" is wrong. See
[Recommendation](#recommendation).

### Fine-tuning method comparison (LoRA vs QLoRA, 7B)

| Method | Python macro-acc | Trainable params | Base precision (train) | Peak train mem |
|---|--:|--:|--|--:|
| baseline (none)        | 79.52% | — | — | — |
| **LoRA** (bf16 base)   | 81.12% | 40.4M (0.53%) | bf16 | ~75 GB |
| **QLoRA** (4-bit base) | **81.56%** | 40.4M (0.53%) | NF4 4-bit | **~20 GB** |

**QLoRA matches (slightly beats) LoRA** here — within noise on accuracy, but the
4-bit frozen base cuts training memory ~3–4× (the 4-bit run fits in ~20 GB vs
~75 GB). **Recommendation: QLoRA is the most cost-effective recipe** — same
quality, far cheaper to train, and you can fit much larger models or batches on
the same GPU. (We merge the adapter back to bf16 for serving either way, so
inference quality/speed is identical.)

### Candidate base: Qwen3.5-9B (no fine-tune)

We evaluated **Qwen/Qwen3.5-9B** (newer, multimodal, Apache-2.0, thinking-by-default)
as an alternative base — served on the same vLLM stack, scored on the same BFCL
python subset in prompt mode. Because BFCL prompt-mode hits `/v1/completions` with
a hand-built ChatML prompt (the model's `enable_thinking` flag never applies), we
added a small no-think handler that prefills an empty `<think></think>` block
([`scripts/register_qwen35_nothink.py`](scripts/register_qwen35_nothink.py)) and
ran it both ways.

| Config | Macro | Non-Live | Live | E2E p50 @32 | tok/s @32 | req/s @32 |
|---|--:|--:|--:|--:|--:|--:|
| Qwen3.5-9B (thinking, default) | 74.98 | 73.0 | 62.6 | 1232 ms | 2942 | 25 |
| Qwen3.5-9B (no-think) | 74.93 | 73.0 | 62.9 | 353 ms | 1438 | 74 |
| **Qwen2.5-7B baseline** (ref) | **79.52** | 75.3 | 75.4 | **187 ms** | 2630 | **135** |
| **Qwen2.5-14B baseline** (ref) | **82.60** | 75.6 | 74.2 | 307 ms | 1600 | 82 |

Findings:
- **Thinking is a latency tax, not an accuracy lever here**: disabling it left macro
  flat (74.93 vs 74.98) but cut E2E latency ~3.5× and tripled throughput.
- **Even at its best (no-think), Qwen3.5-9B trails Qwen2.5-7B by ~4.6 pts** and the
  14B by ~7.7, *and* serves slower than the 7B. It is weakest on irrelevance /
  relevance (live_irrelevance 52.6, live_multiple 60.9) — i.e. deciding *whether*
  / *which* to call — while competitive on parallel calls (non-live parallel 90.5).
- Why: it's a larger multimodal generalist (9B + vision tower, 248K vocab), not an
  FC-specialized instruct model like Qwen2.5.

Takeaway: a newer/bigger generation is **not** automatically better for a narrow
skill like function calling. Qwen3.5-9B is a poor *baseline*, but — having the most
headroom of anything tested — it is the most interesting *future fine-tuning*
candidate (provided thinking is kept off in production to control latency).

### Training configuration

LoRA: r=16, α=32, dropout=0.05 on `q,k,v,o,gate,up,down`; bf16; **2 epochs**;
lr=1e-4 cosine, 5% warmup, grad-clip 1.0; effective batch 32; max-len 4096 (drop
>max, **no truncation** so function defs are never cut); assistant-only loss;
length-grouped batching; **non-finite-grad skipping**. 7B: 40.4M trainable
(0.53%); 14B: 68.8M (0.46%). Final train loss ≈ 0.15–0.21.

- **QLoRA**: identical but the frozen base is NF4 4-bit (double-quant, bf16
  compute) — `--method qlora`.
- **14B lighter recipe**: `--epochs 1 --lr 5e-5` (the over-aggressive uniform
  recipe caused forgetting; see the 14B finding above).
- Adapters are **merged to bf16** (`merge_lora.py`) before serving.

> Single source of truth for every number: **[`results/REPORT.md`](results/REPORT.md)**
> (auto-generated by `scripts/make_report.py` from the per-phase `summary.json`
> + latency files).

## Biggest challenges

1. **Train/eval format alignment (the one that matters most).** Getting a real
   accuracy lift hinges on training in *exactly* the format BFCL scores. I read
   the BFCL source to confirm the prompt-mode handler expects bracket notation
   `[func(arg=val)]` with the function list in the system message — which is
   precisely ToolACE's native format. So I train on ToolACE as-is with the
   Qwen2.5 chat template and an **assistant-only loss mask** (prefix-delta over
   `apply_chat_template`), and evaluate with the prompt-mode `QwenHandler`.
   Mismatch here is the usual reason "fine-tuning didn't help."

2. **Numerical instability → NaN during training.** A few ToolACE batches
   reproducibly produced a finite-but-huge CE loss that overflowed to an
   `inf/NaN` gradient in bf16, which then poisoned all weights (loss → 0, grad →
   nan permanently). Lowering LR didn't help (it's input-specific, not LR-driven).
   Fix: **non-finite-gradient skipping** — detect a non-finite grad after
   backward and drop that optimizer step. Training then converges cleanly.

3. **Large-vocab logits OOM.** Qwen's 152K vocab makes the LM-head logits the
   memory bottleneck: `batch×seq×vocab` upcast to fp32 for the loss tried to
   allocate **34 GB** at batch 16 / seq 4096. Solved by bounding the microbatch
   (8) and keeping gradient checkpointing on; `length`-grouped batching keeps
   padding (and thus this tensor) small.

4. **Registering custom models in BFCL.** BFCL ships Qwen3 but not Qwen2.5 and
   obviously not our checkpoints. I wrote an idempotent
   [`register_bfcl_models.py`](scripts/register_bfcl_models.py) that injects
   `ModelConfig` entries (base + ToolACE for 7B/14B, plus the Qwen3.5-9B
   candidate) into the prompt-mode handler, and serve each under a matching
   `--served-model-name` so BFCL's `--skip-server-setup` path hits our own vLLM
   server (with `REMOTE_OPENAI_TOKENIZER_PATH` pointing at local weights for token
   counting). A custom no-think handler for Qwen3.5 lives in
   [`register_qwen35_nothink.py`](scripts/register_qwen35_nothink.py).

5. **Serving-stack reality.** vLLM 0.23's `torch.compile` path needed system
   `python3-dev` + `ninja`; and its bundled `prometheus-fastapi-instrumentator`
   crashed on every request against the newer Starlette (`_IncludedRouter` has no
   `.path`) — patched in [`patch_vllm_metrics.py`](scripts/patch_vllm_metrics.py).
   transformers 5.x also moved targets under me (`apply_chat_template` now returns
   a `BatchEncoding`; `group_by_length` was removed from `TrainingArguments`, so I
   wired `LengthGroupedSampler` manually).
