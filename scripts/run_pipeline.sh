#!/usr/bin/env bash
# End-to-end reproducible pipeline for ONE model size.
#
#   scripts/run_pipeline.sh <size>            # size = 7b | 14b
#
# Stages:
#   1. baseline: serve base model (bf16) -> BFCL python eval -> latency bench
#   2. train:    LoRA SFT on ToolACE -> merge adapter
#   3. finetuned: serve merged model (bf16) -> BFCL python eval -> latency bench
#   4. optimized: serve merged model (fp8)  -> BFCL python eval -> latency bench
#
# Each serving stage runs in the background, waits for /v1/models, evaluates,
# then is shut down before the next GPU-heavy stage. Driven sequentially so the
# single H100 is never double-booked.
set -euo pipefail
SIZE="${1:?usage: run_pipeline.sh <7b|14b>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

case "$SIZE" in
  # 14B needs a smaller microbatch: the 152K-vocab logits tensor (batch*seq*vocab,
  # fp32 for the loss) is the memory bottleneck, not the weights.
  7b)  BASE="models/Qwen2.5-7B-Instruct";  KEY="Qwen2.5-7B-Instruct";  MAXSEQ=64; PDB=8; GA=4 ;;
  14b) BASE="models/Qwen2.5-14B-Instruct"; KEY="Qwen2.5-14B-Instruct"; MAXSEQ=48; PDB=4; GA=8 ;;
  *) echo "size must be 7b or 14b"; exit 1 ;;
esac
FTKEY="${KEY}-ToolACE"
ADAPTER="checkpoints/${KEY}-lora"
MERGED="models/${FTKEY}"
PORT=8000

wait_ready() {  # $1 = served name
  for _ in $(seq 1 200); do
    curl -s "http://localhost:${PORT}/v1/models" 2>/dev/null | grep -q "$1" && return 0
    sleep 4
  done
  echo "server did not become ready"; return 1
}
start_server() { # $1 path  $2 served  $3 quant
  nohup bash scripts/serve_vllm.sh "$1" "$2" "$PORT" "$3" 16384 "$MAXSEQ" \
    > "logs/serve_${2}_${3}.log" 2>&1 &
  echo $! > /tmp/vllm_pipe.pid
  wait_ready "$2"
}
stop_server() { bash scripts/kill_server.sh; }

# ---------- 1. baseline ----------
start_server "$BASE" "$KEY" none
scripts/run_bfcl.sh "$KEY" "baseline-${SIZE}" "$PORT" python 16 "$BASE"
python scripts/benchmark_latency.py --model "$KEY" --port "$PORT" \
  --concurrency 1 16 32 --requests-per-level 96 --out "results/latency/${SIZE}_baseline_bf16.json"
stop_server

# ---------- 2. train + merge ----------
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/train_sft.py --base-model "$BASE" --output-dir "$ADAPTER" \
  --method lora --epochs 2 --lr 1e-4 --warmup-ratio 0.05 --max-grad-norm 1.0 \
  --per-device-batch "$PDB" --grad-accum "$GA" --max-len 4096 --grad-checkpointing 1
python scripts/merge_lora.py --base-model "$BASE" --adapter "$ADAPTER" --out "$MERGED"

# ---------- 3. finetuned (bf16) ----------
start_server "$MERGED" "$FTKEY" none
scripts/run_bfcl.sh "$FTKEY" "finetuned-${SIZE}" "$PORT" python 16 "$MERGED"
python scripts/benchmark_latency.py --model "$FTKEY" --port "$PORT" \
  --concurrency 1 16 32 --requests-per-level 96 --out "results/latency/${SIZE}_finetuned_bf16.json"
stop_server

# ---------- 4. optimized (fp8) ----------
start_server "$MERGED" "$FTKEY" fp8
scripts/run_bfcl.sh "$FTKEY" "finetuned-${SIZE}-fp8" "$PORT" python 16 "$MERGED"
python scripts/benchmark_latency.py --model "$FTKEY" --port "$PORT" \
  --concurrency 1 16 32 --requests-per-level 96 --out "results/latency/${SIZE}_finetuned_fp8.json"
stop_server

echo "PIPELINE DONE for $SIZE"
