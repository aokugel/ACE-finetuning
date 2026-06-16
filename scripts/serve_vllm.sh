#!/usr/bin/env bash
# Serve a model with vLLM's OpenAI-compatible server (production-style).
#
# Usage:
#   scripts/serve_vllm.sh <model_path> <served_name> [port] [quant] [max_model_len] [max_num_seqs]
#
#   quant: none | fp8   (fp8 = on-the-fly W8A8 FP8 on H100 -> the "optimized" deployment)
#
# Example (baseline bf16):
#   scripts/serve_vllm.sh models/Qwen2.5-7B-Instruct Qwen2.5-7B-Instruct 8000 none
# Example (optimized fp8):
#   scripts/serve_vllm.sh models/Qwen2.5-7B-Instruct Qwen2.5-7B-Instruct 8000 fp8
set -euo pipefail

MODEL_PATH="${1:?model path}"
SERVED_NAME="${2:?served model name (must match BFCL registry model_name)}"
PORT="${3:-8000}"
QUANT="${4:-none}"
MAX_LEN="${5:-16384}"
MAX_SEQS="${6:-64}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VLLM="$ROOT/.venv-vllm/bin/vllm"

QUANT_ARG=()
if [[ "$QUANT" == "fp8" ]]; then
  QUANT_ARG=(--quantization fp8)
fi

echo ">>> serving $MODEL_PATH as '$SERVED_NAME' on :$PORT (quant=$QUANT, max_len=$MAX_LEN, max_seqs=$MAX_SEQS)"
exec "$VLLM" serve "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 --port "$PORT" \
  --dtype bfloat16 \
  "${QUANT_ARG[@]}" \
  --max-model-len "$MAX_LEN" \
  --max-num-seqs "$MAX_SEQS" \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --disable-uvicorn-access-log
