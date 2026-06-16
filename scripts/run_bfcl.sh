#!/usr/bin/env bash
# Run BFCL generation + evaluation for a model that is ALREADY being served by
# vLLM (scripts/serve_vllm.sh) on the given port.
#
# Usage:
#   scripts/run_bfcl.sh <model_key> <phase_dir> [port] [test_category] [num_threads] [tokenizer_path]
#
#   model_key       registry key == vLLM --served-model-name (e.g. Qwen2.5-7B-Instruct)
#   phase_dir       subdir under results/ to hold result+score (e.g. baseline-7b)
#   test_category   default: python   (the 11-category BFCL Python subset)
#   tokenizer_path  local dir BFCL loads the tokenizer from (for token counting);
#                   the served model_key is still what's sent to the server.
#                   default: models/<model_key>
#
# Example:
#   scripts/run_bfcl.sh Qwen2.5-7B-Instruct baseline-7b 8000 python 16 models/Qwen2.5-7B-Instruct
set -euo pipefail

MODEL_KEY="${1:?model key}"
PHASE="${2:?phase dir name}"
PORT="${3:-8000}"
CATEGORY="${4:-python}"
THREADS="${5:-16}"
TOKENIZER_PATH="${6:-models/${MODEL_KEY}}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BFCL="$ROOT/.venv-bfcl/bin/bfcl"

export BFCL_PROJECT_ROOT="$ROOT"
export REMOTE_OPENAI_BASE_URL="http://localhost:${PORT}/v1"
export REMOTE_OPENAI_API_KEY="EMPTY"
# Load the tokenizer locally so BFCL doesn't try to fetch the served name from HF.
export REMOTE_OPENAI_TOKENIZER_PATH="$ROOT/${TOKENIZER_PATH}"

RES="results/${PHASE}/result"
SCORE="results/${PHASE}/score"
mkdir -p "$ROOT/$RES" "$ROOT/$SCORE"

echo ">>> [generate] model=$MODEL_KEY category=$CATEGORY threads=$THREADS -> $RES"
"$BFCL" generate \
  --model "$MODEL_KEY" \
  --test-category "$CATEGORY" \
  --backend vllm \
  --skip-server-setup \
  --temperature 0.001 \
  --num-threads "$THREADS" \
  --result-dir "$RES" \
  --allow-overwrite

echo ">>> [evaluate] model=$MODEL_KEY -> $SCORE"
"$BFCL" evaluate \
  --model "$MODEL_KEY" \
  --test-category "$CATEGORY" \
  --result-dir "$RES" \
  --score-dir "$SCORE"

echo ">>> overall scores:"
cat "$ROOT/$SCORE/data_overall.csv" 2>/dev/null || echo "(no data_overall.csv)"
