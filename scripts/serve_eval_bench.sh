#!/usr/bin/env bash
# Serve a model, run the BFCL python-subset eval + latency benchmark, then stop.
# Encapsulates the full server lifecycle so each eval phase is one command.
#
#   scripts/serve_eval_bench.sh <model_path> <served_name> <phase> <quant> <tokenizer_path> [port] [maxseq] [bench_lat]
#
#   quant         none | fp8
#   bench_lat     1 to run latency benchmark, 0 to skip (default 1)
set -euo pipefail
MODEL_PATH="${1:?model path}"
SERVED="${2:?served name}"
PHASE="${3:?phase dir}"
QUANT="${4:-none}"
TOKPATH="${5:?tokenizer path}"
PORT="${6:-8000}"
MAXSEQ="${7:-64}"
BENCH="${8:-1}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

bash scripts/kill_server.sh   # ensure GPU is free
nohup bash scripts/serve_vllm.sh "$MODEL_PATH" "$SERVED" "$PORT" "$QUANT" 16384 "$MAXSEQ" \
  > "logs/serve_${PHASE}.log" 2>&1 &

echo ">>> waiting for server ($SERVED, quant=$QUANT) ..."
ready=0
for _ in $(seq 1 250); do
  if curl -s "http://localhost:${PORT}/v1/models" 2>/dev/null | grep -q "$SERVED"; then ready=1; break; fi
  if grep -qiE "EngineCore failed|RuntimeError: Engine|less than desired|Address already" "logs/serve_${PHASE}.log" 2>/dev/null; then
    echo "SERVER FAILED:"; grep -iE "error|less than desired" "logs/serve_${PHASE}.log" | tail -5; bash scripts/kill_server.sh; exit 1
  fi
  sleep 4
done
[ "$ready" = 1 ] || { echo "server timeout"; bash scripts/kill_server.sh; exit 1; }
echo ">>> server ready."

scripts/run_bfcl.sh "$SERVED" "$PHASE" "$PORT" python 16 "$TOKPATH"

if [ "$BENCH" = "1" ]; then
  .venv-vllm/bin/python scripts/benchmark_latency.py --model "$SERVED" --port "$PORT" \
    --concurrency 1 16 32 --requests-per-level 96 --out "results/latency/${PHASE}.json"
fi

bash scripts/kill_server.sh
.venv-bfcl/bin/python scripts/summarize_results.py "results/${PHASE}/score" --json "results/${PHASE}/summary.json"
echo "===PHASE_${PHASE}_DONE==="
