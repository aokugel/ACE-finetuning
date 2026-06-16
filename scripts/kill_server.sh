#!/usr/bin/env bash
# Stop any running vLLM server AND its engine subprocesses.
#
# Notes:
#  * vLLM spawns separate EngineCore processes that hold the GPU memory; killing
#    only the `vllm serve` parent leaks ~tens of GiB of VRAM. So we kill every
#    process currently using the GPU (the surest signal).
#  * The bracket trick `[v]llm` keeps this command from matching its own / the
#    harness wrapper's command line (a naive `pkill -f vllm` kills the caller).
# Free the GPU by killing whatever compute apps are on it.
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | xargs -r kill -9 2>/dev/null || true
# Belt-and-suspenders: kill the server parent if it lingers without GPU procs.
pkill -9 -f "[v]llm serve" 2>/dev/null || true
sleep 3
echo "vLLM stopped; GPU memory used: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
