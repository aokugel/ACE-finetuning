#!/usr/bin/env bash
# Reproducible environment setup on a fresh 1xH100 box (Ubuntu 24.04, Python 3.12).
# Creates three isolated venvs so vLLM / training / BFCL deps never collide.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# --- system packages required by vLLM's torch.compile / inductor path ---
#   python3-dev  -> Python.h (compiling cuda_utils)
#   ninja-build  -> ninja (inductor backend)
sudo apt-get update -qq
sudo apt-get install -y python3-dev python3.12-dev ninja-build build-essential

# --- uv (fast installer) ---
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# --- 1) training venv ---
uv venv .venv-train --python 3.12
source .venv-train/bin/activate
uv pip install torch transformers trl peft datasets accelerate bitsandbytes \
               sentencepiece "huggingface_hub" hf_transfer
deactivate

# --- 2) vLLM serving venv ---
uv venv .venv-vllm --python 3.12
source .venv-vllm/bin/activate
uv pip install vllm
deactivate
# Patch prometheus-fastapi-instrumentator for Starlette 1.3.x (see script docstring)
.venv-vllm/bin/python scripts/patch_vllm_metrics.py

# --- 3) BFCL eval venv ---
uv venv .venv-bfcl --python 3.12
source .venv-bfcl/bin/activate
uv pip install bfcl-eval soundfile   # soundfile: transitive dep of qwen_agent import chain
deactivate
# Register our 4 models (base + ToolACE, 7B + 14B) into the BFCL registry
.venv-bfcl/bin/python scripts/register_bfcl_models.py

# --- model downloads ---
export HF_HUB_ENABLE_HF_TRANSFER=1
.venv-train/bin/hf download Qwen/Qwen2.5-7B-Instruct  --local-dir models/Qwen2.5-7B-Instruct
.venv-train/bin/hf download Qwen/Qwen2.5-14B-Instruct --local-dir models/Qwen2.5-14B-Instruct

echo "Environment ready."
