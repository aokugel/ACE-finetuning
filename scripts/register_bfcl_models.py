#!/usr/bin/env python
"""Register our Qwen2.5 base + ToolACE-finetuned models into the installed
bfcl_eval model registry.

BFCL ships Qwen3 entries but no Qwen2.5, and obviously no entry for our
finetuned checkpoints. Rather than fork BFCL, we idempotently inject four
ModelConfig entries into `local_inference_model_map` inside the installed
package. Each model is registered in PROMPT mode (QwenHandler), because:

  * ToolACE's training format (system prompt listing JSON functions +
    assistant turns in `[func(arg=val)]` bracket notation) is identical to
    BFCL's prompt-mode (`classic` style) format. Prompt mode is therefore the
    apples-to-apples way to measure the lift from ToolACE finetuning.

The registry KEY (what you pass to `bfcl --model`) equals `model_name` (what
BFCL sends as the `model` field to the OpenAI-compatible server). So when you
serve a checkpoint with vLLM you must pass
`--served-model-name <that exact string>`.

Run once after every `pip install bfcl-eval`. Idempotent.
"""
import re
import sys
import bfcl_eval
from pathlib import Path

CONFIG = Path(bfcl_eval.__file__).parent / "constants" / "model_config.py"

# (registry_key == served model_name, display_name, hf_url)
MODELS = [
    ("Qwen2.5-7B-Instruct",          "Qwen2.5-7B-Instruct (Prompt)",          "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct"),
    ("Qwen2.5-14B-Instruct",         "Qwen2.5-14B-Instruct (Prompt)",         "https://huggingface.co/Qwen/Qwen2.5-14B-Instruct"),
    ("Qwen2.5-7B-Instruct-ToolACE",  "Qwen2.5-7B-Instruct ToolACE (Prompt)",  "local-finetune"),
    ("Qwen2.5-14B-Instruct-ToolACE", "Qwen2.5-14B-Instruct ToolACE (Prompt)", "local-finetune"),
    # Additional candidate base (no fine-tune) — newer/stronger generation, evaluated for comparison.
    ("Qwen3.5-9B",                   "Qwen3.5-9B (Prompt)",                   "https://huggingface.co/Qwen/Qwen3.5-9B"),
]

ENTRY_TMPL = '''    "{key}": ModelConfig(
        model_name="{key}",
        display_name="{display}",
        url="{url}",
        org="Qwen",
        license="apache-2.0",
        model_handler=QwenHandler,
        input_price=None,
        output_price=None,
        is_fc_model=False,
        underscore_to_dot=False,
    ),
'''


def main():
    src = CONFIG.read_text()
    anchor = "local_inference_model_map = {\n"
    if anchor not in src:
        sys.exit(f"Could not find anchor in {CONFIG}; BFCL layout changed.")

    block = ""
    for key, display, url in MODELS:
        if f'"{key}": ModelConfig(' in src:
            print(f"[skip] {key} already registered")
            continue
        block += ENTRY_TMPL.format(key=key, display=display, url=url)
        print(f"[add ] {key}")

    if not block:
        print("Nothing to do; all models already registered.")
        return

    src = src.replace(anchor, anchor + block, 1)
    CONFIG.write_text(src)
    print(f"Patched {CONFIG}")

    # sanity: import the mapping fresh in a subprocess-free way
    import importlib
    importlib.reload(bfcl_eval)
    print("Done.")


if __name__ == "__main__":
    main()
