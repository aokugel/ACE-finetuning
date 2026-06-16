#!/usr/bin/env python
"""Register a NO-THINKING variant of Qwen3.5-9B for BFCL prompt-mode.

Qwen3.5 reasons by default, which (a) inflates serving latency ~6x and (b) appears
to hurt relevance/irrelevance accuracy. BFCL prompt-mode talks to /v1/completions
with a manually-built ChatML prompt, so the model's chat-template
`enable_thinking` flag never applies. We replicate Qwen3's official no-think
mechanism by prefilling an empty `<think>\n\n</think>\n\n` block right after the
assistant tag, so the model emits the answer directly.

Adds (idempotently):
  * handler module bfcl_eval/.../local_inference/qwen35_nothink.py
  * a "Qwen3.5-9B-nothink" ModelConfig (served name is still "Qwen3.5-9B", so the
    same running vLLM server is reused).
"""
import bfcl_eval
from pathlib import Path

PKG = Path(bfcl_eval.__file__).parent
HANDLER = PKG / "model_handler" / "local_inference" / "qwen35_nothink.py"
CONFIG = PKG / "constants" / "model_config.py"

HANDLER_SRC = '''from bfcl_eval.model_handler.local_inference.qwen import QwenHandler
from overrides import override


class Qwen35NoThinkHandler(QwenHandler):
    """Qwen3.5 in non-thinking mode. BFCL prompt-mode bypasses the chat template,
    so we prefill an empty think block (equivalent to enable_thinking=False) to
    make the model answer directly instead of emitting a reasoning trace."""

    @override
    def _format_prompt(self, messages, function):
        prompt = super()._format_prompt(messages, function)
        return prompt + "<think>\\n\\n</think>\\n\\n"
'''

ENTRY = '''    "Qwen3.5-9B-nothink": ModelConfig(
        model_name="Qwen3.5-9B",
        display_name="Qwen3.5-9B (Prompt no-think)",
        url="https://huggingface.co/Qwen/Qwen3.5-9B",
        org="Qwen",
        license="apache-2.0",
        model_handler=Qwen35NoThinkHandler,
        input_price=None,
        output_price=None,
        is_fc_model=False,
        underscore_to_dot=False,
    ),
'''


def main():
    HANDLER.write_text(HANDLER_SRC)
    print(f"wrote handler -> {HANDLER}")

    src = CONFIG.read_text()
    imp = "from bfcl_eval.model_handler.local_inference.qwen35_nothink import Qwen35NoThinkHandler\n"
    if imp not in src:
        anchor = "from bfcl_eval.model_handler.local_inference.qwen import QwenHandler\n"
        if anchor not in src:
            raise SystemExit("QwenHandler import anchor not found; BFCL layout changed")
        src = src.replace(anchor, anchor + imp, 1)
        print("added import")
    if '"Qwen3.5-9B-nothink": ModelConfig(' not in src:
        anchor = "local_inference_model_map = {\n"
        src = src.replace(anchor, anchor + ENTRY, 1)
        print("added Qwen3.5-9B-nothink ModelConfig")
    else:
        print("ModelConfig already present")
    CONFIG.write_text(src)
    print("done")


if __name__ == "__main__":
    main()
