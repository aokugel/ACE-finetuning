#!/usr/bin/env python
"""Merge a PEFT LoRA/QLoRA adapter into the base model and save fp16/bf16
weights ready for vLLM serving.

For QLoRA adapters we reload the base in bf16 (NOT 4-bit) before merging, so
the merged weights are full-precision -- this is the standard, accuracy-safe
way to deploy a QLoRA-trained adapter.
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"Loading base {args.base_model} in bf16 ...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="cpu")
    print(f"Applying adapter {args.adapter} ...")
    model = PeftModel.from_pretrained(base, args.adapter)
    print("Merging ...")
    model = model.merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base_model).save_pretrained(args.out)
    print(f"Merged model saved to {args.out}")


if __name__ == "__main__":
    main()
