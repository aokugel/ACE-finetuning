#!/usr/bin/env python
"""LoRA / QLoRA SFT for tool-calling on ToolACE.

Design choices (best practices applied):
  * Chat-template tokenization that EXACTLY matches BFCL prompt-mode eval
    (Qwen2.5 ChatML; system message carries the JSON function list).
  * Multi-turn assistant-only loss: every non-assistant token is masked to
    -100 so we only learn the model's tool-call / answer turns, not the user
    questions or tool outputs.
  * LoRA on all attention + MLP projection matrices (q,k,v,o,gate,up,down).
  * QLoRA = same, but the frozen base is loaded in 4-bit NF4 (double quant,
    bf16 compute) -> fits 14B comfortably and trains cheaply.
  * bf16, gradient checkpointing, cosine schedule w/ warmup, packing off.

Outputs a PEFT adapter dir. Merge with scripts/merge_lora.py before serving.
"""
import argparse
import json
import os

import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, Trainer, TrainingArguments)
from transformers.trainer_pt_utils import LengthGroupedSampler
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


class LengthGroupedTrainer(Trainer):
    """Trainer with two robustness features:

    1. Length-grouped batching (transformers 5.x dropped the `group_by_length`
       flag) via the still-shipped LengthGroupedSampler -> far less padding.
    2. Non-finite gradient skipping: a few ToolACE examples reproducibly produce
       an inf/NaN gradient in bf16 backward (finite but huge CE loss -> overflow),
       which would poison all weights. We detect non-finite grads after backward
       and drop that accumulation window instead of stepping the optimizer.
    """
    def _get_train_sampler(self, *args, **kwargs):
        lengths = self.train_dataset["length"]
        return LengthGroupedSampler(
            batch_size=self.args.train_batch_size,
            dataset=self.train_dataset,
            lengths=lengths,
        )

    def training_step(self, model, inputs, *args, **kwargs):
        loss = super().training_step(model, inputs, *args, **kwargs)
        if not torch.isfinite(loss):
            model.zero_grad(set_to_none=True)
            self._nan_skips = getattr(self, "_nan_skips", 0) + 1
            return torch.zeros_like(loss)
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                model.zero_grad(set_to_none=True)
                self._nan_skips = getattr(self, "_nan_skips", 0) + 1
                return torch.zeros_like(loss)
        return loss


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--train-file", default="data/toolace_train.jsonl")
    ap.add_argument("--val-file", default="data/toolace_val.jsonl")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--method", choices=["lora", "qlora"], default="lora")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--per-device-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--max-grad-norm", type=float, default=0.3)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=-1, help="override for smoke tests")
    ap.add_argument("--grad-checkpointing", type=int, default=1,
                    help="1=on (saves memory, ~1.3x slower), 0=off (faster if it fits)")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


def build_tokenize_fn(tokenizer, max_len):
    """Return a function mapping {'messages': [...]} -> {input_ids, labels}.

    Uses the prefix-delta trick: render messages[:i+1] and diff against the
    previous render to get the tokens contributed by message i, unmasking only
    assistant contributions.
    """
    def fn(example):
        messages = example["messages"]
        input_ids, labels = [], []
        prev = []
        for i, msg in enumerate(messages):
            enc = tokenizer.apply_chat_template(
                messages[: i + 1], tokenize=True, add_generation_prompt=False,
                return_dict=True)
            ids = enc["input_ids"]
            if ids and isinstance(ids[0], list):  # de-batch if nested
                ids = ids[0]
            delta = ids[len(prev):]
            if msg["role"] == "assistant":
                labels += delta
            else:
                labels += [-100] * len(delta)
            prev = ids
        input_ids = prev
        # NOTE: we do NOT truncate. Left-truncation would cut the function
        # definitions out of the system prompt (teaching the model to call
        # undefined functions); right-truncation would cut the assistant label.
        # Over-length examples are dropped downstream via the `length` filter.
        return {"input_ids": input_ids, "labels": labels,
                "length": len(input_ids),
                "n_label_tokens": sum(1 for x in labels if x != -100)}
    return fn


class DataCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, features):
        maxlen = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attn = [], [], []
        for f in features:
            ids, lab = f["input_ids"], f["labels"]
            pad = maxlen - len(ids)
            input_ids.append(ids + [self.pad_id] * pad)
            labels.append(lab + [-100] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = None
    model_kwargs = dict(torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    if args.method == "qlora":
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs["quantization_config"] = quant_cfg

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    model.config.use_cache = False
    gc_on = bool(args.grad_checkpointing)
    if args.method == "qlora":
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gc_on)

    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM", target_modules=TARGET_MODULES)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ---- data ----
    tok_fn = build_tokenize_fn(tokenizer, args.max_len)
    data_files = {"train": args.train_file, "validation": args.val_file}
    raw = load_dataset("json", data_files=data_files)
    cols = raw["train"].column_names
    ds = raw.map(tok_fn, remove_columns=cols, num_proc=8,
                 desc="tokenize+mask")
    # drop examples with no trainable tokens
    n_before = {k: len(v) for k, v in ds.items()}
    # keep examples that have trainable tokens AND fit in max_len (drop, don't truncate)
    ds = ds.filter(lambda x: x["n_label_tokens"] > 0 and x["length"] <= args.max_len)
    ds = ds.remove_columns(["n_label_tokens"])
    print("rows before filter:", n_before, "after:", {k: len(v) for k, v in ds.items()})
    import numpy as np
    tl = np.array(ds["train"]["length"])
    print(f"train token lengths: mean={tl.mean():.0f} p95={np.percentile(tl,95):.0f} max={tl.max()}")

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch,
        per_device_eval_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        weight_decay=0.0,
        bf16=True,
        gradient_checkpointing=gc_on,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,   # keep 'length' + let our collator handle cols
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=1,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=4,
    )

    trainer = LengthGroupedTrainer(
        model=model, args=targs,
        train_dataset=ds["train"], eval_dataset=ds["validation"],
        data_collator=DataCollator(tokenizer.pad_token_id),
    )
    trainer.train()
    trainer.save_model(args.output_dir)  # saves adapter
    tokenizer.save_pretrained(args.output_dir)

    with open(os.path.join(args.output_dir, "train_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
