#!/usr/bin/env python
"""Convert Team-ACE/ToolACE into a chat-formatted SFT dataset.

ToolACE rows:  {"system": <str with JSON function list>,
                "conversations": [{"from": "user|assistant|tool", "value": str}, ...]}

We emit JSONL rows of the form {"messages": [{"role","content"}, ...]} with the
system message first. Roles map directly (user->user, assistant->assistant,
tool->tool). This matches BFCL prompt-mode at eval time: the system message
lists the callable functions and the assistant replies in `[func(arg=val)]`
bracket notation; tool outputs are fed back as role="tool".

We keep the data essentially as-is (this is the documented ToolACE -> BFCL
recipe) and only apply light hygiene:
  * drop rows whose conversation has no assistant turn,
  * ensure the conversation starts with a user turn after system,
  * deterministic train/val split.
"""
import argparse
import json
import random
from datasets import load_dataset

ROLE_MAP = {"user": "user", "assistant": "assistant", "tool": "tool",
            "gpt": "assistant", "human": "user", "function": "tool"}


def build_messages(row):
    msgs = [{"role": "system", "content": row["system"].strip()}]
    for turn in row["conversations"]:
        role = ROLE_MAP.get(turn["from"].lower())
        if role is None:
            return None
        msgs.append({"role": role, "content": turn["value"]})
    # need at least system + user + assistant, and >=1 assistant turn
    if not any(m["role"] == "assistant" for m in msgs):
        return None
    if len(msgs) < 3:
        return None
    # first non-system turn should be user
    if msgs[1]["role"] != "user":
        return None
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-rows", type=int, default=None, help="cap for smoke tests")
    args = ap.parse_args()

    ds = load_dataset("Team-ACE/ToolACE", split="train")
    rows = []
    dropped = 0
    for row in ds:
        msgs = build_messages(row)
        if msgs is None:
            dropped += 1
            continue
        rows.append({"messages": msgs})
        if args.max_rows and len(rows) >= args.max_rows:
            break

    random.Random(args.seed).shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_frac))
    val, train = rows[:n_val], rows[n_val:]

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    for name, split in [("toolace_train.jsonl", train), ("toolace_val.jsonl", val)]:
        path = os.path.join(args.out_dir, name)
        with open(path, "w") as f:
            for r in split:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(split):>6} rows -> {path}")
    print(f"kept {len(rows)} / {len(ds)} rows (dropped {dropped})")


if __name__ == "__main__":
    main()
