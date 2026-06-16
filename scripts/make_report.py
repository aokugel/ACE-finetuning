#!/usr/bin/env python
"""Aggregate every results/<phase>/summary.json + results/latency/<phase>.json
into one consolidated markdown report (results/REPORT.md).

Run after the pipeline so the README numbers can be cross-checked against the
single source of truth.
"""
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# phase -> (pretty model, mode) ; order defines the report order
PHASES = [
    ("baseline-7b",       "Qwen2.5-7B-Instruct",  "baseline bf16"),
    ("finetuned-7b",      "Qwen2.5-7B-Instruct",  "ToolACE LoRA bf16"),
    ("finetuned-7b-fp8",  "Qwen2.5-7B-Instruct",  "ToolACE LoRA fp8"),
    ("finetuned-7b-qlora","Qwen2.5-7B-Instruct",  "ToolACE QLoRA bf16"),
    ("baseline-qwen35-9b","Qwen3.5-9B",           "baseline bf16 (thinking)"),
    ("baseline-qwen35-9b-nothink","Qwen3.5-9B",   "baseline bf16 (no-think)"),
    ("baseline-14b",      "Qwen2.5-14B-Instruct", "baseline bf16"),
    ("finetuned-14b",     "Qwen2.5-14B-Instruct", "ToolACE LoRA bf16 (uniform)"),
    ("finetuned-14b-fp8", "Qwen2.5-14B-Instruct", "ToolACE LoRA fp8"),
    ("finetuned-14b-light","Qwen2.5-14B-Instruct","ToolACE LoRA bf16 (lighter)"),
]


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def lat_at(latency, conc):
    if not latency:
        return None
    for L in latency["levels"]:
        if L["concurrency"] == conc:
            return L
    return None


def main():
    rows = []
    for phase, model, mode in PHASES:
        s = load(f"{ROOT}/results/{phase}/summary.json")
        if not s:
            continue
        lat = load(f"{ROOT}/results/latency/{phase}.json")
        l32 = lat_at(lat, 32)
        rows.append({
            "model": model, "mode": mode, "phase": phase,
            "macro": s.get("python_macro_avg"),
            "non_live": s.get("non_live_overall"),
            "live": s.get("live_overall"),
            "ttft32": l32["ttft_ms"]["p50"] if l32 else None,
            "e2e32": l32["e2e_ms"]["p50"] if l32 else None,
            "tput32": l32["output_tok_per_s"] if l32 else None,
            "rps32": l32["requests_per_s"] if l32 else None,
        })

    lines = ["# Consolidated Results\n",
             "BFCL Python-subset macro accuracy + serving metrics @ 32 concurrency.\n",
             "| Model | Mode | Macro % | Non-Live % | Live % | TTFT p50 | E2E p50 | tok/s | req/s |",
             "|---|---|--:|--:|--:|--:|--:|--:|--:|"]
    for r in rows:
        def f(x, suf=""):
            return f"{x}{suf}" if x is not None else "—"
        lines.append(
            f"| {r['model']} | {r['mode']} | {f(r['macro'])} | {f(r['non_live'])} | "
            f"{f(r['live'])} | {f(r['ttft32'],' ms')} | {f(r['e2e32'],' ms')} | "
            f"{f(r['tput32'])} | {f(r['rps32'])} |")
    report = "\n".join(lines) + "\n"
    out = f"{ROOT}/results/REPORT.md"
    open(out, "w").write(report)
    print(report)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
