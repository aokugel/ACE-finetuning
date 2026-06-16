#!/usr/bin/env python
"""Summarize a BFCL phase's scores into a clean, comparable Python-subset metric.

BFCL's `data_overall.csv` "Overall Acc" averages over ALL leaderboard sections
(multi-turn / web-search / memory) that we deliberately did not run, so it is
NOT a fair number for the Python subset. Instead we read the per-section CSVs and
report:
  * non_live_overall, live_overall  (BFCL's own section aggregates)
  * python_macro_avg                (equal-weight mean of the 11 python cats)
  * the full per-category table

Usage: python scripts/summarize_results.py results/baseline-7b/score [--json out.json]
"""
import argparse
import csv
import json
import os


def pf(x):
    if x is None or x == "N/A" or x == "":
        return None
    return float(x.strip().rstrip("%"))


def read_row(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("score_dir")
    ap.add_argument("--json")
    args = ap.parse_args()

    nl = read_row(os.path.join(args.score_dir, "data_non_live.csv"))
    lv = read_row(os.path.join(args.score_dir, "data_live.csv"))

    cats = {
        "non_live_simple_python":   pf(nl.get("Python Simple AST")),
        "non_live_multiple":        pf(nl.get("Multiple AST")),
        "non_live_parallel":        pf(nl.get("Parallel AST")),
        "non_live_parallel_mult":   pf(nl.get("Parallel Multiple AST")),
        "non_live_irrelevance":     pf(nl.get("Irrelevance Detection")),
        "live_simple":              pf(lv.get("Python Simple AST")),
        "live_multiple":            pf(lv.get("Python Multiple AST")),
        "live_parallel":            pf(lv.get("Python Parallel AST")),
        "live_parallel_mult":       pf(lv.get("Python Parallel Multiple AST")),
        "live_irrelevance":         pf(lv.get("Irrelevance Detection")),
        "live_relevance":           pf(lv.get("Relevance Detection")),
    }
    vals = [v for v in cats.values() if v is not None]
    summary = {
        "phase": os.path.basename(os.path.dirname(args.score_dir.rstrip("/"))) or args.score_dir,
        "non_live_overall": pf(nl.get("Non-Live Overall Acc")),
        "live_overall": pf(lv.get("Live Overall Acc")),
        "python_macro_avg": round(sum(vals) / len(vals), 2) if vals else None,
        "categories": cats,
    }

    print(f"== {args.score_dir} ==")
    print(f"  Non-Live overall : {summary['non_live_overall']}%")
    print(f"  Live overall     : {summary['live_overall']}%")
    print(f"  Python MACRO avg : {summary['python_macro_avg']}%  (mean of 11 cats)")
    print("  per-category:")
    for k, v in cats.items():
        print(f"    {k:26s}: {v}%")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  -> {args.json}")


if __name__ == "__main__":
    main()
