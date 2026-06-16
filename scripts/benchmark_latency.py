#!/usr/bin/env python
"""Production-style latency/throughput benchmark against a running vLLM
OpenAI-compatible server.

Measures, per concurrency level, over streamed chat-completion requests:
  * TTFT  (time to first token)         -- p50 / p90 / p99
  * E2E   (end-to-end request latency)  -- p50 / p90 / p99
  * TPOT  (time per output token, decode)-- mean
  * output tokens/sec (system throughput)
  * requests/sec

The prompts are realistic tool-calling requests (system prompt listing JSON
functions + a user instruction), so token shapes resemble BFCL/production.

Usage:
  python scripts/benchmark_latency.py --model Qwen2.5-7B-Instruct --port 8000 \
      --concurrency 1 16 32 --requests-per-level 96 --out results/latency/7b_bf16.json
"""
import argparse
import asyncio
import json
import statistics
import time

from openai import AsyncOpenAI

SYSTEM = (
    "You are an expert in composing functions. You are given a question and a set "
    "of possible functions. If you decide to invoke a function, return it as "
    "[func_name(param=value)].\n"
    "Here is a list of functions in JSON format:\n"
    '[{"name":"get_weather","description":"Get current weather for a city",'
    '"parameters":{"type":"dict","properties":{"city":{"type":"string"},'
    '"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["city"]}},'
    '{"name":"search_flights","description":"Search flights between two airports",'
    '"parameters":{"type":"dict","properties":{"origin":{"type":"string"},'
    '"destination":{"type":"string"},"date":{"type":"string"}},'
    '"required":["origin","destination","date"]}}]'
)
USER_PROMPTS = [
    "What's the weather in Paris in celsius?",
    "Find flights from JFK to LAX on 2026-07-01.",
    "Tell me the temperature in Tokyo in fahrenheit.",
    "I need a flight from SFO to SEA on 2026-08-15.",
]


def pct(xs, p):
    if not xs:
        return float("nan")
    return statistics.quantiles(xs, n=100)[p - 1] if len(xs) > 1 else xs[0]


async def one_request(client, model, idx, max_tokens):
    user = USER_PROMPTS[idx % len(USER_PROMPTS)]
    t0 = time.perf_counter()
    ttft = None
    n_out = 0
    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": user}],
        temperature=0.0,
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},
    )
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            if ttft is None:
                ttft = time.perf_counter() - t0
            n_out += 1
    e2e = time.perf_counter() - t0
    if ttft is None:
        ttft = e2e
    return {"ttft": ttft, "e2e": e2e, "n_out": max(n_out, 1)}


async def run_level(client, model, concurrency, n_requests, max_tokens):
    sem = asyncio.Semaphore(concurrency)
    results = []

    async def worker(i):
        async with sem:
            results.append(await one_request(client, model, i, max_tokens))

    t0 = time.perf_counter()
    await asyncio.gather(*[worker(i) for i in range(n_requests)])
    wall = time.perf_counter() - t0

    ttfts = sorted(r["ttft"] for r in results)
    e2es = sorted(r["e2e"] for r in results)
    total_out = sum(r["n_out"] for r in results)
    # TPOT approximated from decode time / decode tokens
    tpots = [(r["e2e"] - r["ttft"]) / max(r["n_out"] - 1, 1) for r in results]
    return {
        "concurrency": concurrency,
        "n_requests": n_requests,
        "wall_s": round(wall, 3),
        "ttft_ms": {"p50": round(pct(ttfts, 50) * 1e3, 1),
                    "p90": round(pct(ttfts, 90) * 1e3, 1),
                    "p99": round(pct(ttfts, 99) * 1e3, 1)},
        "e2e_ms": {"p50": round(pct(e2es, 50) * 1e3, 1),
                   "p90": round(pct(e2es, 90) * 1e3, 1),
                   "p99": round(pct(e2es, 99) * 1e3, 1)},
        "tpot_ms_mean": round(statistics.mean(tpots) * 1e3, 2),
        "output_tok_per_s": round(total_out / wall, 1),
        "requests_per_s": round(n_requests / wall, 2),
        "total_output_tokens": total_out,
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 16, 32])
    ap.add_argument("--requests-per-level", type=int, default=96)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    client = AsyncOpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="EMPTY")

    # warmup
    await one_request(client, args.model, 0, 16)

    levels = []
    for c in args.concurrency:
        print(f">>> concurrency={c} ...", flush=True)
        res = await run_level(client, args.model, c, args.requests_per_level, args.max_tokens)
        print(json.dumps(res, indent=2))
        levels.append(res)

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"model": args.model, "max_tokens": args.max_tokens, "levels": levels}, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
