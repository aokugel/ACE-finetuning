# Consolidated Results

BFCL Python-subset macro accuracy + serving metrics @ 32 concurrency.

| Model | Mode | Macro % | Non-Live % | Live % | TTFT p50 | E2E p50 | tok/s | req/s |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| Qwen2.5-7B-Instruct | baseline bf16 | 79.52 | 75.33 | 75.35 | 70.2 ms | 187.3 ms | 2629.7 | 134.85 |
| Qwen2.5-7B-Instruct | ToolACE LoRA bf16 | 81.12 | 74.85 | 76.61 | 70.9 ms | 186.9 ms | 2608.3 | 133.76 |
| Qwen2.5-7B-Instruct | ToolACE LoRA fp8 | 80.76 | 74.65 | 75.72 | 73.2 ms | 162.1 ms | 3227.9 | 165.89 |
| Qwen2.5-7B-Instruct | ToolACE QLoRA bf16 | 81.56 | 74.31 | 76.02 | — | — | — | — |
| Qwen3.5-9B | baseline bf16 (thinking) | 74.98 | 73.0 | 62.62 | 132.6 ms | 1231.8 ms | 2942.3 | 25.09 |
| Qwen3.5-9B | baseline bf16 (no-think) | 74.93 | 73.04 | 62.92 | 146.8 ms | 352.8 ms | 1437.5 | 73.72 |
| Qwen2.5-14B-Instruct | baseline bf16 | 82.6 | 75.58 | 74.17 | 73.2 ms | 306.9 ms | 1600.0 | 82.05 |
| Qwen2.5-14B-Instruct | ToolACE LoRA bf16 (uniform) | 79.33 | 69.5 | 75.72 | 67.8 ms | 304.0 ms | 1585.4 | 81.3 |
| Qwen2.5-14B-Instruct | ToolACE LoRA fp8 | 78.28 | 69.65 | 75.8 | 65.3 ms | 219.3 ms | 2162.4 | 110.89 |
| Qwen2.5-14B-Instruct | ToolACE LoRA bf16 (lighter) | 80.83 | 74.83 | 73.43 | — | — | — | — |
