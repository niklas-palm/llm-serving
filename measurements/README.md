# Measurements

The rows behind the numbers in [docs/tuning.md](../docs/tuning.md), as data. One file for throughput
and latency, one for output quality, one for agentic quality. Every row carries the conditions it was measured under, so a
number can be compared with another only when those columns match.

| File | One row is | Rows |
|---|---|---|
| `benchmarks.csv` | one concurrency level of one benchmark run: hardware, engines, model, weights, KV precision, topology, flags, kernels, prompt shape, cache state, and the results (req/s, tokens/s, p50/p95/p99, time to first token where streamed, decode speed per request) | 1,900+ |
| `quality.csv` | one metric of one evaluation of one served configuration: family, model, weights, KV precision, thinking setting, task, setting, score, standard error | 320+ |
| `agentic.csv` | one metric of one agentic benchmark on one served configuration: family, model, weights, KV precision, engines, benchmark, metric, value, n, harness | 270+ |

All rows are vLLM 0.28.0 in this project's container except the Gemma 4 rows dated 2026-09-16 whose
`engine_version` reads 0.29.0 (the multi-token-prediction drafter and an engine comparison; 0.29.0 measured
within ±5% of 0.28.0 on the same models), measured through the CloudFront endpoint
from an in-region client with `scripts/benchmark.py` (60 to 120 s per level after warm-up) and
`scripts/quality.py` (lm-evaluation-harness 0.4.13; the quality rows dated 2026-09-10 are the standard
suite, evaluation deployments at 0.80 utilisation with a 2,048-token prefill chunk, thinking off; the Gemma 4 rows
dated 2026-09-16 ran at 0.75 with a 3,072-token chunk, their log-likelihood tasks wrapped in the chat template
and without wikitext, as the `notes` column says). The agentic rows are the published scaffolds run against the endpoint with the serving defaults: BFCL v4 (bfcl-eval), tau2-bench with Claude Sonnet 4.5 as the fixed user simulator and judge, mini-swe-agent on SWE-bench Verified graded by the official harness, and `scripts/extraction.py`. `concurrency_total` is requests in flight across the whole fleet or host;
`concurrency_per_engine` divides by `engines`. `cache` is `unique` (a nonce in front of every prompt),
`shared` (one prefix for all), or the multi-turn conversations of `turns` > 1.

Reading it:

```bash
# one model, unique prompts, all hardware: req/s at 64 per engine
python3 - <<'PY'
import csv
for r in csv.DictReader(open("measurements/benchmarks.csv")):
    if r["model"].endswith("30B-A3B-Instruct-2507-FP8") and r["shape"] == "ref" and r["cache"] == "unique" \
       and float(r["concurrency_per_engine"]) == 64:
        print(f'{r["date"]} {r["instance"]:18} {r["flags"][:40]:40} {float(r["rps"]):6.1f} req/s  p95 {r["p95_s"]} s')
PY
```

Two runs of the same configuration differ by about ±4% on throughput and ±25% on the p95 of time to
first token; treat smaller differences as noise. Rows are appended, never rewritten: a new engine
version gets new rows with its own `engine_version`, and the old ones stay for the comparison.
