#!/usr/bin/env python3
"""Score the served model on standard benchmarks through the endpoint, so a cheaper precision can be
checked for what it costs in answers, not only what it saves in GPUs.

    pip install "lm-eval[api,math]==0.4.13" transformers langdetect immutabledict nltk
    python3 scripts/quality.py https://<endpoint> --key "$API_KEY"                      # the standard suite
    python3 scripts/quality.py https://<endpoint> --key "$API_KEY" --suite quick         # 10 minutes
    python3 scripts/quality.py https://<endpoint> --key "$API_KEY" --csv results.csv --tag fp8
    python3 scripts/quality.py https://<endpoint> --key "$API_KEY" --compare <dir of the bf16 run>

Runs lm-evaluation-harness against the deployment's OpenAI-compatible API, so what is scored is the
served configuration: weights, KV cache precision, kernels and all. Two kinds of task, two API paths:

- log-likelihood tasks (multiple choice scored by the probability of each answer, and perplexity) go
  through /v1/completions with echo and logprobs; they need the model's tokenizer, fetched from the
  Hub by model id, and measure the model's raw predictions with no generation involved;
- generative tasks (the model writes an answer that is checked) go through /v1/chat/completions with
  the chat template, greedy, a 1,024-token cap.

The standard suite is the set quantised checkpoints are usually published with: arc_challenge (25-shot),
hellaswag (10-shot), mmlu (5-shot), truthfulqa_mc2, winogrande (5-shot), gsm8k (5-shot), wikitext
perplexity, plus ifeval (instruction following). Left out after trying them: gpqa (gated on the Hub),
MATH (its few-shot answer format is not followed under a chat template, so exact match reads near zero
for every model), and humaneval (an instruct model given a bare function signature writes prose or a
second definition, the stop strings cut it, and quantised builds "beat" bf16 by 3 to 12 points with
one-directional flips; the chat form of the task cannot be run over an API because it pre-fills the
assistant turn). A code score needs a chat-compatible task; until then, do not read precision from it.
Limits keep it to about 40 minutes on one engine; the interval at these sizes is 1 to 2 points, enough to see a precision that answers worse, not enough to certify one that
does not. Run the same command against two deployments and compare rows, or pass --compare to get the
fraction of questions whose answer changed: a precision can keep the aggregate and still flip one
answer in ten.

A model trained with a beginning-of-sequence token needs it on the completions path too, and nothing there
adds one: the harness tokenises without special tokens unless told to, and the engine's tokenisation of a
text prompt follows the tokenizer's add_bos_token flag, which Gemma 4 ships turned off (its chat template
writes <bos> itself, so chat scores are unaffected). Without it the 31B scored a perplexity of 14,931 on a
21-token sentence against 4.5 with it, and every log-likelihood task read chance. This script therefore
asks the harness to add <bos> whenever the tokenizer has one, and hands it a copy of the tokenizer with
the flag on when the original would not comply. `--text-prompts` cannot do that: the engine tokenises,
and for such a model the log-likelihood scores are meaningless in that mode.

That was necessary and not sufficient for Gemma 4: its instruction-tuned checkpoints are out of
distribution on raw text even after `<bos>`. A 58-token paragraph read a perplexity of 654 (12B), 8,923
(26B MoE, fp8) and 86 (31B) as raw text, and 11.7, 17.8 and 18.7 when placed as the model's own turn
inside its chat template; the raw completion of `The capital of France is` opened a thinking channel
or produced gibberish. `--chat-loglik` wraps the log-likelihood prompts in the chat template (the
harness's own support for it on this path) and skips wikitext, which has no turn to live in. Qwen's
instruction-tuned models scored the same raw and wrapped, so it is off by default.

Two things about the deployment under test. A thinking model must be served with thinking off for these
settings (see docs/tuning.md), or the chain of thought eats the cap and every generative score
collapses. And the log-likelihood requests ask the engine for the probability of every prompt token,
which materialises the whole vocabulary for every position of a 2,000-token few-shot prompt: several
GiB per request. At the serving default of 0.95 memory utilisation the engine had 2.4 GiB free and died
with a CUDA out-of-memory on the first batch, and 0.85 was not enough either: the logits buffer scales
with the prefill chunk, not the request. Deploy the configuration you are scoring with
`gpuMemoryUtilization: 0.80` and `maxNumBatchedTokens: 2048` for the duration of the evaluation
(measured to hold at 4 in flight with 10-shot prompts); neither changes what is scored. Two models
refused or needed more: a multimodal model sizes the chunk against its largest image (Gemma 4 refuses
2048 with "max_tokens_per_mm_item (2496) is larger than max_num_batched_tokens" even with images
limited to 0; 3072 at 0.75 held, its 262k vocabulary making the buffer 1.7x larger per token), and a
bf16 model that fills the card cannot give up utilisation, so lower `maxModelLen` to a few thousand
tokens instead (a 31B dense at 0.80 with `maxModelLen: 8192`).

Do not compare with published numbers, which use other prompts and settings; compare deployments with
each other.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import requests

# (kind, tasks, extra harness arguments, limit). Log-likelihood passes carry their own few-shot counts.
SUITES = {
    "standard": [
        ("loglik", "arc_challenge", ["--num_fewshot", "25"], 500),
        ("loglik", "hellaswag", ["--num_fewshot", "10"], 500),
        ("loglik", "winogrande", ["--num_fewshot", "5"], 500),
        ("loglik", "truthfulqa_mc2", [], 500),
        ("loglik", "wikitext", [], 60),
        ("loglik", "mmlu", ["--num_fewshot", "5"], 50),       # per subject: 57 x 50
        ("gen", "gsm8k", ["--num_fewshot", "5"], 500),
        ("gen", "ifeval", [], 600),
    ],
    "ifeval": [("gen", "ifeval", [], 600)],
    "quick": [
        ("loglik", "arc_challenge,winogrande,wikitext", [], 200),
        ("gen", "gsm8k", ["--num_fewshot", "5"], 200),
    ],
}
METRICS = {   # the one number to report per task, and the key of its per-sample score for --compare
    "arc_challenge": "acc_norm,none", "hellaswag": "acc_norm,none", "winogrande": "acc,none",
    "truthfulqa_mc2": "acc,none", "mmlu": "acc,none", "wikitext": "word_perplexity,none",
    "gsm8k": "exact_match,strict-match", "ifeval": "prompt_level_strict_acc,none",
}
SAMPLE_SCORE = {"gsm8k": "exact_match", "mmlu": "acc", "arc_challenge": "acc_norm", "hellaswag": "acc_norm"}


def served_model(url: str, key: str) -> str:
    """The first model /v1/models lists; one line and exit on any failure, since a wrong key or URL is
    the usual first-run mistake and a traceback says nothing about which."""
    try:
        r = requests.get(f"{url}/v1/models", headers={"Authorization": f"Bearer {key}"}, timeout=30)
        r.raise_for_status()
        models = [m["id"] for m in r.json().get("data") or [] if isinstance(m, dict) and m.get("id")]
    except (requests.RequestException, ValueError) as e:
        sys.exit(f"could not read {url}/v1/models: {e}")
    if not models:
        sys.exit(f"{url}/v1/models lists no model")
    return models[0]


def bos_arguments(tok, source: str) -> tuple[str, str]:
    """The tokenizer to hand the harness for the log-likelihood pass, and the model arguments that make it send <bos>.

    Returns the tokenizer path unchanged and no arguments for a tokenizer without a BOS id (Qwen). With one, the
    harness gets add_bos_token=True, so it tokenises with add_special_tokens=True, and custom_prefix_token_id, the
    start token of the rolling perplexity pass. A tokenizer that adds nothing even then (Gemma 4: add_bos_token
    false in its config) is saved with the flag on and the copy is handed over instead; the reloaded copy emits
    the BOS id first. Measured on the 31B: perplexity 14,931 without <bos>, 4.5 with it, on the same sentence.
    """
    bos = getattr(tok, "bos_token_id", None)
    if bos is None:
        return source, ""
    extra = f",add_bos_token=True,custom_prefix_token_id={bos}"
    if list(tok("x", add_special_tokens=True).input_ids[:1]) == [bos]:
        return source, extra
    tok.add_bos_token = True
    path = tempfile.mkdtemp(prefix="tokenizer-bos-")
    tok.save_pretrained(path)
    return path, extra


def loglik_tokenizer(a: argparse.Namespace, model: str) -> str:
    """The `tokenized_requests=...` part of the harness arguments for log-likelihood tasks, computed once per run."""
    if a.text_prompts:
        return "tokenized_requests=False"
    source = a.tokenizer or model
    if a.chat_loglik:      # the chat template writes <bos> itself; adding another would double it
        return f"tokenized_requests=True,tokenizer={source}"
    from transformers import AutoTokenizer
    path, bos = bos_arguments(AutoTokenizer.from_pretrained(source), source)
    if bos and path != source:
        print(f"tokenizer {source} does not add <bos> by itself; the harness gets a copy that does ({path})")
    return f"tokenized_requests=True,tokenizer={path}{bos}"


def harness(kind: str, tasks: str, extra: list[str], limit: int, a: argparse.Namespace, model: str, out: str) -> None:
    if kind == "loglik" and a.chat_loglik and tasks == "wikitext":
        print("\nskipping wikitext: rolling perplexity over raw text, which --chat-loglik exists to avoid", flush=True)
        return
    if kind == "loglik":
        tok = a.loglik_tokenizer
        extra = extra + (["--apply_chat_template"] if a.chat_loglik else [])
        model_args = (f"model={model},base_url={a.url}/v1/completions,num_concurrent={a.loglik_concurrency},"
                      f"max_retries=3,{tok},max_length=8192")
        cmd = ["lm_eval", "--model", "local-completions", "--model_args", model_args]
    else:
        model_args = (f"model={model},base_url={a.url}/v1/chat/completions,num_concurrent={a.concurrency},"
                      f"max_retries=3,tokenized_requests=False")
        cmd = ["lm_eval", "--model", "local-chat-completions", "--model_args", model_args,
               "--apply_chat_template", "--gen_kwargs", "temperature=0,max_gen_toks=1024"]
    cmd += ["--tasks", tasks, "--limit", str(limit), "--log_samples", "--output_path", f"{out}/{kind}-{tasks.split(',')[0]}",
            "--seed", "1234"] + extra
    print(f"\n[{time.strftime('%H:%M:%S')}] {kind}: {tasks} (limit {limit})", flush=True)
    r = subprocess.run(cmd, env={**os.environ, "OPENAI_API_KEY": a.key})
    if r.returncode:
        print(f"  harness failed for {tasks} (exit {r.returncode}); continuing", file=sys.stderr)


def results(out: str) -> dict:
    """Merge every results file the harness wrote under `out` into one task -> metrics dict."""
    merged: dict = {}
    for path in sorted(glob.glob(f"{out}/**/results_*.json", recursive=True)):
        merged.update(json.load(open(path)).get("results", {}))
    return merged


def summarise(out: str, tag: str, model: str, csv_path: str = "") -> list[dict]:
    res = results(out)
    if not res:
        sys.exit("the harness wrote no results")
    rows = []
    print(f"\n{'task':28} {'metric':30} {'score':>8} {'+-':>6} {'n':>6}")
    for task, m in res.items():
        key = METRICS.get(task)
        if key not in m:
            continue
        err = m.get(key.replace(",", "_stderr,", 1), 0.0)
        err = err if isinstance(err, (int, float)) else 0.0
        n = sample_count(out, task)
        score = m[key]
        shown = f"{score:8.2f}" if task == "wikitext" else f"{100 * score:7.1f}%"
        print(f"{task:28} {key:30} {shown:>8} {100 * err:>5.1f} {n:>6}")
        rows.append({"tag": tag, "model": model, "task": task, "metric": key, "score": round(score, 4),
                     "stderr": round(err, 4), "n": n})
    if not rows:
        sys.exit("no task in the results matched METRICS")
    if csv_path:
        new = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            if new:
                w.writeheader()
            w.writerows(rows)
        print(f"\nappended {len(rows)} rows to {csv_path}")
    return rows


def samples(out: str, task: str) -> dict:
    """doc_id -> per-sample score for one task, from the harness's sample logs."""
    key = SAMPLE_SCORE.get(task)
    scores: dict = {}
    for path in glob.glob(f"{out}/**/samples_{task}_*.jsonl", recursive=True):
        subject = os.path.basename(path).rsplit("_", 1)[0]      # mmlu writes one file per subject
        for line in open(path):
            d = json.loads(line)
            if key and d.get("filter", "strict-match") in ("strict-match", "none") and key in d:
                scores[(subject, d["doc_id"])] = float(d[key])
    return scores


def sample_count(out: str, task: str) -> int:
    n = len(samples(out, task))
    if n:
        return n
    return sum(1 for p in glob.glob(f"{out}/**/samples_{task}_*.jsonl", recursive=True) for _ in open(p))


def compare(out: str, other: str) -> None:
    """How many answers changed between two runs, question by question. The aggregate can stay while a
    tenth of the answers flip; that is what a cheaper precision usually does first."""
    print(f"\n{'task':14} {'shared':>7} {'flipped':>8} {'this right, other wrong':>24} {'other right, this wrong':>24}")
    for task in SAMPLE_SCORE:
        a, b = samples(out, task), samples(other, task)
        shared = sorted(set(a) & set(b))
        if not shared:
            continue
        gained = sum(1 for i in shared if a[i] > b[i])
        lost = sum(1 for i in shared if a[i] < b[i])
        print(f"{task:14} {len(shared):>7} {100 * (gained + lost) / len(shared):>7.1f}% {gained:>24} {lost:>24}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url", help="the Endpoint stack output")
    ap.add_argument("--key", required=True, help="the ApiKeyValue stack output")
    ap.add_argument("--suite", default="standard", choices=sorted(SUITES), help="which task set")
    ap.add_argument("--tokenizer", default="", help="tokenizer id for log-likelihood tasks; default the served model id")
    ap.add_argument("--text-prompts", action="store_true",
                    help="send text instead of token ids for log-likelihood tasks: for models whose Hub tokenizer "
                         "does not match the engine's (Mistral tekken tokenizers), or that have no Hub tokenizer")
    ap.add_argument("--chat-loglik", action="store_true",
                    help="wrap the log-likelihood prompts in the chat template and skip wikitext: for instruction-tuned "
                         "models that collapse on raw text (Gemma 4)")
    ap.add_argument("--concurrency", type=int, default=16, help="generative requests in flight")
    ap.add_argument("--loglik-concurrency", type=int, default=4,
                    help="log-likelihood requests in flight; each holds vocabulary x prompt logits on the GPU")
    ap.add_argument("--output", default="", help="directory for the harness output; default a temp dir")
    ap.add_argument("--tag", default="", help="label written to --csv rows")
    ap.add_argument("--csv", default="", help="append one row per task here")
    ap.add_argument("--compare", default="", help="output directory of another run: report flipped answers")
    a = ap.parse_args()
    if shutil.which("lm_eval") is None:
        sys.exit('lm_eval not found: pip install "lm-eval[api]==0.4.13" transformers')
    a.url = a.url.rstrip("/")
    out = a.output or tempfile.mkdtemp(prefix="quality-")
    model = served_model(a.url, a.key)
    print(f"model {model}, suite {a.suite}, output {out}")
    if any(kind == "loglik" for kind, *_ in SUITES[a.suite]):
        a.loglik_tokenizer = loglik_tokenizer(a, model)
    for kind, tasks, extra, limit in SUITES[a.suite]:
        harness(kind, tasks, extra, limit, a, model, out)
    summarise(out, a.tag or model, model, a.csv)
    if a.compare:
        compare(out, a.compare)
    return 0


if __name__ == "__main__":
    sys.exit(main())
