#!/usr/bin/env python3
"""Load a deployed endpoint and report what it holds up to.

    python3 scripts/benchmark.py https://<endpoint> --key "$API_KEY"
    python3 scripts/benchmark.py https://<endpoint> --key "$API_KEY" --concurrency 64,128,256,512 \
        --input-tokens 1000 --output-tokens 190 --seconds 120
    python3 scripts/benchmark.py https://<endpoint> --key "$API_KEY" --concurrency 256 \
        --input-tokens 1000,4000,8000,8000,16000 --output-tokens 200,400,400,800     # a mixed workload
    python3 scripts/benchmark.py https://<endpoint> --key "$API_KEY" --stream --budget-seconds 8
    python3 scripts/benchmark.py https://<endpoint> --key "$API_KEY" --turns 6                # multi-turn
    python3 scripts/benchmark.py https://<endpoint> --key "$API_KEY" --schema                 # JSON output
    python3 scripts/benchmark.py https://<endpoint> --key "$API_KEY" --reasoning-effort low   # reasoning models
    python3 scripts/benchmark.py https://<endpoint> --key "$API_KEY" --images 1 --input-tokens 200  # image input

One row per concurrency level: requests/sec, input and output tokens/sec, latency p50/p95/p99, and the
decode speed a single request saw. Size a fleet on the aggregate numbers; check your latency budget on
the percentiles. Both come from the same run, which is the point. --stream adds time to first token and
measures decode from the first token to the last; --budget-seconds adds goodput, the requests per second
that finished inside the budget. --turns N runs N-turn conversations with a cookie jar per conversation,
so sticky routing is honoured; --schema asks for structured JSON; --reasoning-effort sets a reasoning
model's effort. A comma-separated --input-tokens or --output-tokens is a mix, each request drawing one.
--images N puts N generated images (--image-px square) before the text of each request, over
/v1/chat/completions, unique per request unless --same-image; the input token count then includes them.

Three things the script does that the numbers depend on, each with its measurement in docs/tuning.md
(*If you are benchmarking this yourself* and *Two things about measuring itself*): prompts are unique by default, a nonce up front defeating the
prefix cache, because that is the shape a fleet is sized for (--shared-prefix measures the cached case);
load is spread across processes, because one process driving 768 connections measured a third of the
true throughput; and every level drains its open requests before the next starts, because behind
CloudFront an abandoned request keeps generating and capped the next level at a fifth of its rate.
"""

from __future__ import annotations

import argparse
import base64
import functools
import json
import multiprocessing as mp
import os
import random
import signal
import statistics
import struct
import sys
import threading
import time
import uuid
import zlib

import requests

FILLER = ("The customer placed an order containing several items and asked about delivery timing, "
          "refunds, and the status of a previous return. ")
FOLLOW_UP = " Thanks. One more question about the same order: what happens if only part of it arrives?"
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["summary", "items", "refund_eligible"],
          "properties": {"summary": {"type": "string"},
                         "items": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                                              "required": ["name", "status"],
                                                              "properties": {"name": {"type": "string"},
                                                                             "status": {"type": "string"}}}},
                         "refund_eligible": {"type": "boolean"}}}


def prompt(n_tokens: int, shared: bool) -> str:
    body = FILLER * max(1, n_tokens // 25)
    # Unique prompts lead with the nonce. A trailing nonce would leave every earlier cache block
    # identical and still cacheable.
    return (body if shared else f"Request {uuid.uuid4()}. {body}")[: n_tokens * 5]


@functools.lru_cache(maxsize=4)
def _gradient(side: int) -> bytes:
    """Every row but the first, PNG-filtered. Built once per size: in pure Python it takes most of a second."""
    return b"".join(b"\x00" + bytes(v for x in range(side) for v in (x * 255 // side, y * 255 // side, 128))
                    for y in range(1, side))


def image(side: int, same: bool = False) -> str:
    """A side x side PNG as a data URL: a gradient with one random top row, so each is unique (no cache hit)
    and still a few hundred KB. The vision encoder's cost depends on the pixel count, not the content."""
    top = bytes(i % 251 for i in range(side * 3)) if same else os.urandom(side * 3)
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))  # noqa: E731
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(b"\x00" + top + _gradient(side), 1)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


def ask_images(sess, url: str, model: str, text: str, max_out: int,
               images: list[str]) -> tuple[float, float, int, int, str]:
    """One request with images first, over Chat Completions (the Responses API takes no images here)."""
    content = [{"type": "image_url", "image_url": {"url": u}} for u in images] + [{"type": "text", "text": text}]
    body = {"model": model, "max_tokens": max_out, "temperature": 0.0,
            "messages": [{"role": "user", "content": content}]}
    t0 = time.perf_counter()
    r = sess.post(f"{url}/v1/chat/completions", json=body, timeout=300)
    dt = time.perf_counter() - t0
    if r.status_code != 200:
        raise ValueError(f"status {r.status_code}")
    d = r.json()
    u = d.get("usage") or {}
    return (dt, 0.0, int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0),
            d["choices"][0]["message"]["content"] or "")


def ask(sess, url: str, model: str, text: str, max_out: int, stream: bool, schema: bool = False,
        effort: str = "") -> tuple[float, float, int, int, str]:
    """One request. Returns (latency, time to first token, input tokens, output tokens, answer text).
    Time to first token is 0 without streaming. Raises on anything but a well-formed answer."""
    body = {"model": model, "input": text, "max_output_tokens": max_out, "temperature": 0.0}
    if schema:
        body["text"] = {"format": {"type": "json_schema", "name": "order", "strict": True, "schema": SCHEMA}}
    if effort:
        body["reasoning"] = {"effort": effort}
    t0 = time.perf_counter()
    if not stream:
        r = sess.post(f"{url}/v1/responses", json=body, timeout=300)
        dt = time.perf_counter() - t0
        if r.status_code != 200:
            raise ValueError(f"status {r.status_code}")
        # Parsed before anything is counted, so a malformed body raises before "ok" is counted.
        d = r.json()
        u = d.get("usage") or {}
        answer = "".join(c.get("text", "") for o in d.get("output") or [] for c in o.get("content") or [])
        return dt, 0.0, int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0), answer
    body["stream"] = True
    ttft, usage, parts = 0.0, {}, []
    with sess.post(f"{url}/v1/responses", json=body, timeout=300, stream=True) as r:
        if r.status_code != 200:
            raise ValueError(f"status {r.status_code}")
        for line in r.iter_lines():
            if not line.startswith(b"data: "):
                continue
            ev = json.loads(line[6:])
            kind = ev.get("type", "")
            # The first generated token of any kind: a reasoning model streams its thinking first and
            # its answer text only at the end, and time to first token must not wait for the answer.
            if kind.endswith(".delta") and not ttft:
                ttft = time.perf_counter() - t0
            if kind == "response.output_text.delta":
                parts.append(ev.get("delta", ""))
            elif kind == "response.completed":
                usage = (ev.get("response") or {}).get("usage") or {}
    dt = time.perf_counter() - t0
    if not usage:
        raise ValueError("stream ended without response.completed")
    return dt, ttft, int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0), "".join(parts)


def worker(url: str, key: str, model: str, conc: int, in_tok: list[int], out_tok: list[int],
           seconds: float, shared: bool, q: mp.Queue, stream: bool = False, turns: int = 1,
           budget: float = 0.0, schema: bool = False, effort: str = "", images: int = 0, image_px: int = 1024,
           same_image: bool = False) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)   # the parent handles Ctrl-C and terminates us

    stop = threading.Event()
    lock = threading.Lock()
    stats = {"ok": 0, "fail": 0, "in": 0, "out": 0, "lat": [], "ttft": [], "decode": [], "in_budget": 0}
    fixed = [image(image_px, same=True)] * images if images and same_image else []
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def record(dt: float, ttft: float, used_in: int, used_out: int) -> None:
        with lock:
            stats["ok"] += 1
            stats["in"] += used_in
            stats["out"] += used_out
            stats["lat"].append(dt)
            stats["ttft"].append(ttft)
            # Decode speed one request saw. With streaming, tokens after the first over the time after
            # the first token; without, output tokens over the whole request, which folds prefill in.
            if stream and used_out > 1 and dt > ttft:
                stats["decode"].append((used_out - 1) / (dt - ttft))
            elif dt > 0:
                stats["decode"].append(used_out / dt)
            if budget and dt <= budget:
                stats["in_budget"] += 1

    def one() -> None:
        while not stop.is_set():
            # A session per conversation: its own cookie jar, so a sticky load balancer keeps the
            # conversation on one engine, and a fresh connection so it is balanced like a new client.
            sess = requests.Session()
            sess.headers.update(headers)
            text = ""
            for _ in range(turns):
                if stop.is_set():
                    break
                try:
                    text = text or prompt(random.choice(in_tok), shared)
                    if images:
                        dt, ttft, used_in, used_out, answer = ask_images(
                            sess, url, model, text, random.choice(out_tok),
                            fixed or [image(image_px) for _ in range(images)])
                    else:
                        dt, ttft, used_in, used_out, answer = ask(sess, url, model, text, random.choice(out_tok),
                                                                  stream, schema, effort)
                except (requests.RequestException, ValueError, AttributeError, TypeError, KeyError):
                    # A body that is not the promised shape is a failure too. Left uncaught, it killed
                    # the thread silently and the row reported a plausible number at lower concurrency.
                    with lock:
                        stats["fail"] += 1
                    # An endpoint that fails instantly (503 while targets are unhealthy, refused
                    # connections) would otherwise be hit thousands of times a second per thread.
                    time.sleep(0.5)
                    break
                if stop.is_set():
                    break             # finished after the window: drained, not counted
                record(dt, ttft, used_in, used_out)
                text = f"{text} {answer}{FOLLOW_UP}"

    threads = [threading.Thread(target=one, daemon=True) for _ in range(conc)]
    t_start = time.perf_counter()
    for t in threads:
        t.start()
    time.sleep(seconds)
    stop.set()
    wall = time.perf_counter() - t_start
    for t in threads:   # drain: every request opened inside the window finishes before we report
        t.join(timeout=330)
    with lock:   # copies, because a thread still inside its 330 s join could append while this pickles
        q.put({"wall": wall, **stats, "lat": list(stats["lat"]), "ttft": list(stats["ttft"]),
               "decode": list(stats["decode"])})


def run_level(a: argparse.Namespace, model: str, total: int) -> dict:
    procs_n = min(a.processes, total)
    per_proc = [total // procs_n + (1 if i < total % procs_n else 0) for i in range(procs_n)]
    q: mp.Queue = mp.Queue()
    procs = [mp.Process(target=worker, args=(a.url, a.key, model, n, a.input_tokens, a.output_tokens,
                                             a.seconds, a.shared_prefix, q, a.stream, a.turns,
                                             a.budget_seconds, a.schema, a.reasoning_effort, a.images, a.image_px,
                                             a.same_image)) for n in per_proc]
    for p in procs:
        p.start()
    # A worker killed by the OS (OOM) would leave a bare q.get() waiting forever; the level, the drain
    # and the per-request timeout bound how long a healthy worker can take.
    results = []
    for _ in procs:
        try:
            results.append(q.get(timeout=a.seconds + 700))
        except Exception:                                   # noqa: BLE001 - queue.Empty
            print("  a load process did not report; its share is missing from this level",
                  file=sys.stderr)
    for p in procs:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()
    if not results:
        sys.exit("no load process reported; is the endpoint reachable?")

    wall = max(r["wall"] for r in results)
    ok = sum(r["ok"] for r in results)
    fail = sum(r["fail"] for r in results)
    lat = sorted(x for r in results for x in r["lat"])
    ttft = sorted(x for r in results for x in r["ttft"])
    decode = [x for r in results for x in r["decode"]]
    pct = lambda xs, p: xs[min(int(len(xs) * p), len(xs) - 1)] if xs else 0.0  # noqa: E731
    return {
        "concurrency": total, "rps": ok / wall, "input_tok_s": sum(r["in"] for r in results) / wall,
        "output_tok_s": sum(r["out"] for r in results) / wall,
        "p50": statistics.median(lat) if lat else 0.0, "p95": pct(lat, 0.95), "p99": pct(lat, 0.99),
        "ttft_p50": statistics.median(ttft) if ttft else 0.0, "ttft_p95": pct(ttft, 0.95),
        "goodput_rps": sum(r["in_budget"] for r in results) / wall,
        "failed": fail,
        "decode_tok_s_per_request": statistics.mean(decode) if decode else 0.0,
        "output_tokens_per_request": sum(r["out"] for r in results) / ok if ok else 0.0,
    }


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url", help="the Endpoint stack output")
    ap.add_argument("--key", required=True, help="the ApiKeyValue stack output")
    ap.add_argument("--concurrency", default="64,128,256",
                    help="comma-separated levels to sweep; keep going until throughput stops rising")
    ap.add_argument("--input-tokens", default="1000", help="one size, or a comma-separated mix")
    ap.add_argument("--output-tokens", default="190", help="one size, or a comma-separated mix")
    ap.add_argument("--seconds", type=float, default=90, help="per level, after warm-up")
    ap.add_argument("--warmup-seconds", type=float, default=20)
    ap.add_argument("--processes", type=int, default=8)
    ap.add_argument("--shared-prefix", action="store_true",
                    help="identical prompts, so the prefix cache hits; default is unique prompts")
    ap.add_argument("--stream", action="store_true",
                    help="stream answers and report time to first token; decode speed is then first-to-last token")
    ap.add_argument("--turns", type=int, default=1,
                    help="turns per conversation; each turn resends the conversation so far plus the answer")
    ap.add_argument("--budget-seconds", type=float, default=0.0,
                    help="latency budget; adds goodput, the requests/s that finished inside it")
    ap.add_argument("--schema", action="store_true",
                    help="ask for structured JSON output against a fixed schema, as tool-calling traffic does")
    ap.add_argument("--reasoning-effort", default="", choices=["", "low", "medium", "high"],
                    help="reasoning effort for models that support it; changes tokens per answer")
    ap.add_argument("--images", type=int, default=0,
                    help="images per request, before the text; for models that read images")
    ap.add_argument("--image-px", type=int, default=1024, help="side of each square image, in pixels")
    ap.add_argument("--same-image", action="store_true",
                    help="the same image in every request, so it can be a cached prefix")
    ap.add_argument("--json", action="store_true", help="also print one JSON line per level")
    a = ap.parse_args()
    a.url = a.url.rstrip("/")

    def sizes(flag: str, raw: str) -> list[int]:
        try:
            values = [int(x) for x in raw.split(",") if x.strip()]
        except ValueError:
            sys.exit(f"{flag} must be comma-separated integers (got {raw!r})")
        if not values or min(values) < 1:
            sys.exit(f"{flag} needs one or more sizes of at least 1 (got {raw!r})")
        return values

    levels = sizes("--concurrency", a.concurrency)
    a.input_tokens = sizes("--input-tokens", a.input_tokens)
    a.output_tokens = sizes("--output-tokens", a.output_tokens)
    if a.processes < 1 or a.seconds <= 0 or a.warmup_seconds < 0 or a.turns < 1 or a.budget_seconds < 0:
        sys.exit("--processes and --turns at least 1, --seconds above 0, --warmup-seconds and --budget-seconds at least 0")
    if a.images < 0 or a.image_px < 32 or (a.images and (a.stream or a.schema or a.reasoning_effort)):
        sys.exit("--images at least 0 and --image-px at least 32; --images does not combine with --stream, "
                 "--schema or --reasoning-effort")
    model = served_model(a.url, a.key)
    shape = lambda v: str(v[0]) if len(v) == 1 else f"a mix of {','.join(map(str, v))}"  # noqa: E731
    print(f"model {model}\n{shape(a.input_tokens)} input / {shape(a.output_tokens)} output tokens, "
          f"{'shared-prefix' if a.shared_prefix else 'unique'} prompts"
          f"{f', {a.turns}-turn conversations' if a.turns > 1 else ''}"
          f"{', streamed' if a.stream else ''}{', structured output' if a.schema else ''}"
          f"{f', reasoning effort {a.reasoning_effort}' if a.reasoning_effort else ''}"
          f"{f', {a.images} image(s) of {a.image_px} px per request' if a.images else ''}"
          f"{' (the same each time)' if a.same_image and a.images else ''}, {a.seconds:g}s per level "
          f"after {a.warmup_seconds:g}s warm-up, {a.processes} client processes\n")

    # Warm-up: the first requests pay for CUDA graph capture and kernel autotuning. Measuring from the
    # instant a target reports healthy read 21% low here.
    saved = a.seconds
    a.seconds = a.warmup_seconds
    run_level(a, model, levels[0])
    a.seconds = saved

    extra = (f" {'ttft p50':>9} {'ttft p95':>9}" if a.stream else "") + (f" {'goodput':>8}" if a.budget_seconds else "")
    print(f"{'conc':>6} {'req/s':>7} {'in tok/s':>9} {'out tok/s':>10} {'p50':>7} {'p95':>7} "
          f"{'p99':>7} {'tok/s/req':>10} {'failed':>7}{extra}")
    prev = None
    for level in levels:
        r = run_level(a, model, level)
        extra = ((f" {r['ttft_p50']:>8.2f}s {r['ttft_p95']:>8.2f}s" if a.stream else "")
                 + (f" {r['goodput_rps']:>8.1f}" if a.budget_seconds else ""))
        print(f"{level:>6} {r['rps']:>7.1f} {r['input_tok_s']:>9,.0f} {r['output_tok_s']:>10,.0f} "
              f"{r['p50']:>6.2f}s {r['p95']:>6.2f}s {r['p99']:>6.2f}s "
              f"{r['decode_tok_s_per_request']:>10.1f} {r['failed']:>7}{extra}")
        if a.json:
            print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}))
        if prev and r["input_tok_s"] < 0.85 * prev["input_tok_s"]:
            print("        throughput fell by more than 15% with more concurrency. A saturated server "
                  "plateaus; a fall usually means the client is the bottleneck. Add --processes.",
                  file=sys.stderr)
        prev = r

    print("\nSize on the aggregate columns at the highest level whose p95 (or p99) is inside your budget."
          "\nUnique-prompt numbers. With every prompt cached the same hardware measured +46% on 1k prompts"
          "\nand 2.7x on 4k prompts for a mixture-of-experts model, and nothing for a dense one.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        # Without this, Ctrl-C printed one traceback per worker and waited for the level to finish.
        for p in mp.active_children():
            p.terminate()
        print("\ninterrupted.", file=sys.stderr)
        sys.exit(130)
