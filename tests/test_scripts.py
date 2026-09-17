"""The scripts are not imported by the stack tests, so a NameError in one of them reached a user:
`local_config_path()` was called in two places and defined nowhere."""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "..", "scripts")
sys.path.insert(0, SCRIPTS)   # endpoint_info imports build_image from the same directory


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPTS, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_script_runs_help():
    """--help runs the module body and the argparse setup; neither needs credentials."""
    import subprocess
    for name in ("build_image", "endpoint_info", "test_endpoint", "benchmark", "size_fleet", "quality", "extraction"):
        r = subprocess.run([sys.executable, os.path.join(SCRIPTS, f"{name}.py"), "--help"],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


def test_region_and_image_follow_the_config_in_use(tmp_path, monkeypatch):
    """With $CONFIG pointing elsewhere, the region must come from that file and the image URI must be
    written next to it, or the deploy reads a file the build never wrote."""
    bi = _load("build_image")
    cfg = tmp_path / "other.yaml"
    cfg.write_text("region: eu-west-2\n")
    monkeypatch.setenv("CONFIG", str(cfg))
    assert bi.config_region() == "eu-west-2"
    (tmp_path / "config.local.yaml").write_text("region: eu-north-1\napiKey: keep-me-1234567890\n")
    assert bi.config_region() == "eu-north-1", "config.local.yaml wins, like in app.py"
    uri = "111122223333.dkr.ecr.eu-west-2.amazonaws.com/gpu-llm-serving:vllm-0.28.0"
    bi.write_image_uri(uri)
    assert (tmp_path / "config.local.yaml").read_text() == f"region: eu-north-1\napiKey: keep-me-1234567890\nimage: {uri}\n"


def test_codebuild_may_assume_the_role_only_from_this_account_and_project_in_any_region():
    bi = _load("build_image")
    cond = bi.trust_policy("111122223333", "aws")["Statement"][0]["Condition"]
    assert cond["StringEquals"] == {"aws:SourceAccount": "111122223333"}
    assert cond["ArnLike"] == {"aws:SourceArn": "arn:aws:codebuild:*:111122223333:project/gpu-llm-serving-build"}, \
        "any region: one account-wide role serves every region this account builds in"


def test_a_malformed_usage_body_is_one_failed_request_not_an_ok_and_a_failure(monkeypatch):
    """`ok` was incremented before the usage fields were read, so a body with a bad usage shape raised
    inside the lock and counted as both ok and failed, with no latency recorded: inflated rps, p50 0.0."""
    import multiprocessing as mp
    bench = _load("benchmark")

    class Resp:
        status_code = 200
        def __init__(self, body): self._b = body
        def json(self): return self._b

    class Session:
        def __init__(self): self.headers = {}
        def post(self, *a, **k): return Resp({"usage": {"input_tokens": [1], "output_tokens": 7}})

    monkeypatch.setattr(bench.requests, "Session", Session)
    monkeypatch.setattr("signal.signal", lambda *a: None)   # worker ignores SIGINT; not in pytest
    q = mp.Queue()
    bench.worker("http://u", "k", "m", conc=2, in_tok=10, out_tok=5, seconds=0.05, shared=True, q=q)
    r = q.get(timeout=5)
    assert r["ok"] == 0 and r["lat"] == [] and r["fail"] > 0


def test_a_size_list_is_a_per_request_mix(monkeypatch):
    """--input-tokens 1000,8000,16000 must produce prompts of each size, not one size; a bad entry or a
    zero is a one-line exit."""
    import multiprocessing as mp
    bench = _load("benchmark")
    seen = []

    class Resp:
        status_code = 200
        def json(self): return {"usage": {"input_tokens": 1, "output_tokens": 1}}

    class Session:
        def __init__(self): self.headers = {}
        def post(self, url, json, timeout):
            seen.append((len(json["input"]) // 5, json["max_output_tokens"])); return Resp()

    monkeypatch.setattr(bench.requests, "Session", Session)
    monkeypatch.setattr("signal.signal", lambda *a: None)
    q = mp.Queue()
    bench.worker("http://u", "k", "m", conc=2, in_tok=[100, 1000], out_tok=[50, 800], seconds=0.1, shared=True, q=q)
    q.get(timeout=5)
    assert {round(n, -2) for n, _ in seen} == {100, 1000} and {o for _, o in seen} == {50, 800}


def test_a_level_drains_its_in_flight_requests_and_counts_only_the_window(monkeypatch):
    """Exiting with requests open left the engine generating them behind CloudFront, and the next level
    started behind that backlog. The worker now waits for them and does not count the late ones."""
    import multiprocessing as mp
    import time
    bench = _load("benchmark")

    class Resp:
        status_code = 200
        def json(self): return {"usage": {"input_tokens": 1, "output_tokens": 1}}

    class Session:
        def __init__(self): self.headers = {}
        def post(self, *a, **k): time.sleep(0.4); return Resp()

    monkeypatch.setattr(bench.requests, "Session", Session)
    monkeypatch.setattr("signal.signal", lambda *a: None)
    q = mp.Queue(); t0 = time.perf_counter()
    bench.worker("http://u", "k", "m", conc=3, in_tok=[10], out_tok=[5], seconds=0.5, shared=True, q=q)
    r = q.get(timeout=5); elapsed = time.perf_counter() - t0
    assert r["ok"] == 3, "one request per thread completed inside the 0.5 s window"
    assert elapsed >= 0.8, "the second request of each thread was drained, not abandoned"
    assert r["wall"] < 0.6, "the reported wall is the window, not the drain"


def test_turns_resend_the_growing_conversation_on_one_session(monkeypatch):
    """--turns N: every turn after the first carries the previous prompt plus the answer, on the same
    session (cookie jar), so a sticky load balancer can keep it on one engine."""
    import multiprocessing as mp
    bench = _load("benchmark")
    seen = []

    class Resp:
        status_code = 200
        def json(self): return {"usage": {"input_tokens": 1, "output_tokens": 1},
                                "output": [{"content": [{"text": "ANSWER"}]}]}

    class Session:
        def __init__(self): self.headers = {}
        def post(self, url, json, timeout): seen.append((id(self), json["input"])); return Resp()

    monkeypatch.setattr(bench.requests, "Session", Session)
    monkeypatch.setattr("signal.signal", lambda *a: None)
    q = mp.Queue()
    bench.worker("http://u", "k", "m", conc=1, in_tok=[50], out_tok=[5], seconds=0.05, shared=True, q=q, turns=3)
    q.get(timeout=5)
    first = seen[:3]
    assert len({s for s, _ in first}) == 1, "one session per conversation"
    assert first[1][1].startswith(first[0][1]) and "ANSWER" in first[1][1], "turn 2 = turn 1 + answer + follow-up"
    assert first[2][1].startswith(first[1][1])


def test_streaming_measures_time_to_first_token_and_reads_usage_from_the_completed_event(monkeypatch):
    import multiprocessing as mp
    import json as js
    bench = _load("benchmark")

    class Resp:
        status_code = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def iter_lines(self):
            yield b"event: response.reasoning_text.delta"
            yield b"data: " + js.dumps({"type": "response.reasoning_text.delta", "delta": "hmm"}).encode()
            yield b"data: " + js.dumps({"type": "response.output_text.delta", "delta": "hi"}).encode()
            yield b""
            yield b"data: " + js.dumps({"type": "response.completed",
                                        "response": {"usage": {"input_tokens": 12, "output_tokens": 7}}}).encode()

    class Session:
        def __init__(self): self.headers = {}
        def post(self, url, json, timeout, stream): assert json["stream"] is True; return Resp()

    monkeypatch.setattr(bench.requests, "Session", Session)
    monkeypatch.setattr("signal.signal", lambda *a: None)
    q = mp.Queue()
    bench.worker("http://u", "k", "m", conc=1, in_tok=[10], out_tok=[5], seconds=0.05, shared=True, q=q, stream=True)
    r = q.get(timeout=5)
    assert r["ok"] >= 1 and r["in"] == 12 * r["ok"] and r["out"] == 7 * r["ok"]
    assert len(r["ttft"]) == r["ok"] and all(t > 0 for t in r["ttft"])


def test_schema_and_reasoning_effort_land_in_the_request_body(monkeypatch):
    import multiprocessing as mp
    bench = _load("benchmark")
    seen = []

    class Resp:
        status_code = 200
        def json(self): return {"usage": {"input_tokens": 1, "output_tokens": 1}, "output": []}

    class Session:
        def __init__(self): self.headers = {}
        def post(self, url, json, timeout): seen.append(json); return Resp()

    monkeypatch.setattr(bench.requests, "Session", Session)
    monkeypatch.setattr("signal.signal", lambda *a: None)
    q = mp.Queue()
    bench.worker("http://u", "k", "m", conc=1, in_tok=[10], out_tok=[5], seconds=0.05, shared=True, q=q,
                 schema=True, effort="low")
    q.get(timeout=5)
    body = seen[0]
    assert body["text"]["format"]["type"] == "json_schema" and body["text"]["format"]["schema"] == bench.SCHEMA
    assert body["reasoning"] == {"effort": "low"}


def test_size_fleet_rounds_up_and_prices_per_million_tokens():
    sf = _load("size_fleet")
    p = sf.plan(engine_rps=12.8, demand_rps=70, price_per_hour=5.85, gpus_per_instance=1,
                input_tokens=1000, output_tokens=190, headroom=0.15)
    assert p["instances"] == 7, "70 / (12.8 * 0.85) = 6.4 -> 7"
    assert p["fleet_price_per_hour"] == 7 * 5.85
    # demand 70 rps * 1190 tokens * 3600 s = 299.9M tokens/h at $40.95/h
    assert abs(p["price_per_million_tokens_at_demand"] - 40.95 / 299.88) < 1e-3
    assert p["utilisation_at_demand"] < 0.85
    p8 = sf.plan(12.8, 70, 33.14, 8, 1000, 190, 0.15)
    assert p8["instances"] == 1, "one eight-GPU instance holds 102 rps"


def test_quality_summary_merges_harness_results_and_compare_counts_flipped_answers(tmp_path, capsys):
    import json as js
    q = _load("quality")
    a = tmp_path / "a"; b = tmp_path / "b"
    for root, gsm_scores in ((a, [1, 1, 1, 0]), (b, [1, 0, 1, 1])):
        d = root / "loglik-arc_challenge" / "m"; d.mkdir(parents=True)
        (d / "results_1.json").write_text(js.dumps({"results": {
            "arc_challenge": {"acc_norm,none": 0.9, "acc_norm_stderr,none": 0.01, "acc,none": 0.88},
            "wikitext": {"word_perplexity,none": 9.87, "word_perplexity_stderr,none": "N/A"}}}))
        g = root / "gen-gsm8k" / "m"; g.mkdir(parents=True)
        (g / "results_2.json").write_text(js.dumps({"results": {"gsm8k": {"exact_match,strict-match": sum(gsm_scores) / 4,
                                                                          "exact_match_stderr,strict-match": 0.02}}}))
        (g / "samples_gsm8k_x.jsonl").write_text("\n".join(
            js.dumps({"doc_id": i, "filter": "strict-match", "exact_match": float(v)}) for i, v in enumerate(gsm_scores)) + "\n")
    rows = q.summarise(str(a), "fp8", "m", str(tmp_path / "q.csv"))
    out = capsys.readouterr().out
    assert {r["task"] for r in rows} == {"arc_challenge", "wikitext", "gsm8k"}, "results from both passes are merged"
    assert "9.87" in out and "90.0%" in out and "75.0%" in out
    assert (tmp_path / "q.csv").read_text().count("\n") == 4, "header plus three rows"
    q.compare(str(a), str(b))
    out = capsys.readouterr().out
    assert "gsm8k" in out and "50.0%" in out, "two of four shared questions flipped: one gained, one lost"


def test_extraction_reads_bio_tags_parses_leniently_and_scores_mentions():
    """Entity-level scoring on (type, mention): a mention with the right text and the wrong type is both a
    false positive and a false negative, and a fenced or chatty answer still parses."""
    e = _load("extraction")
    names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
    gold = e.entities(["Ada", "Lovelace", "joined", "IBM", "in", "New", "York", "."], [1, 2, 0, 3, 0, 5, 6, 0], names)
    assert gold == {"persons": ["Ada Lovelace"], "organizations": ["IBM"], "locations": ["New York"], "misc": []}
    pred = e.parse('Sure!\n```json\n{"persons": ["Ada  Lovelace"], "organizations": ["New York"], "locations": [], "misc": []}\n```')
    assert pred["persons"] == ["Ada Lovelace"], "whitespace inside a mention is normalised"
    assert e.score(gold, pred) == (1, 1, 2), "one right, New York as an organisation is wrong, IBM and the location are missed"
    assert e.parse("no json here") is None and e.score(gold, None) == (0, 0, 3)
    b = e.body("m", "IBM hired Ada .", [("x", gold)], "schema")
    assert b["response_format"]["json_schema"]["strict"] and b["messages"][-1]["content"] == "IBM hired Ada ." and len(b["messages"]) == 4
    assert "response_format" not in e.body("m", "x", [], "free")


def test_quality_adds_bos_for_tokenizers_that_have_one_and_copies_those_that_will_not(tmp_path):
    """Every Gemma 4 log-likelihood score sat at chance (wikitext perplexity 10,709, winogrande 52.8%) while the chat
    scores were fine: the tokenizer ships add_bos_token false, so the harness sent BOS-less prompts and the model
    scored a 21-token sentence at perplexity 14,931 instead of 4.5. Qwen has no BOS and must be left alone."""
    from types import SimpleNamespace
    q = _load("quality")

    class Tok:
        def __init__(self, bos, adds):
            self.bos_token_id, self.add_bos_token, self.saved = bos, adds, None

        def __call__(self, text, add_special_tokens=False):
            ids = [7, 8]
            return SimpleNamespace(input_ids=([self.bos_token_id] + ids) if add_special_tokens and self.add_bos_token else ids)

        def save_pretrained(self, path):
            self.saved = path

    assert q.bos_arguments(Tok(None, False), "Qwen/x") == ("Qwen/x", ""), "no BOS id: nothing to add"
    assert q.bos_arguments(Tok(1, True), "meta/x") == ("meta/x", ",add_bos_token=True,custom_prefix_token_id=1"), \
        "adds BOS by itself: the harness only has to ask"
    gemma = Tok(2, False)
    path, extra = q.bos_arguments(gemma, "google/x")
    assert extra == ",add_bos_token=True,custom_prefix_token_id=2"
    assert path != "google/x" and gemma.saved == path and gemma.add_bos_token is True, "a copy with the flag on is handed over"


def test_quality_chat_loglik_wraps_prompts_in_the_chat_template_and_skips_wikitext(monkeypatch, tmp_path):
    """Gemma 4's instruction-tuned checkpoints read perplexity 654 to 8,923 on a plain paragraph after <bos> and 12 to
    19 on the same paragraph inside their chat turn, so raw log-likelihood prompts measured nothing. wikitext has no
    turn structure and is skipped rather than reported."""
    import subprocess
    from types import SimpleNamespace
    q = _load("quality")
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append(cmd) or SimpleNamespace(returncode=0))
    a = SimpleNamespace(chat_loglik=True, loglik_tokenizer="tokenized_requests=True,tokenizer=m", url="https://e", key="k",
                        loglik_concurrency=4, concurrency=16, text_prompts=False, tokenizer="")
    q.harness("loglik", "wikitext", [], 60, a, "m", str(tmp_path))
    q.harness("loglik", "arc_challenge", ["--num_fewshot", "25"], 500, a, "m", str(tmp_path))
    q.harness("gen", "gsm8k", ["--num_fewshot", "5"], 500, a, "m", str(tmp_path))
    assert len(calls) == 2, "wikitext was skipped"
    assert "--apply_chat_template" in calls[0] and "--num_fewshot" in calls[0], "the multiple-choice pass is wrapped"
    assert calls[1].count("--apply_chat_template") == 1, "the generative pass was already wrapped and is not doubled"
    assert q.loglik_tokenizer(a, "m") == "tokenized_requests=True,tokenizer=m", "no BOS copy: the template writes <bos>"
