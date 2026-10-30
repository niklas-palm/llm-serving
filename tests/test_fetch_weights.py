"""container/fetch_weights.py against a fake Hugging Face on localhost: no network, no credentials.

The engine's own download of a 475 GiB checkpoint sat for ~90 minutes (c17c-dsv41-tp8); this script replaces
it on the start path, so a bug in it would stop every new instance from getting its weights."""
import hashlib
import importlib.util
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


def _load():
    spec = importlib.util.spec_from_file_location("fetch_weights", os.path.join(HERE, "..", "container", "fetch_weights.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BIG = os.urandom(5 * 1024 * 1024 + 123)   # an LFS file, fetched over several byte ranges
SMALL = b'{"architectures": ["Toy"]}'      # a small file, served by the hub itself
REV = "abc123"


def git_oid(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


TREE = [
    {"type": "file", "path": "config.json", "size": len(SMALL), "oid": git_oid(SMALL)},
    {"type": "file", "path": "sub/model.safetensors", "size": len(BIG), "oid": "x",
     "lfs": {"oid": hashlib.sha256(BIG).hexdigest(), "size": len(BIG)}},
    {"type": "directory", "path": "sub"},
]


class FakeHub:
    """Serves the model API, resolve (with a redirect to a 'CDN' for the LFS file) and byte ranges."""

    def __init__(self, gated=False, cut_first=0, corrupt=False, fail_after=None, slow_first=0):
        self.gated, self.cut_left, self.corrupt, self.fail_after = gated, cut_first, corrupt, fail_after
        self.slow_left = slow_first
        self.cdn_auth = []   # Authorization headers the CDN saw: must stay empty

        hub = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _range(self, data):
                a, b = self.headers["Range"].split("=")[1].split("-")
                part = data[int(a):int(b) + 1]
                if hub.corrupt:
                    part = bytes(len(part))
                self.send_response(206)
                self.send_header("Content-Length", str(len(part)))
                self.end_headers()
                if hub.cut_left > 0:   # drop the connection half way, once per cut
                    hub.cut_left -= 1
                    self.wfile.write(part[: len(part) // 2])
                    return
                if hub.slow_left > 0:   # trickle: data keeps coming, far too slowly, until the client gives up
                    hub.slow_left -= 1
                    try:
                        for i in range(0, len(part), 4096):
                            self.wfile.write(part[i:i + 4096])
                            self.wfile.flush()
                            time.sleep(0.2)
                    except OSError:
                        pass
                    return
                self.wfile.write(part)

            def do_HEAD(self):
                if hub.gated and not self.headers.get("Authorization"):
                    self.send_response(401)
                    self.end_headers()
                elif self.path.endswith("model.safetensors"):
                    self.send_response(302)
                    self.send_header("Location", f"http://127.0.0.1:{hub.port}/cdn/model.safetensors?sig=1")
                    self.end_headers()
                else:
                    self.send_response(200)
                    self.end_headers()

            def do_GET(self):
                if self.path.startswith("/api/models/org/toy/tree/"):
                    body = json.dumps(TREE).encode()
                elif self.path.startswith("/api/models/org/toy"):
                    body = json.dumps({"sha": REV}).encode()
                elif self.path.startswith("/cdn/"):
                    hub.cdn_auth.append(self.headers.get("Authorization"))
                    if hub.fail_after is not None and len(hub.cdn_auth) > hub.fail_after:
                        self.send_response(500)
                        self.end_headers()
                        return
                    return self._range(BIG)
                elif self.path.endswith("config.json"):
                    return self._range(SMALL)
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


@pytest.fixture
def fw(monkeypatch, tmp_path):
    mod = _load()
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(mod, "RANGE_BYTES", 1 << 20)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_TOKEN", "secret")
    return mod


def _use(fw, monkeypatch, hub):
    monkeypatch.setattr(fw, "HUB", f"http://127.0.0.1:{hub.port}")
    return os.path.join(os.environ["HF_HOME"], "hub", "models--org--toy")


def test_writes_the_hub_cache_layout_with_checked_files(fw, monkeypatch):
    """The engine finds the files only in the exact layout huggingface_hub uses."""
    hub = FakeHub(cut_first=2)   # two dropped connections, resumed from the byte they reached
    base = _use(fw, monkeypatch, hub)
    assert fw.main("org/toy") == 0
    assert hub.cut_left == 0 and len(hub.cdn_auth) > len(fw.ranges(len(BIG))) > 1   # several ranges, two resumed
    snap = os.path.join(base, "snapshots", REV)
    assert open(os.path.join(base, "refs", "main")).read() == REV
    assert open(os.path.join(snap, "config.json"), "rb").read() == SMALL
    assert open(os.path.join(snap, "sub", "model.safetensors"), "rb").read() == BIG
    assert os.path.islink(os.path.join(snap, "sub", "model.safetensors"))
    assert not [n for n in os.listdir(os.path.join(base, "blobs")) if ".fetch" in n]


def test_the_token_never_reaches_the_cdn(fw, monkeypatch):
    """The signed CDN URL needs no token; sending it there would hand it to a third host."""
    hub = FakeHub()
    _use(fw, monkeypatch, hub)
    fw.main("org/toy")
    assert hub.cdn_auth and all(a is None for a in hub.cdn_auth)


def test_a_second_run_downloads_nothing(fw, monkeypatch):
    """Several tasks share one host cache; the ones after the first must find it done."""
    hub = FakeHub()
    _use(fw, monkeypatch, hub)
    fw.main("org/toy")
    seen = len(hub.cdn_auth)
    fw.main("org/toy")
    assert len(hub.cdn_auth) == seen


def test_a_file_that_does_not_match_its_hash_is_never_used(fw, monkeypatch):
    base = _use(fw, monkeypatch, FakeHub(corrupt=True))
    with pytest.raises(RuntimeError, match="hash"):
        fw.main("org/toy")
    assert not os.path.exists(os.path.join(base, "blobs", hashlib.sha256(BIG).hexdigest()))


def test_no_access_fails_at_once(fw, monkeypatch):
    """A gated model without a token: retrying for minutes would only delay the engine's own error."""
    monkeypatch.delenv("HF_TOKEN")
    _use(fw, monkeypatch, FakeHub(gated=True))
    with pytest.raises(RuntimeError, match="401"):
        fw.main("org/toy")


def test_ranges_cover_every_byte_once(fw):
    for size in (0, 1, (1 << 20) - 1, 1 << 20, (5 << 20) + 7):
        spans = fw.ranges(size)
        assert sum(b - a for a, b in spans) == size
        assert all(spans[i][1] == spans[i + 1][0] for i in range(len(spans) - 1))


def test_a_restarted_task_fetches_only_the_ranges_it_is_missing(fw, monkeypatch):
    """A task replaced part way through a large download must not start the file again."""
    monkeypatch.setattr(fw, "TRIES", 2)
    monkeypatch.setattr(fw, "CONNECTIONS", 1)   # ranges in order, so the first run completes exactly two
    hub = FakeHub(fail_after=2)
    _use(fw, monkeypatch, hub)
    with pytest.raises(RuntimeError):
        fw.main("org/toy")
    hub.fail_after, before = None, len(hub.cdn_auth)
    assert fw.main("org/toy") == 0
    assert len(hub.cdn_auth) - before == len(fw.ranges(len(BIG))) - 2


def test_a_connection_that_trickles_is_dropped_and_resumed(fw, monkeypatch):
    """On a B200 host two CDN connections stayed under 1 MB/s for minutes without ever stalling, so the stall
    timeout never fired and the start waited on them (c17d). A connection under the minimum rate is cut."""
    monkeypatch.setattr(fw, "MIN_RATE", 2 << 20)
    monkeypatch.setattr(fw, "SLOW_WINDOW", 0.3)
    hub = FakeHub(slow_first=1)
    base = _use(fw, monkeypatch, hub)
    t = time.monotonic()
    assert fw.main("org/toy") == 0
    assert hub.slow_left == 0 and time.monotonic() - t < 20
    assert open(os.path.join(base, "snapshots", REV, "sub", "model.safetensors"), "rb").read() == BIG
