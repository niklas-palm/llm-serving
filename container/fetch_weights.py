#!/usr/bin/env python3
"""Download a model's files into the Hugging Face cache before the engine starts.

The engine can download the weights itself, but on very large models it is slow and silent. A 475 GiB
checkpoint sat for about 90 minutes with its two ~95 GiB files unfinished, and the engine logs nothing
while it downloads (c17c-dsv41-tp8); one connection to the CDN gave ~50 MB/s. This script fetches every file over
plain HTTPS, splits large files into 1 GiB byte ranges fetched in parallel, resumes a range that
stalls, checks each file against the hash Hugging Face publishes for it, and logs progress. A prototype
of it finished the same 475 GiB's last 112 GB in ~2.5 minutes and fetched a 155 GiB model in 338 s (c17c).

It writes the standard cache layout (blobs/, snapshots/<commit>/, refs/main), so the engine then finds
every file in the cache and downloads nothing, and files a host already has are skipped. Tasks that
share a host take turns through a lock file. Exit status 0 means every file is in the cache; on any
other result the entrypoint lets the engine download as before.

Usage: fetch_weights.py <model id>   (HF_HOME or HF_HUB_CACHE, and HF_TOKEN, from the environment)
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import http.client
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HUB = "https://huggingface.co"
RANGE_BYTES = 1 << 30   # each file is fetched as byte ranges of this size
CONNECTIONS = 16        # ranges in flight at once, over all files (one connection gave ~50 MB/s)
STALL_SECONDS = 60      # a read that gets no data for this long is retried from the byte it reached
MIN_RATE = 5 << 20      # bytes/s; a connection slower than this over SLOW_WINDOW s is dropped and resumed on a new
SLOW_WINDOW = 30        # one (some CDN connections trickle: on a B200 host two stayed under 1 MB/s for minutes)
TRIES = 20              # per range


def log(msg: str) -> None:
    print(f"[fetch_weights] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def auth() -> dict:
    token = os.environ.get("HF_TOKEN", "")
    return {"Authorization": f"Bearer {token}"} if token else {}


def hub_cache() -> str:
    explicit = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    return explicit or os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")


def ranges(size: int) -> list[tuple[int, int]]:
    """Byte ranges [start, end) that cover size bytes, each at most RANGE_BYTES long."""
    return [(a, min(size, a + RANGE_BYTES)) for a in range(0, size, RANGE_BYTES)]


def new_hash(f: dict):
    """A hash object that ends at the file's id: sha256 for LFS files; for small files git's blob id,
    the sha1 of "blob <size>\\0" and the content."""
    if f.get("lfs"):
        return hashlib.sha256()
    h = hashlib.sha1()
    h.update(b"blob %d\0" % f["size"])
    return h


def blob_id(f: dict) -> str:
    return f["lfs"]["oid"] if f.get("lfs") else f["oid"]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def resolve(repo: str, rev: str, path: str) -> tuple[str, dict]:
    """Where the file's bytes come from, and the headers to send there. Large files redirect to a CDN
    with a signed URL that expires, so each range attempt resolves again. The token goes only to
    huggingface.co, never to the CDN."""
    url = f"{HUB}/{repo}/resolve/{rev}/{urllib.parse.quote(path)}"
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(urllib.request.Request(url, headers=auth(), method="HEAD"), timeout=60):
            return url, auth()
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 303, 307, 308):
            return urllib.parse.urljoin(url, e.headers["Location"]), {}
        raise


def fetch_range(repo: str, rev: str, path: str, dst: str, start: int, end: int) -> None:
    """Write bytes [start, end) of the file into dst at the same offsets. A stalled or dropped connection
    resumes from the byte it reached."""
    pos = start
    for attempt in range(TRIES):
        if pos >= end:
            return
        try:
            url, headers = resolve(repo, rev, path)
            req = urllib.request.Request(url, headers={**headers, "Range": f"bytes={pos}-{end - 1}"})
            with urllib.request.urlopen(req, timeout=STALL_SECONDS) as r:
                if r.status != 206:
                    raise OSError(f"no byte range in the answer (HTTP {r.status})")
                with open(dst, "r+b") as out:
                    out.seek(pos)
                    since, mark = time.monotonic(), pos
                    # read1 returns what has arrived, so a slow connection shows as slow instead of blocking
                    # inside one large read.
                    while pos < end and (chunk := r.read1(min(1 << 20, end - pos))):
                        out.write(chunk)
                        pos += len(chunk)
                        if (now := time.monotonic()) - since >= SLOW_WINDOW:
                            if pos - mark < MIN_RATE * (now - since):
                                raise OSError(f"{(pos - mark) / (now - since) / 2**20:.1f} MB/s, under the minimum")
                            since, mark = now, pos
        except urllib.error.HTTPError as e:
            # No access (a gated model without a token) or no such file: retrying cannot help. A 403 from
            # the CDN is an expired signed URL, which the next attempt resolves again.
            if e.code in (401, 404) or (e.code == 403 and str(e.filename).startswith(HUB)):
                raise RuntimeError(f"{path}: {e.code} {e.reason} from {urllib.parse.urlsplit(str(e.filename)).netloc}") from None
            log(f"{path} bytes {pos}-{end - 1}: HTTP {e.code}; retry {attempt + 1}")
            time.sleep(min(30, 2 ** attempt))
        except (OSError, http.client.HTTPException) as e:   # timeouts, resets, a short read
            if getattr(e, "errno", None) == errno.ENOSPC:
                raise   # a full disk does not get better by retrying
            log(f"{path} bytes {pos}-{end - 1}: {e}; retry {attempt + 1}")
            time.sleep(min(30, 2 ** attempt))
    if pos < end:
        raise RuntimeError(f"{path}: bytes {pos}-{end - 1} still missing after {TRIES} tries")


def fetch_file(repo: str, rev: str, f: dict, blobs: str, snapshot: str, pool: ThreadPoolExecutor) -> int:
    """Put one file into blobs/ (checked against its hash) and link it into the snapshot.

    The ranges are written straight into one file of the final size, so the data goes to disk once. Which
    ranges are complete is kept next to it, so a restarted task fetches only the rest. Not *.incomplete:
    that name is huggingface_hub's, and it would append to our file."""
    path, size, blob = f["path"], f["size"], os.path.join(blobs, blob_id(f))
    if not (os.path.exists(blob) and os.path.getsize(blob) == size):
        tmp, ledger = blob + ".fetch", blob + ".fetch-done"
        if not os.path.exists(tmp):
            with open(tmp, "wb") as out:
                out.truncate(size)
            open(ledger, "w").close()
        done = {int(n) for n in open(ledger).read().split()} if os.path.exists(ledger) else set()
        spans = ranges(size)

        def one_range(i: int) -> None:
            fetch_range(repo, rev, path, tmp, *spans[i])
            with lock, open(ledger, "a") as lg:
                lg.write(f"{i}\n")
                done.add(i)
                if len(spans) > 4:   # a large file: show it moving, not only when it is complete
                    log(f"{path}: {len(done)} of {len(spans)} ranges")

        lock = threading.Lock()
        for fut in [pool.submit(one_range, i) for i in range(len(spans)) if i not in done]:
            fut.result()
        # The blob only gets its name once size and hash match what the repo publishes, so a bad download
        # is never used.
        h = new_hash(f)
        with open(tmp, "rb") as src:
            while chunk := src.read(64 << 20):
                h.update(chunk)
        if os.path.getsize(tmp) != size or h.hexdigest() != blob_id(f):
            os.remove(tmp)
            os.remove(ledger)
            raise RuntimeError(f"{path}: the download does not match the repo's size or hash")
        os.replace(tmp, blob)
        os.remove(ledger)
    link = os.path.join(snapshot, path)
    os.makedirs(os.path.dirname(link), exist_ok=True)
    if not os.path.lexists(link):
        os.symlink(os.path.relpath(blob, os.path.dirname(link)), link)
    return size


def main(repo: str) -> int:
    if os.path.isdir(repo):
        log(f"{repo} is a local directory: nothing to download")
        return 0
    def get(path: str):
        with urllib.request.urlopen(urllib.request.Request(HUB + path, headers=auth()), timeout=60) as r:
            return json.load(r)
    rev = get(f"/api/models/{repo}")["sha"]
    files = [f for f in get(f"/api/models/{repo}/tree/{rev}?recursive=true") if f["type"] == "file"]
    base = os.path.join(hub_cache(), "models--" + repo.replace("/", "--"))
    blobs, snapshot = os.path.join(base, "blobs"), os.path.join(base, "snapshots", rev)
    for d in (blobs, snapshot, os.path.join(base, "refs")):
        os.makedirs(d, exist_ok=True)
    total, done, lock, t0 = sum(f["size"] for f in files), [0], threading.Lock(), time.time()

    def one(f: dict) -> None:
        size = fetch_file(repo, rev, f, blobs, snapshot, ranges_pool)
        with lock:
            done[0] += size
            log(f"{f['path']} ready; {done[0] / 2**30:.1f} of {total / 2**30:.1f} GiB, {time.time() - t0:.0f} s")

    with open(os.path.join(base, ".fetch_weights.lock"), "w") as lockfile:
        fcntl.flock(lockfile, fcntl.LOCK_EX)   # one task per host downloads; the others wait, then find it done
        log(f"{repo} at {rev[:12]}: {len(files)} files, {total / 2**30:.1f} GiB")
        # Two pools: files wait on their ranges, so ranges need workers of their own.
        with ThreadPoolExecutor(CONNECTIONS) as ranges_pool, ThreadPoolExecutor(8) as files_pool:
            for fut in [files_pool.submit(one, f) for f in sorted(files, key=lambda f: -f["size"])]:
                fut.result()
        with open(os.path.join(base, "refs", "main"), "w") as ref:
            ref.write(rev)
    log(f"done: {total / 2**30:.1f} GiB in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
