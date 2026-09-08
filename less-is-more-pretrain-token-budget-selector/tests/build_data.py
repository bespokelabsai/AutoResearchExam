from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

EOT_TOKEN = 50256
_WS = re.compile(r"\s+")


def norm_hash(text: str) -> str:
    return hashlib.sha256(_WS.sub(" ", text).strip().lower().encode("utf-8")).hexdigest()[:32]


def host_of(url: str) -> str:
    u = url.split("://", 1)[-1]
    host = u.split("/", 1)[0].split("?", 1)[0].split("@")[-1].split(":")[0].lower()
    if host.startswith("www."):
        host = host[4:]
    return host or "unknown"


def read_docs(paths, skip_docs: int = 0):
    seen = 0
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                seen += 1
                if seen <= skip_docs:
                    continue
                yield json.loads(line)


def encode_stream(tokenizer_json: str, docs_iter, batch: int = 2000):
    tok = Tokenizer.from_file(tokenizer_json)
    buf = []
    for rec in docs_iter:
        buf.append(rec)
        if len(buf) == batch:
            for rec, enc in zip(buf, tok.encode_batch([r["text"] for r in buf])):
                yield rec, enc.ids
            buf = []
    if buf:
        for rec, enc in zip(buf, tok.encode_batch([r["text"] for r in buf])):
            yield rec, enc.ids


def build_eval(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    chunks, hashes, urls = [], [], []
    total = 0
    n_docs = 0
    for rec, ids in encode_stream(args.tokenizer, read_docs(args.shards, args.skip_docs)):
        ids = ids + [EOT_TOKEN]
        chunks.append(np.asarray(ids, dtype=np.uint16))
        hashes.append(norm_hash(rec["text"]))
        urls.append(rec.get("url", ""))
        total += len(ids)
        n_docs += 1
        if total >= args.target_tokens:
            break
    flat = np.concatenate(chunks)[: args.target_tokens]
    np.save(out / "tokens.npy", flat)
    meta = {"kind": "eval", "n_docs": n_docs, "n_tokens": int(flat.shape[0]),
            "shards": [Path(s).name for s in args.shards]}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    (out / "guard.json").write_text(json.dumps(
        {"hashes": sorted(set(hashes)), "urls": sorted(set(urls))}))
    print(f"[eval] {meta}", flush=True)


def _merged_stream(args):
    """Read the pool's corpus files.

    A pool may be drawn from two corpus variants of the same crawl -- a filtered one and
    a minimally-filtered one -- in which case documents are interleaved deterministically
    in proportion to their token targets, before any grouping or ordering happens.
    """
    a = encode_stream(args.tokenizer, read_docs(args.shards, args.skip_docs))
    if not args.shards_b:
        yield from a
        return
    b = encode_stream(args.tokenizer, read_docs(args.shards_b, args.skip_docs_b))
    want_a = args.target_tokens - args.target_tokens_b
    want_b = args.target_tokens_b
    got_a = got_b = 0
    a_done = b_done = False
    while not (a_done and b_done):
        take_b = (not b_done) and (b_done or a_done or
                                   got_b * max(1, want_a) <= got_a * max(1, want_b))
        if take_b:
            try:
                rec, ids = next(b)
                got_b += len(ids) + 1
                yield rec, ids
                continue
            except StopIteration:
                b_done = True
                continue
        try:
            rec, ids = next(a)
            got_a += len(ids) + 1
            yield rec, ids
        except StopIteration:
            a_done = True


def build_pool(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    guard_hashes, guard_urls = set(), set()
    for g in args.guard or []:
        blob = json.loads(Path(g).read_text())
        guard_hashes.update(blob["hashes"])
        guard_urls.update(blob["urls"])

    kept, seen, seen_urls = [], set(), set()
    total = 0
    n_seen = n_guard_hash = n_guard_url = n_dup = n_empty = n_dup_url = 0
    for rec, ids in _merged_stream(args):
        n_seen += 1
        text = rec["text"]
        if not ids:
            n_empty += 1
            continue
        h = norm_hash(text)
        url = rec.get("url", "")
        if h in guard_hashes:
            n_guard_hash += 1
            continue
        if url and url in guard_urls:
            n_guard_url += 1
            continue
        if h in seen:
            n_dup += 1
            continue
        if url and url in seen_urls:
            n_dup_url += 1
            continue
        seen.add(h)
        if url:
            seen_urls.add(url)
        ids = ids + [EOT_TOKEN]
        kept.append((url, text, np.asarray(ids, dtype=np.uint16)))
        total += len(ids)
        if total >= args.target_tokens:
            break

    by_host: dict[str, list[int]] = {}
    host_tokens: dict[str, int] = {}
    for i, (url, _text, ids) in enumerate(kept):
        host = host_of(url)
        by_host.setdefault(host, []).append(i)
        host_tokens[host] = host_tokens.get(host, 0) + int(ids.shape[0])
    order: list[int] = []
    for host in sorted(host_tokens, key=lambda h: (-host_tokens[h], h)):
        order.extend(by_host[host])

    offsets = np.zeros(len(order) + 1, dtype=np.int64)
    with open(out / "docs.jsonl", "w", encoding="utf-8") as fh:
        for new_id, old in enumerate(order):
            url, text, ids = kept[old]
            offsets[new_id + 1] = offsets[new_id] + int(ids.shape[0])
            fh.write(json.dumps({"id": new_id, "text": text, "url": url,
                                 "ntokens": int(ids.shape[0])}, ensure_ascii=False) + "\n")
    flat = np.empty(int(offsets[-1]), dtype=np.uint16)
    for new_id, old in enumerate(order):
        flat[offsets[new_id]:offsets[new_id + 1]] = kept[old][2]
    np.save(out / "tokens.npy", flat)
    np.save(out / "offsets.npy", offsets)

    meta = {"kind": "pool", "n_docs": len(order), "n_tokens": int(offsets[-1]),
            "n_hosts": len(host_tokens), "n_seen": n_seen,
            "dropped_eval_hash": n_guard_hash, "dropped_eval_url": n_guard_url,
            "dropped_internal_dup": n_dup, "dropped_internal_dup_url": n_dup_url,
            "dropped_empty": n_empty,
            "shards": [Path(s).name for s in args.shards], "skip_docs": args.skip_docs}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    (out / "guard.json").write_text(json.dumps(
        {"hashes": sorted(seen), "urls": sorted(u for u in seen_urls if u)}))
    print(f"[pool] {meta}", flush=True)

    pool_hashes = seen
    assert not (pool_hashes & guard_hashes), "pool/eval text-hash overlap survived the guard"
    print(f"[pool] split integrity OK: {len(pool_hashes)} pool hashes, "
          f"{len(guard_hashes)} eval hashes, intersection 0", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["eval", "pool"])
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", required=True, help="path to a HF tokenizer.json")
    ap.add_argument("--target-tokens", type=int, required=True)
    ap.add_argument("--shards-b", nargs="*", default=None,
                    help="second corpus variant to interleave into the pool")
    ap.add_argument("--target-tokens-b", type=int, default=0,
                    help="how many of --target-tokens come from --shards-b")
    ap.add_argument("--skip-docs-b", type=int, default=0)
    ap.add_argument("--skip-docs", type=int, default=0,
                    help="skip this many leading documents of the shard list (disjoint draws)")
    ap.add_argument("--guard", nargs="*", default=None,
                    help="eval guard.json files whose documents must not enter the pool")
    args = ap.parse_args(argv)
    (build_eval if args.kind == "eval" else build_pool)(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
