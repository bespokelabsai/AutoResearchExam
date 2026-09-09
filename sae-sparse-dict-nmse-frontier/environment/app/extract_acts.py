#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import re

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

D_MODEL = 768
HIDDEN_STATE_INDEX = 9


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parquet", required=True, help="path to an OpenWebText parquet shard")
    p.add_argument("--doc-start", type=int, default=0, help="first document row, inclusive")
    p.add_argument("--doc-end", type=int, required=True, help="last document row, exclusive")
    p.add_argument("--max-tokens", type=int, required=True, help="stop after this many rows")
    p.add_argument("--out", required=True, help="destination .npy path")
    p.add_argument("--model-dir", default="/opt/assets/gpt2", help="pinned GPT-2 checkpoint")
    p.add_argument("--ctx", type=int, default=64, help="window length in tokens")
    p.add_argument("--batch-windows", type=int, default=256, help="windows per forward pass")
    p.add_argument("--device", default="cuda")
    p.add_argument("--doc-chunk", type=int, default=512, help="documents tokenized at a time")
    p.add_argument("--exclude-doc-indices", default=None,
                   help="optional JSON file holding a list of parquet row indices to skip")
    return p.parse_args()


def normalize_doc(text: str) -> str:
    """Stable content fingerprint of a document, used for cross-shard duplicate removal."""
    return hashlib.sha256(
        re.sub(r"\s+", " ", text.lower()).strip()[:2000].encode("utf-8")).hexdigest()


def iter_windows(texts, tokenizer, ctx, doc_chunk):
    """Yield lists of int windows, document by document, in parquet row order."""
    for start in range(0, len(texts), doc_chunk):
        chunk = texts[start:start + doc_chunk]
        encoded = tokenizer(chunk, add_special_tokens=False)["input_ids"]
        for ids in encoded:
            n_windows = len(ids) // ctx
            for w in range(n_windows):
                yield ids[w * ctx:(w + 1) * ctx]


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.float32)
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad_(False)

    excluded: set[int] = set()
    if args.exclude_doc_indices:
        with open(args.exclude_doc_indices) as fh:
            excluded = {int(i) for i in json.load(fh)}

    table = pq.read_table(args.parquet, columns=["text"])
    all_texts = table.column("text").to_pylist()
    texts = [t for i, t in enumerate(all_texts[args.doc_start:args.doc_end], args.doc_start)
             if i not in excluded]
    del table, all_texts

    out = np.empty((args.max_tokens, D_MODEL), dtype=np.float16)
    written = 0
    batch: list[list[int]] = []

    def flush(batch_windows):
        nonlocal written
        if not batch_windows:
            return
        ids = torch.tensor(batch_windows, dtype=torch.long, device=device)
        with torch.no_grad():
            hidden = model(input_ids=ids, output_hidden_states=True,
                           use_cache=False).hidden_states[HIDDEN_STATE_INDEX]
        acts = hidden.reshape(-1, D_MODEL).float()
        acts = acts - acts.mean(dim=1, keepdim=True)
        acts = acts / acts.norm(dim=1, keepdim=True).clamp_min(1e-12)
        take = min(acts.shape[0], args.max_tokens - written)
        out[written:written + take] = acts[:take].to(torch.float16).cpu().numpy()
        written += take

    for window in iter_windows(texts, tokenizer, args.ctx, args.doc_chunk):
        batch.append(window)
        if len(batch) == args.batch_windows:
            flush(batch)
            batch = []
            if written >= args.max_tokens:
                break
    if written < args.max_tokens:
        flush(batch)

    if written < args.max_tokens:
        raise SystemExit(
            f"only {written} activation rows available from {args.parquet} documents "
            f"[{args.doc_start}, {args.doc_end}); asked for {args.max_tokens}")

    parent = os.path.dirname(os.path.abspath(args.out))
    if parent:
        os.makedirs(parent, exist_ok=True)
    np.save(args.out, out)
    print(f"wrote {out.shape} float16 to {args.out}")


if __name__ == "__main__":
    main()
