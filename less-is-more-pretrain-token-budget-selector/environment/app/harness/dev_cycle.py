#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pretrain_harness as H

DEFAULT_POOL = "/app/data/dev_pool"
DEFAULT_EVAL = "/app/data/dev_eval"
DEFAULT_CONFIG = "/app/harness/train_config.json"
BUDGET_TOKENS = 125_000_000


def load_selector(path: str):
    spec = importlib.util.spec_from_file_location("dev_select", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(path).resolve().parent))
    spec.loader.exec_module(module)
    return module.select


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selector", default="/app/output/select.py")
    ap.add_argument("--pool", default=DEFAULT_POOL)
    ap.add_argument("--eval", default=DEFAULT_EVAL)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--budget-tokens", type=int, default=BUDGET_TOKENS)
    ap.add_argument("--workdir", default="/app/scratch")
    ap.add_argument("--reference", action="store_true",
                    help="train on the entire pool (the percent-reduction denominator)")
    ap.add_argument("--out", default=None, help="also write the result as JSON here")
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    cfg = H.load_config(args.config)
    pool = H.Pool(args.pool)
    eval_blocks = H.load_eval_blocks(args.eval, cfg["block_size"])
    print(f"pool: {pool.n_docs} documents, {int(pool.offsets[-1])} tokens")
    print(f"eval: {eval_blocks.shape[0]} blocks of {cfg['block_size']} tokens")

    if args.reference:
        doc_ids = list(range(pool.n_docs))
        stats = {"n_accepted": pool.n_docs, "tokens_used": int(pool.offsets[-1])}
    else:
        select = load_selector(args.selector)
        started = time.monotonic()
        raw = select(args.pool, args.budget_tokens, args.workdir)
        elapsed = time.monotonic() - started
        print(f"select() returned in {elapsed:.1f}s")
        doc_ids, _used, stats = H.sanitize_selection(raw, pool.ntokens, args.budget_tokens)
        print(f"selection: {json.dumps(stats)}")
        if not doc_ids:
            print("selection is empty -- this scores 0")
            return 1

    out = H.run_cycle(cfg, pool, doc_ids, eval_blocks)
    out["selection_stats"] = stats
    print(json.dumps({k: v for k, v in out.items() if k != "selection_stats"}, indent=2))
    print(f"perplexity = {out['perplexity']:.6f}   (log-perplexity {math.log(out['perplexity']):.6f})")
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
