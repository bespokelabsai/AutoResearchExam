from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import statistics
import sys
import tempfile
from pathlib import Path

import numpy as np

HARNESS = Path(__file__).resolve().parent / "harness"
sys.path.insert(0, str(HARNESS))

import runner
from worker_client import CALL_BUDGET_SEC, IMPORT_BUDGET_SEC


def read_texts(path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line)["text"] for line in fh]


def parse_seeds(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", default="a,b,c")
    ap.add_argument("--seeds", default="0-5")
    ap.add_argument("--module-dir", default="/app/output")
    ap.add_argument("--entry", default="prune")
    ap.add_argument("--workers", type=int, default=runner.MAX_CONCURRENT_RUNS)
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent / "data"))
    ap.add_argument("--json", default=None, help="also write the summary as JSON here")
    args = ap.parse_args()

    DATA = Path(args.data)
    draws = [d.strip() for d in args.draws.split(",") if d.strip()]
    seeds = parse_seeds(args.seeds)
    eval_y = np.load(DATA / "dev_eval_y.npy")

    scratch = tempfile.mkdtemp(prefix="dev_run_")
    specs = []
    for draw in draws:
        texts = read_texts(DATA / f"pool_{draw}.jsonl.gz")
        pool_texts_path = runner.stage_pool_texts(texts, Path(scratch) / f"pool_{draw}.json")
        pool_labels = np.load(DATA / f"pool_{draw}_y.npy")
        pool_X = np.load(DATA / f"pool_{draw}_X.npy")
        eval_X = np.load(DATA / f"eval_{draw}_X.npy")
        for seed in seeds:
            specs.append({"draw": draw, "seed": seed, "pool_labels": pool_labels,
                          "pool_X": pool_X, "eval_y": eval_y, "eval_X": eval_X,
                          "pool_texts_path": pool_texts_path})

    print(f"{len(specs)} runs: draws={draws} seeds={seeds} "
          f"import_budget={IMPORT_BUDGET_SEC:g}s call_budget={CALL_BUDGET_SEC:g}s")
    err_root = Path(scratch) / "err"
    err_root.mkdir(exist_ok=True)
    staged = Path(scratch) / "deliverable"
    shutil.copytree(args.module_dir, staged, symlinks=True)
    runner.make_readonly(staged)
    results = runner.execute_runs(
        specs, python_exe=sys.executable, worker_script=str(HARNESS / "prune_worker.py"),
        module_dir=str(staged), entry_module=args.entry, scratch_root=scratch,
        as_user=None, new_session=os.setsid, max_workers=args.workers,
        stderr_root=str(err_root))

    failed = [r for r in results if r["error"]]
    for r in failed[:5]:
        print(f"  FAILED draw={r['draw']} seed={r['seed']}: {r['error']}")
    if failed:
        print(f"{len(failed)}/{len(results)} runs failed -- a submission with any failing run "
              f"is invalid.  Worker stderr: {scratch}/err")
        if args.json:
            Path(args.json).write_text(json.dumps(
                {"ok": False, "failed": len(failed), "first_error": failed[0]["error"]}))
        return 1

    per_draw = {}
    for r in results:
        per_draw.setdefault(r["draw"], []).append(r["final"])
    print()
    for draw in draws:
        vals = per_draw[draw]
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"draw {draw}: final macro-F1 mean {statistics.fmean(vals):8.4f}  "
              f"sd {sd:6.4f}  n {len(vals)}")
    allv = [r["final"] for r in results]
    print(f"OVERALL mean final macro-F1: {statistics.fmean(allv):.4f}  (n={len(allv)})")
    curve = np.mean(np.array([r["curve"] for r in results]), axis=0)
    print("mean per-iteration macro-F1: " + "  ".join(f"{v:.3f}" for v in curve))
    slowest = max(max(r["timings"]) for r in results)
    print(f"slowest prune call: {slowest:.2f}s of the {CALL_BUDGET_SEC:g}s budget")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"ok": True, "overall_mean": statistics.fmean(allv),
             "per_draw_mean": {d: statistics.fmean(v) for d, v in per_draw.items()},
             "per_draw_sd": {d: (statistics.stdev(v) if len(v) > 1 else 0.0)
                             for d, v in per_draw.items()},
             "per_iteration_mean": [float(v) for v in curve],
             "slowest_call_sec": slowest, "n_runs": len(allv)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
