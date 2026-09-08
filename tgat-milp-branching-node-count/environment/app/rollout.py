from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bnbenv import runner
from bnbenv.runner import STARTUP_LIMIT
from bnbenv.setcover import NODE_LIMIT, TIME_LIMIT

PKG_ROOT = str(Path(__file__).resolve().parent)


def parse_ints(text):
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part[1:]:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def _run(job):
    seed, shift, submission, time_limit, startup = job
    return runner.run_one(seed, shift, submission, PKG_ROOT,
                          time_limit=time_limit, startup_timeout=startup,
                          stderr_to=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-4",
                    help="instance seeds, e.g. '0-19' or '0,3,7'")
    ap.add_argument("--shifts", default="0,1",
                    help="solver seed shifts, e.g. '0,1'")
    ap.add_argument("--submission", default="/app/submission")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--time-limit", type=float, default=TIME_LIMIT)
    ap.add_argument("--startup-limit", type=float, default=STARTUP_LIMIT)
    ap.add_argument("--json", default=None, help="also write records here")
    args = ap.parse_args()

    jobs = [(s, k, args.submission, args.time_limit, args.startup_limit)
            for s in parse_ints(args.seeds) for k in parse_ints(args.shifts)]
    records = []
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for rec in pool.map(_run, jobs):
            records.append(rec)
            print(f"seed={rec['seed']} shift={rec['shift']} "
                  f"nodes={rec['nodes']} status={rec['status']} "
                  f"decisions={rec['decisions']} "
                  f"secs={rec['solve_seconds']}"
                  + (f" error={rec['error']}" if rec["error"] else ""),
                  flush=True)

    nodes = [r["nodes"] for r in records]
    metric = math.exp(sum(math.log(n + 1.0) for n in nodes) / len(nodes)) - 1.0
    capped = sum(1 for r in records if not r["proved_optimal"])
    print(f"\nruns={len(nodes)} counted_at_cap={capped} "
          f"(cap={NODE_LIMIT} nodes / {TIME_LIMIT:.0f}s)")
    print(f"1-shifted geometric mean of nodes = {metric:.2f}")
    if args.json:
        Path(args.json).write_text(json.dumps(records, indent=1))


if __name__ == "__main__":
    main()
