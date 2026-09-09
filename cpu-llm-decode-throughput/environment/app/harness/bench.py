from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import metric_core

DATA_DIR = "/app/data"
REFERENCE = "/app/reference/reference_engine.py"
MODEL_DIR = "/opt/model"
WARMUP = "/app/data/warmup_workload.json"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", default="/app/output/engine.py")
    ap.add_argument("--draws", default="0", help="comma-separated dev workload indices, or 'all'")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N requests")
    args = ap.parse_args(argv)

    indices = [0, 1, 2, 3] if args.draws == "all" else [int(x) for x in args.draws.split(",")]
    workloads = []
    for i in indices:
        path = os.path.join(DATA_DIR, f"dev_workload_{i}.json")
        with open(path) as fh:
            reqs = json.load(fh)
        if args.limit:
            reqs = reqs[: args.limit]
        workloads.append((f"dev_workload_{i}", reqs))

    if not os.path.isfile(args.engine):
        print(f"no engine at {args.engine}")
        return 1

    with tempfile.TemporaryDirectory(prefix="bench-") as work_root:
        staged = os.path.join(work_root, "candidate")
        shutil.copytree(os.path.dirname(os.path.abspath(args.engine)), staged)
        for root, dirs, files in os.walk(staged):
            for name in files:
                os.chmod(os.path.join(root, name), 0o400)
            os.chmod(root, 0o500)
        rounds = metric_core.run_rounds(
            REFERENCE,
            os.path.join(staged, os.path.basename(args.engine)),
            MODEL_DIR,
            WARMUP,
            workloads,
            work_root,
        )
        for root, dirs, files in os.walk(staged):
            os.chmod(root, 0o700)
            for name in files:
                os.chmod(os.path.join(root, name), 0o600)
    summary = metric_core.summarise(rounds)
    print()
    for r in rounds:
        if r.get("ok"):
            print(
                f"  {r['workload']}: speedup {r['speedup']:.3f}x  "
                f"(reference {r['reference_s']:.2f}s, candidate {r['candidate_s']:.2f}s, "
                f"agreement {r['matched_positions']}/{r['total_positions']})"
            )
        else:
            print(f"  {r['workload']}: FAILED - {r.get('detail')}")
    print()
    print(f"token agreement : {summary['agreement']:.4f} "
          f"(must be >= {metric_core.AGREEMENT_MIN})")
    print(f"speedup S       : {summary['speedup']:.4f}")
    if summary["detail"]:
        print(f"note            : {summary['detail']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
