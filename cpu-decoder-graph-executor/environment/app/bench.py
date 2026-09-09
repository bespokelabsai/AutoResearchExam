#!/usr/bin/env python3
import argparse
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness

HERE = os.path.dirname(os.path.abspath(__file__))
PUBLIC_SEEDS = list(range(16))
CORE = ("worker.py", "graph_spec.py", "reference_executor.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/app/output", help="directory holding executor.py")
    ap.add_argument("--seeds", default=",".join(str(s) for s in PUBLIC_SEEDS))
    ap.add_argument("--json", default=None, help="also write the full record here")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    run = tempfile.mkdtemp(prefix="bench_")
    try:
        for f in CORE:
            shutil.copy2(os.path.join(HERE, f), os.path.join(run, f))
        shutil.copytree(args.dir, os.path.join(run, "submission"))
        rec = harness.measure(seeds, run, python=sys.executable)
    finally:
        shutil.rmtree(run, ignore_errors=True)

    print("%5s %5s %6s %6s %11s %11s %8s %10s %10s"
          % ("seed", "T", "d", "L", "ref best ms", "sub best ms", "ratio",
             "max abs", "rel fro"))
    for r in rec["per_instance"]:
        print("%5d %5d %6d %6d %11.2f %11.2f %8.3f %10.2e %10.2e%s"
              % (r["seed"], r["shape"]["T"], r["shape"]["d_model"],
                 r["shape"]["n_layers"], r["ref_best_s"] * 1e3, r["sub_best_s"] * 1e3,
                 r["ratio"], r["max_abs_err"], r["rel_fro_err"],
                 "" if r["equivalent"] else "   NOT EQUIVALENT"))
    if rec["failure"]:
        print("\nFAILED: %s" % rec["failure"])
        if rec["stderr_tail"]:
            print(rec["stderr_tail"])
    print("\ninstances %d, equivalent %d, geometric-mean speedup %.4f"
          % (rec["n_instances"], rec["n_equivalent"], rec["metric"]))
    if args.json:
        harness.dump(rec, args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
