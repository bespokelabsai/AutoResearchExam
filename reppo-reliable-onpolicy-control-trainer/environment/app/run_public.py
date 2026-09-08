from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import envs as envs_mod
import panel as panel_mod

HERE = os.path.dirname(os.path.abspath(__file__))
PANEL = os.path.join(HERE, "public_panel.json")
SOLUTION = os.path.join(HERE, "solution")


def _job(args):
    env_json, eval_init, seed, name, harness = args
    spec = envs_mod.EnvSpec.from_json(env_json)
    res = panel_mod.run_one(
        spec, seed, np.asarray(eval_init, dtype=np.float64), SOLUTION, harness,
        python_exe=sys.executable,
    )
    res["env"] = name
    res["seed"] = seed
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", default="", help="comma-separated env names, default all")
    ap.add_argument("--seeds", default="", help="comma-separated run seeds, default all")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--json", default="", help="also write the raw results here")
    args = ap.parse_args()

    panel = json.load(open(PANEL))
    wanted = set(args.envs.split(",")) if args.envs else None
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else panel["run_seeds"]
    selected = [e for e in panel["envs"] if wanted is None or e["name"] in wanted]

    harness = tempfile.mkdtemp(prefix="harness_")
    with open(os.path.join(harness, "child_main.py"), "w") as fh:
        fh.write(open(os.path.join(HERE, "child_main.py")).read())

    jobs = [(e["spec"], e["eval_init"], s, e["name"], harness) for e in selected for s in seeds]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(_job, jobs):
            results.append(r)

    by_name = {e["name"]: e for e in panel["envs"]}
    for r in sorted(results, key=lambda x: (x["env"], x["seed"])):
        env = by_name[r["env"]]
        series = " ".join(
            "-" if x is None else "%.0f" % x for x in r["checkpoint_returns"]
        )
        print(
            "%s (%s) seed=%d  R_rand=%.1f  R_ref=%.1f  cpu=%.1fs  status=%s"
            % (
                r["env"],
                env.get("family", "?"),
                r["seed"],
                env["r_rand"],
                env["r_ref"],
                r["child_cpu_seconds"],
                r["status"],
            )
        )
        print("    checkpoint returns: %s" % series)
        if r["status"] != "ok" and r.get("stderr_tail"):
            print("    child stderr: %s" % r["stderr_tail"].strip().replace("\n", "\n    "))
    if args.json:
        json.dump(results, open(args.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
