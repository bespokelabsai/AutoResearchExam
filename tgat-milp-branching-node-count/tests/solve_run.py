from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bnbenv import runner


class _Expired(Exception):
    pass


def _on_alarm(signum, frame):
    raise _Expired("run driver exceeded its own hard limit")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--shift", type=int, required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--host-pkg-root", required=True)
    ap.add_argument("--python", default="/usr/local/bin/python3")
    ap.add_argument("--time-limit", type=float, required=True)
    ap.add_argument("--startup-timeout", type=float, required=True)
    ap.add_argument("--launch-prefix", default="")
    ap.add_argument("--drop-to-uid", type=int, default=None)
    args = ap.parse_args()

    prefix = [p for p in args.launch_prefix.split(",") if p]
    signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(int(args.time_limit + args.startup_timeout + 60))
    try:
        record = runner.run_one(
            args.seed, args.shift, args.submission, args.host_pkg_root,
            python_exe=args.python, launch_prefix=prefix,
            time_limit=args.time_limit,
            startup_timeout=args.startup_timeout,
            drop_to_uid=args.drop_to_uid)
    except BaseException as exc:
        from bnbenv.setcover import NODE_LIMIT
        record = {
            "seed": args.seed, "shift": args.shift, "nodes": NODE_LIMIT,
            "raw_nodes": None, "status": "run_driver_failed",
            "proved_optimal": False, "decisions": 0, "solve_seconds": 0.0,
            "error": f"{type(exc).__name__}: {exc}"[:200],
            "extra_processes": 0,
        }
    finally:
        signal.alarm(0)
    print(json.dumps(record), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
