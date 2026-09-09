#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import runner
import sopcc

CPU_SECONDS_PER_INSTANCE = 75.0
WALL_SECONDS_PER_INSTANCE = 150.0
ADDRESS_SPACE_BYTES = 6 * 1024 ** 3
FILE_SIZE_BYTES = 64 * 1024 ** 2
MAX_DELIVERABLE_BYTES = 50 * 1024 * 1024


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution", default="/app/solution",
                        help="directory holding policy.py")
    parser.add_argument("--draws", default="0,1,2,3",
                        help="comma-separated dev draw ids to evaluate")
    parser.add_argument("--instances", type=int, default=12,
                        help="instances per draw (up to %d)" % sopcc.DEV_INSTANCES_PER_DRAW)
    parser.add_argument("--episodes", type=int, default=sopcc.EPISODES_PER_INSTANCE,
                        help="episodes per instance")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args(argv)

    ok, reason, size = runner.validate_solution_dir(args.solution, MAX_DELIVERABLE_BYTES)
    if not ok:
        print(f"deliverable rejected: {reason}")
        return 1
    print(f"deliverable: {args.solution} ({size} bytes)")

    draws = [int(d) for d in args.draws.split(",") if d.strip() != ""]
    full = sopcc.make_dev_panel()
    panel = [e for d in draws
             for e in [x for x in full if x["draw"] == d][:args.instances]]
    if not panel:
        print("no instances selected")
        return 1

    sandbox = runner.Sandbox(
        cpu_seconds=CPU_SECONDS_PER_INSTANCE,
        timeout=WALL_SECONDS_PER_INSTANCE,
        address_space_bytes=ADDRESS_SPACE_BYTES,
        file_size_bytes=FILE_SIZE_BYTES,
    )
    staging = tempfile.mkdtemp(prefix="sopcc-dev-")
    os.chmod(staging, 0o755)
    candidate = runner.stage_candidate(args.solution, os.path.join(staging, "solution"))
    summary = runner.grade_panel(
        panel, sandbox, candidate,
        worker_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "policy_worker.py"),
        workers=max(1, args.workers), episodes=args.episodes,
    )

    print(f"\n{len(panel)} instances x {args.episodes} episodes "
          f"= {summary['episodes']} episodes")
    print(f"{'draw':>6} {'metric':>9} {'R':>9} {'F':>8} {'penalty':>8}")
    for draw in sorted(summary["per_draw"]):
        s = summary["per_draw"][draw]
        print(f"{draw:>6} {s['metric']:>9.3f} {s['R']:>9.3f} "
              f"{s['F']:>8.3f} {s['penalty']:>8.3f}")
    print(f"{'mean':>6} {summary['draw_metric_mean']:>9.3f} {summary['R']:>9.3f} "
          f"{summary['F']:>8.3f} {summary['penalty']:>8.3f}")
    print(f"\npooled metric {summary['metric']:.4f}  "
          f"(R={summary['R']:.4f}, F={summary['F']:.4f}, "
          f"per-episode delivered sd={summary['delivered_std']:.3f})")
    print(f"max per-instance cpu {summary['max_cpu_seconds']:.1f}s "
          f"(cap {CPU_SECONDS_PER_INSTANCE:.0f}s), "
          f"wall {summary['max_wall_seconds']:.1f}s "
          f"(cap {WALL_SECONDS_PER_INSTANCE:.0f}s), "
          f"aborted instances {summary['aborted_instances']}")

    reasons = collections.Counter()
    for report in summary["reports"]:
        for outcome in report["episodes"]:
            if outcome["failed"]:
                reasons[outcome["reason"].split(":")[0]] += 1
    if reasons:
        print("failures: " + ", ".join(f"{k}={v}" for k, v in reasons.most_common(6)))
    for report in summary["reports"]:
        if report["abort"]:
            print(f"\nfirst aborted instance: {report['abort']}")
            if report["stderr_tail"].strip():
                print("stderr tail:\n" + report["stderr_tail"][-1200:])
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
