from __future__ import annotations

import collections
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import grader_core
import runner
import sopcc

TESTS = Path(__file__).resolve().parent
DELIVERABLE = Path("/app/solution")
STAGING = Path("/tmp/sopcc-candidate")
LOGS = Path("/logs/verifier")


MAX_DELIVERABLE_BYTES = 50 * 1024 * 1024
CPU_SECONDS_PER_INSTANCE = 75.0
WALL_SECONDS_PER_INSTANCE = 150.0
ADDRESS_SPACE_BYTES = 6 * 1024 ** 3
FILE_SIZE_BYTES = 64 * 1024 ** 2
OPEN_FILES = 256
AGENT_UID = 1001
AGENT_GID = 1001
WORKERS = 8








SANDBOX = runner.Sandbox(
    cpu_seconds=CPU_SECONDS_PER_INSTANCE,
    timeout=WALL_SECONDS_PER_INSTANCE,
    address_space_bytes=ADDRESS_SPACE_BYTES,
    file_size_bytes=FILE_SIZE_BYTES,
    open_files=OPEN_FILES,
    run_as_uid=AGENT_UID,
    run_as_gid=AGENT_GID,
    new_session=True,
)


def _write_metric(payload: dict) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "metric.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _redacted(split: str) -> bool:
    """The intermediate grader's stdout goes back to the agent, so it stays aggregate-only."""
    return split != "final"


@pytest.fixture(scope="session")
def graded() -> dict:
    """Grade the sealed panel exactly once, and write the metric file."""
    split = os.environ.get("GRADE_SPLIT", "final")
    payload = {
        "split": split,
        "metric": 0.0,
        "reward": 0.0,
        "R": 0.0,
        "F": 1.0,
        "penalty": 0.0,
        "valid": False,
        "reason": "grading did not complete",
    }
    _write_metric(payload)

    if split not in grader_core.SPLITS:
        payload["reason"] = f"unknown split {split!r}"
        _write_metric(payload)
        return payload

    ok, reason, size = runner.validate_solution_dir(str(DELIVERABLE), MAX_DELIVERABLE_BYTES)
    payload["deliverable_bytes"] = size
    if not ok:
        payload["reason"] = f"deliverable rejected: {reason}"
        _write_metric(payload)
        return payload

    panel = grader_core.load_panel(split)
    STAGING.parent.mkdir(parents=True, exist_ok=True)
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir()
    STAGING.chmod(0o755)
    candidate = runner.stage_candidate(str(DELIVERABLE), str(STAGING / "solution"))




    worker = STAGING / "policy_worker.py"
    shutil.copyfile(TESTS / "policy_worker.py", worker)
    os.chown(worker, 0, 0)
    worker.chmod(0o644)

    summary = runner.grade_panel(
        panel, SANDBOX, candidate,
        worker_path=str(worker),
        python_exe=sys.executable,
        workers=WORKERS,
        episodes=sopcc.EPISODES_PER_INSTANCE,
    )
    reasons = collections.Counter()
    aborts = []
    for i, report in enumerate(summary["reports"]):
        for outcome in report["episodes"]:
            if outcome["failed"]:
                reasons[outcome["reason"].split(":")[0]] += 1
        if report["abort"]:
            aborts.append({
                "instance": i,
                "abort": report["abort"],
                "cpu_seconds": report["cpu_seconds"],
                "wall_seconds": report["wall_seconds"],


                "unexplained": (("closed the channel" in report["abort"]
                                 or "is not reading" in report["abort"])
                                and report["wall_seconds"] < 0.25 * WALL_SECONDS_PER_INSTANCE
                                and report["cpu_seconds"] < 0.5 * CPU_SECONDS_PER_INSTANCE),
            })

    payload.update({
        "metric": summary["metric"],
        "reward": grader_core.graded_reward(summary["metric"], split),
        "R": summary["R"],
        "F": summary["F"],
        "penalty": summary["penalty"],
        "episodes": summary["episodes"],
        "instances": summary["instances"],
        "per_draw_metric": {str(d): summary["per_draw"][d]["metric"]
                            for d in sorted(summary["per_draw"])},
        "per_draw_F": {str(d): summary["per_draw"][d]["F"]
                       for d in sorted(summary["per_draw"])},
        "delivered_std": summary["delivered_std"],
        "failure_reasons": dict(reasons),
        "aborted_instances": summary["aborted_instances"],
        "aborts": aborts,
        "unexplained_aborts": sum(1 for a in aborts if a["unexplained"]),
        "retried_instances": summary["retried_instances"],
        "max_cpu_seconds": summary["max_cpu_seconds"],
        "max_wall_seconds": summary["max_wall_seconds"],
        "valid": True,
        "reason": "ok",
    })
    _write_metric(payload)

    if _redacted(split):
        print(f"episodes: {payload['episodes']} over {payload['instances']} instances")
        print(f"failure rate F: {payload['F']:.4f}  penalty factor: {payload['penalty']:.4f}")
        print("failure rate per draw: " + ", ".join(
            f"{d}={v:.4f}" for d, v in payload["per_draw_F"].items()))
        print("failure reasons: " + (", ".join(
            f"{k}={v}" for k, v in sorted(reasons.items())) or "none"))
        print(f"largest per-instance cpu {payload['max_cpu_seconds']:.1f}s of "
              f"{CPU_SECONDS_PER_INSTANCE:.0f}s, wall {payload['max_wall_seconds']:.1f}s of "
              f"{WALL_SECONDS_PER_INSTANCE:.0f}s")
        print(f"instances that ended early: {payload['aborted_instances']}"
              + (" (" + "; ".join(a["abort"][:80] for a in aborts[:3]) + ")" if aborts else ""))
    else:
        print(json.dumps({k: v for k, v in payload.items() if k != "failure_reasons"},
                         indent=2, sort_keys=True))
        print("failure reasons: " + json.dumps(dict(reasons), sort_keys=True))
    return payload


def test_deliverable_runs(graded):
    """The submitted policy must exist, import, and answer decisions at all.

    WHY: everything else is meaningless if the deliverable never ran. A missing, oversized,
    link-bearing or unimportable deliverable is scored 0 rather than skipped.
    """
    assert graded["valid"], graded["reason"]
    assert graded["episodes"] == graded["instances"] * sopcc.EPISODES_PER_INSTANCE


def test_metric_is_well_formed(graded):
    """The metric and its parts must be finite and inside their definitions.

    WHY: the reward is a function of these numbers, so a NaN or an out-of-range failure rate
    has to fail loudly here instead of propagating into reward.txt.
    """
    assert 0.0 <= graded["F"] <= 1.0
    assert 0.0 <= graded["penalty"] <= 1.0
    assert graded["R"] >= 0.0
    assert graded["metric"] == pytest.approx(graded["R"] * graded["penalty"], abs=1e-9)
    assert 0.0 <= graded["reward"] < 1.0


def test_reward_follows_the_metric(graded):
    """The reward written for this run must be exactly the map applied to the measured metric.

    WHY: the reward is the only number that leaves the verifier, and it is computed once in the
    fixture. This re-derives it from the metric through the same map, so a mis-keyed split or a
    hand-edited payload cannot slip a different number into reward.txt. It also pins the floor:
    a metric at or below the trivial anchor is worth exactly 0.
    """
    split = graded["split"]
    assert graded["reward"] == pytest.approx(
        grader_core.graded_reward(graded["metric"], split), abs=1e-12)
    m0, m_ref = grader_core.ANCHORS[split]
    assert m_ref > m0
    assert grader_core.graded_reward(m0, split) == 0.0
    assert grader_core.graded_reward(m0 - 1.0, split) == 0.0
    assert grader_core.graded_reward(m_ref, split) == pytest.approx(0.5, abs=1e-12)


def test_no_infrastructure_failure(graded):
    """No instance may end early for a reason that is not the policy's own doing.

    WHY: an instance whose child process vanished contributes 32 failures and drags the metric
    down. Retries absorb a transient death; anything still unexplained afterwards is surfaced
    here, on a channel the candidate cannot reach or trigger, instead of being folded silently
    into a low score. A policy that crashes or blows its own budget is NOT unexplained.
    """
    assert graded.get("unexplained_aborts", 0) == 0, (
        f"{graded.get('unexplained_aborts')} instances died without spending their own "
        f"budget: {[a for a in graded.get('aborts', []) if a['unexplained']][:3]}")
