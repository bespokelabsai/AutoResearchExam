from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import grader_core

TESTS = Path(__file__).resolve().parent
SUBMISSION = Path("/app/submission")
ENTRY_POINT = "policy.py"
HOST_PKG_ROOT = "/opt"
PANEL = os.environ.get("GRADE_SPLIT", "final")
PANEL_DIR = TESTS / "hidden_data" / PANEL
METRIC_PATH = Path("/logs/verifier/metric.json")

PYTHON = "/usr/local/bin/python3"
AGENT_UID = 1001
SBIN = "/usr/sbin:/sbin:/usr/bin:/bin:/usr/local/bin"


def _launch_prefix():
    """How the policy process is dropped to uid 1001.

    Measured during the build: wrapping the launch in `setsid runuser -u agent`
    inserts a PAM fork, so the pid the solver process holds is the wrapper and
    the policy ends up in a different session. `killpg` on a hung policy then
    misses it and the run leaks a process that competes with the rest of the
    panel for the 8 CPUs. So the solver process drops privilege itself
    (`drop_to_uid`, setgroups/setgid/setuid before exec) with
    `start_new_session=True`, i.e. setsid(2) in the child: the policy process is
    the session leader, uid 1001, and the whole session is killable by pid.

    An explicit prefix is still supported for a host without a root verifier;
    `runuser` is resolved here so that path stays available.
    """
    if os.geteuid() == 0:
        return []
    runuser = shutil.which("runuser", path=SBIN)
    setsid = shutil.which("setsid", path=SBIN)
    if runuser and setsid:
        return [setsid, runuser, "-u", "agent", "--"]
    return []


LAUNCH_PREFIX = _launch_prefix()

WORKERS = 8
SOLVE_TIMEOUT = 120.0
STARTUP_TIMEOUT = 30.0
PANEL_TIMEOUT = 1900.0
RUN_HARD_TIMEOUT = SOLVE_TIMEOUT + STARTUP_TIMEOUT + 45.0

_RESULT = None


def _validate_submission():
    """Fail-closed contract checks on the deliverable directory."""
    if not SUBMISSION.is_dir():
        return False, f"{SUBMISSION} is not a directory"
    entry = SUBMISSION / ENTRY_POINT
    if not entry.is_file():
        return False, f"no {ENTRY_POINT} in {SUBMISSION}"
    if entry.stat().st_size == 0:
        return False, f"{entry} is empty"
    total = 0
    root = SUBMISSION.resolve()
    for dirpath, dirnames, filenames in os.walk(SUBMISSION, followlinks=False):
        for name in list(dirnames) + list(filenames):
            path = Path(dirpath) / name
            if path.is_symlink():
                target = Path(os.path.realpath(path))
                if root not in target.parents and target != root:
                    return False, f"symlink escapes the submission: {path}"
            else:
                try:
                    total += path.stat().st_size
                except OSError:
                    pass
    if total > grader_core.MAX_SUBMISSION_BYTES:
        return False, (f"submission is {total} bytes, over "
                       f"{grader_core.MAX_SUBMISSION_BYTES}")
    return True, None


def _run_one(job):
    """One graded run, in its own solver subprocess owned by the verifier."""
    seed, shift = job
    cmd = [
        PYTHON, str(TESTS / "solve_run.py"),
        "--seed", str(seed), "--shift", str(shift),
        "--submission", str(SUBMISSION),
        "--host-pkg-root", HOST_PKG_ROOT,
        "--python", PYTHON,
        "--time-limit", str(SOLVE_TIMEOUT),
        "--startup-timeout", str(STARTUP_TIMEOUT),
        "--launch-prefix", ",".join(LAUNCH_PREFIX),
        "--drop-to-uid", str(AGENT_UID),
    ]
    started = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=RUN_HARD_TIMEOUT)
        record = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:
        record = {
            "seed": int(seed), "shift": int(shift),
            "nodes": grader_core.NODE_CAP, "raw_nodes": None,
            "status": "driver_failed", "proved_optimal": False,
            "decisions": 0, "solve_seconds": 0.0,
            "error": f"{type(exc).__name__}: {exc}"[:200],
            "extra_processes": 0,
        }
    record["wall"] = round(time.monotonic() - started, 2)
    return record


def _grade():
    """Grade once per pytest session and cache the outcome."""
    global _RESULT
    if _RESULT is not None:
        return _RESULT

    METRIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "panel": PANEL,
        "valid": False,
        "metric": None,
        "reward": 0.0,
        "invalid_reason": None,
        "runs": [],
        "launcher": LAUNCH_PREFIX or f"in-process setuid({AGENT_UID})",
    }
    try:
        seeds, shifts = grader_core.load_panel(PANEL_DIR)
        ok, reason = _validate_submission()
        if not ok:
            payload["invalid_reason"] = reason
        else:
            subprocess.run(["chmod", "-R", "go+rX", str(SUBMISSION)],
                           check=False, timeout=120)
            jobs = [(s, k) for s in seeds for k in shifts]
            deadline = time.monotonic() + PANEL_TIMEOUT
            records = {}
            with ProcessPoolExecutor(max_workers=WORKERS) as pool:
                futures = {pool.submit(_run_one, job): job for job in jobs}
                for fut in as_completed(futures):
                    rec = fut.result()
                    records[(rec["seed"], rec["shift"])] = rec
                    if time.monotonic() > deadline:
                        for pending in futures:
                            pending.cancel()
                        break
            for job in jobs:
                if job not in records:
                    records[job] = {
                        "seed": job[0], "shift": job[1],
                        "nodes": grader_core.NODE_CAP, "raw_nodes": None,
                        "status": "panel_budget_exhausted",
                        "proved_optimal": False, "decisions": 0,
                        "solve_seconds": 0.0, "error": None,
                        "extra_processes": 0, "wall": 0.0,
                    }
            runs = [records[job] for job in jobs]
            nodes = [r["nodes"] for r in runs]
            metric = grader_core.shifted_geomean(nodes)
            payload.update({
                "valid": True,
                "metric": round(metric, 4),
                "reward": grader_core.reward_from_metric(
                    metric, m0=grader_core.m0_for(PANEL)),
                "runs": runs,
                "n_runs": len(runs),
                "n_proved_optimal": sum(1 for r in runs if r["proved_optimal"]),
                "n_charged_cap": sum(1 for r in runs
                                     if not r["proved_optimal"]),
                "node_min": min(nodes),
                "node_max": max(nodes),
                "m0": grader_core.m0_for(PANEL),
                "baseline_score": grader_core.BASELINE_SCORE,
            })
    except Exception as exc:
        payload["invalid_reason"] = f"grader error: {type(exc).__name__}: {exc}"
        payload["valid"] = False
        payload["reward"] = 0.0
    METRIC_PATH.write_text(json.dumps(payload, indent=1))
    _RESULT = payload
    return payload


def test_deliverable_contract():
    """The submission directory satisfies the contract stated in the task.

    An entry point that is missing, empty, oversized, or reaching outside the
    submission directory through a symlink is scored 0 and nothing is executed.
    """
    result = _grade()
    assert result["invalid_reason"] is None, result["invalid_reason"]


def test_graded_panel():
    """Every run of the sealed panel executed and the metric was aggregated.

    The node count of a run is the solver's own count, taken in a process that
    never loaded submission code; a run that fails to prove optimality inside
    the caps is charged the node cap.
    """
    result = _grade()
    assert result["valid"], result["invalid_reason"]
    assert result["n_runs"] == len(result["runs"])
    assert result["metric"] is not None and result["metric"] > 0
    print(f"panel={result['panel']} metric={result['metric']} "
          f"charged_cap={result['n_charged_cap']}/{result['n_runs']}")


def test_reward_is_bounded():
    """The reward derived from the metric is finite and inside [0, 1]."""
    result = _grade()
    assert 0.0 <= float(result["reward"]) <= 1.0
