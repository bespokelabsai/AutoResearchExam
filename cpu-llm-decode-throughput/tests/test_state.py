from __future__ import annotations

import json
import os
import pwd
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import grader_core
import metric_core

TESTS_DIR = Path(__file__).resolve().parent
DELIVERABLE_DIR = Path("/app/output")
ENTRY_POINT = "engine.py"
MODEL_DIR = "/opt/model"
GRADE_ROOT = Path("/grade")
LOG_DIR = Path("/logs/verifier")
SPLIT = os.environ.get("GRADE_SPLIT", "final")
MAX_STAGED_BYTES = 4 * 1024 * 1024 * 1024

_RESULT = None




def _agent_ids():
    entry = pwd.getpwnam("agent")
    return entry.pw_uid, entry.pw_gid


def _reject_unsafe_paths(root: Path):
    """Refuse symlinks, hard links and special files anywhere under the deliverable.

    Checked at every depth BEFORE anything is copied, and the copy itself never
    follows a link, so a submitted path cannot materialise a verifier-owned file
    into the tree the score is computed from.
    """
    total = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in list(dirnames) + list(filenames):
            path = Path(dirpath) / name
            st = path.lstat()
            if stat.S_ISLNK(st.st_mode):
                return f"deliverable contains a symlink: {path}"
            if stat.S_ISDIR(st.st_mode):
                continue
            if not stat.S_ISREG(st.st_mode):
                return f"deliverable contains a special file: {path}"
            if st.st_nlink > 1:
                return f"deliverable contains a hard link: {path}"
            total += st.st_size
            if total > MAX_STAGED_BYTES:
                return "deliverable is larger than the staging budget"
    return ""


def _clear_grade_root():
    """Empty the staging tree without removing the directory itself."""
    GRADE_ROOT.mkdir(mode=0o755, parents=True, exist_ok=True)
    for child in GRADE_ROOT.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def _stage():
    """Build the root-owned read-only tree every timed run is launched from."""
    _clear_grade_root()

    if not DELIVERABLE_DIR.is_dir():
        return None, f"no deliverable directory at {DELIVERABLE_DIR}"
    problem = _reject_unsafe_paths(DELIVERABLE_DIR)
    if problem:
        return None, problem
    entry = DELIVERABLE_DIR / ENTRY_POINT
    if not entry.is_file() or entry.stat().st_size == 0:
        return None, f"no entry point at {DELIVERABLE_DIR / ENTRY_POINT}"

    candidate = GRADE_ROOT / "candidate"
    shutil.copytree(DELIVERABLE_DIR, candidate, symlinks=True)
    reference = GRADE_ROOT / "reference"
    reference.mkdir()
    shutil.copy2(TESTS_DIR / "reference_engine.py", reference / "reference_engine.py")
    harness = GRADE_ROOT / "harness"
    harness.mkdir()
    shutil.copy2(TESTS_DIR / "runner.py", harness / "runner.py")
    shutil.copy2(TESTS_DIR / "warmup_workload.json", GRADE_ROOT / "warmup_workload.json")

    subprocess.run(
        ["chown", "-R", "root:root", str(GRADE_ROOT)], check=True, timeout=120
    )
    subprocess.run(["chmod", "-R", "a=rX", str(GRADE_ROOT)], check=True, timeout=120)
    subprocess.run(["chmod", "755", str(GRADE_ROOT)], check=True, timeout=60)

    work = GRADE_ROOT / "work"
    work.mkdir(mode=0o755)
    return {
        "candidate_engine": str(candidate / ENTRY_POINT),
        "reference_engine": str(reference / "reference_engine.py"),
        "runner": str(harness / "runner.py"),
        "warmup": str(GRADE_ROOT / "warmup_workload.json"),
        "work": str(work),
    }, ""


def _workloads():
    directory = TESTS_DIR / "hidden_data" / SPLIT
    out = []
    for path in sorted(directory.glob("workload_*.json")):
        with open(path) as fh:
            out.append((path.stem, json.load(fh)))
    assert out, f"no sealed workloads under {directory}"
    return out




def _write_metric(payload):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_DIR / "metric.json", "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def _run_grade():
    started = time.monotonic()
    staged, problem = _stage()
    if staged is None:
        return {
            "split": SPLIT,
            "valid": False,
            "detail": problem,
            "metric_name": "speedup",
            "metric": 0.0,
            "speedup": 0.0,
            "agreement": 0.0,
            "reward": 0.0,
            "rounds": [],
        }

    uid, gid = _agent_ids()
    runuser = shutil.which("runuser", path="/usr/sbin:/sbin:/usr/bin:/bin")
    assert runuser, "runuser is missing from the verifier image"
    prefix = [runuser, "-u", "agent", "--"]
    round_log = (lambda *a, **k: None) if SPLIT == "intermediate" else print
    rounds = metric_core.run_rounds(
        staged["reference_engine"],
        staged["candidate_engine"],
        MODEL_DIR,
        staged["warmup"],
        _workloads(),
        staged["work"],
        cmd_prefix=prefix,
        runner_path=staged["runner"],
        run_dir_owner=(uid, gid),
        reap_uid=uid,
        log=round_log,
    )
    summary = metric_core.summarise(rounds)
    reward = grader_core.graded_reward(summary["speedup"], summary["valid"])
    return {
        "split": SPLIT,
        "valid": bool(summary["valid"]),
        "detail": summary["detail"],
        "metric_name": "speedup",
        "metric": float(summary["speedup"]),
        "speedup": float(summary["speedup"]),
        "agreement": float(summary["agreement"]),
        "agreement_minimum": metric_core.AGREEMENT_MIN,
        "per_round_speedup": summary.get("per_round_speedup", []),
        "reward": float(reward),
        "rounds": rounds,
        "grading_seconds": round(time.monotonic() - started, 2),
    }


def _grade():
    global _RESULT
    if _RESULT is None:
        try:
            _RESULT = _run_grade()
        except Exception as exc:
            _RESULT = {
                "split": SPLIT,
                "valid": False,
                "detail": f"grader error: {type(exc).__name__}: {exc}",
                "metric_name": "speedup",
                "metric": 0.0,
                "speedup": 0.0,
                "agreement": 0.0,
                "reward": 0.0,
                "rounds": [],
            }
        _write_metric(_RESULT)
        _clear_grade_root()
    return _RESULT




def test_deliverable_is_present_and_safe():
    """The submitted engine exists, is a plain file tree, and loads and runs at all."""
    result = _grade()
    assert result["rounds"], f"the engine never produced a timed run: {result['detail']}"


def test_outputs_match_the_reference():
    """Returned token ids reproduce the reference engine's, position by position."""
    result = _grade()
    assert result["agreement"] >= metric_core.AGREEMENT_MIN, (
        f"token agreement {result['agreement']:.4f} < {metric_core.AGREEMENT_MIN}: "
        f"{result['detail']}"
    )


def test_speedup_is_measured_and_recorded():
    """A speedup over the reference was measured on every sealed workload."""
    result = _grade()
    assert (LOG_DIR / "metric.json").is_file(), "metric.json was not written"
    assert result["speedup"] > 0.0, f"no speedup was measured: {result['detail']}"
    print(
        f"split={result['split']} speedup_S={result['speedup']:.4f} "
        f"token_agreement={result['agreement']:.4f} "
        f"per_round={[round(s, 3) for s in result.get('per_round_speedup', [])]}"
    )
