from __future__ import annotations

import json
import os
import pwd
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import grader_core as gc

PY = "/usr/local/bin/python3"
AGENT_UID = 1001
try:
    AGENT_GID = pwd.getpwuid(AGENT_UID).pw_gid
except KeyError:
    AGENT_GID = 1001
GRADE_ROOT = Path("/opt/grade")
METRIC_PATH = Path("/logs/verifier/metric.json")

CELL_BUDGET_SEC = gc.CELL_BUDGET_SEC

SCRUB_DIRS = ("/tmp", "/var/tmp", "/dev/shm", "/home/agent")

CHILD_ENV = {
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "LC_ALL": "C",
    "PYTHONHASHSEED": "0",
    "PYTHONSAFEPATH": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}




def _tool(name: str, fallback: str) -> str:
    return shutil.which(name) or fallback


RUNUSER = _tool("runuser", "/usr/sbin/runuser")


def _chown_tree(path: Path, uid: int, gid: int) -> None:
    os.chown(path, uid, gid)
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames.sort()
        for name in sorted(dirnames + filenames):
            os.chown(os.path.join(dirpath, name), uid, gid)


def _drop_privilege_works() -> bool:
    """Probe the privilege-drop mechanism before grading anything.

    The agent cannot reach or influence this probe; its result is recorded in
    metric.json so an infra failure is distinguishable from a bad submission.
    """
    if os.geteuid() != 0:
        return False
    try:
        probe = subprocess.run(
            [RUNUSER, "-u", "agent", "--", PY, "-c", "import os;print(os.getuid())"],
            capture_output=True,
            text=True,
            timeout=120,
            env=dict(CHILD_ENV),
            start_new_session=True,
        )
    except Exception:
        return False
    return probe.returncode == 0 and probe.stdout.strip() == str(AGENT_UID)


def _launch(argv, cwd, env, budget, log_dir: Path, drop_privilege: bool):
    """Run argv in a session of its own; SIGKILL the whole group at `budget` seconds.

    start_new_session=True is setsid(2): the child leads a new session and process group,
    so os.killpg reaches anything it forked. Output goes to files rather than pipes so
    that a grandchild holding a pipe open cannot make the grader itself hang.

    `drop_privilege` uses Popen's own setuid/setgid, which is the fallback when runuser is
    unusable. Candidate code is never executed as root on any path.
    """
    out_path = log_dir / "stdout.txt"
    err_path = log_dir / "stderr.txt"
    extra = {"user": AGENT_UID, "group": AGENT_GID} if drop_privilege else {}
    started = time.monotonic()
    with open(out_path, "w") as out_fh, open(err_path, "w") as err_fh:
        proc = subprocess.Popen(
            argv,
            cwd=str(cwd),
            env=env,
            stdout=out_fh,
            stderr=err_fh,
            start_new_session=True,
            **extra,
        )
        try:
            proc.wait(timeout=budget)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                proc.kill()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                pass
    elapsed = time.monotonic() - started
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        pass
    try:
        stderr_tail = err_path.read_text(errors="replace")[-800:]
    except Exception:
        stderr_tail = ""
    return {
        "returncode": proc.returncode,
        "stderr": stderr_tail,
        "timed_out": timed_out,
        "elapsed": elapsed,
    }


def _stage_deliverable() -> dict:
    """Copy /app/output into a root-owned tree, pruning symlinks at every depth."""
    staged = GRADE_ROOT / "deliverable"
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir(parents=True)

    info = {"present": False, "entry_bytes": 0, "files": 0, "pruned_symlinks": 0}
    src = gc.DELIVERABLE_DIR
    if not src.is_dir():
        return info

    files, links = gc.deliverable_files(src)
    info["pruned_symlinks"] = len(links)
    entry_src = str(src / gc.ENTRY_POINT.name)
    files.sort(key=lambda p: (p != entry_src, p))
    if len(files) > gc.MAX_DELIVERABLE_FILES:
        files = files[: gc.MAX_DELIVERABLE_FILES]

    total = 0
    for full in files:
        rel = os.path.relpath(full, src)
        size = os.path.getsize(full)
        if total + size > gc.MAX_DELIVERABLE_BYTES:
            break
        dest = staged / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(full, dest, follow_symlinks=False)
        os.chmod(dest, 0o644)
        total += size
        info["files"] += 1

    entry = staged / gc.ENTRY_POINT.name
    if entry.is_file():
        info["entry_bytes"] = entry.stat().st_size
        info["present"] = info["entry_bytes"] > 0
    _chown_tree(staged, 0, 0)
    os.chmod(staged, 0o755)
    return info


def _snapshot_writable() -> dict:
    """Record what is in the shared writable directories before any candidate runs."""
    seen = {}
    for d in SCRUB_DIRS:
        try:
            seen[d] = set(sorted(os.listdir(d)))
        except OSError:
            seen[d] = set()
    return seen


def _scrub_writable(seen: dict) -> None:
    """Remove everything a cell left behind in the shared writable directories."""
    for d, before in seen.items():
        try:
            entries = sorted(os.listdir(d))
        except OSError:
            continue
        for name in entries:
            if name in before:
                continue
            full = os.path.join(d, name)
            try:
                if os.path.islink(full) or os.path.isfile(full):
                    os.unlink(full)
                else:
                    shutil.rmtree(full, ignore_errors=True)
            except OSError:
                pass


def _prepare_runner() -> Path:
    runner_dir = GRADE_ROOT / "runner"
    runner_dir.mkdir(parents=True, exist_ok=True)
    dest = runner_dir / "cellrunner.py"
    shutil.copyfile("/tests/cellrunner.py", dest)
    os.chmod(runner_dir, 0o755)
    os.chmod(dest, 0o644)
    return dest


def _run_cell(runner: Path, staged: Path, index: int, X_train, X_eval, seed):
    cell_dir = GRADE_ROOT / "cells" / f"c{index:03d}"
    if cell_dir.exists():
        shutil.rmtree(cell_dir)
    work = cell_dir / "work"
    work.mkdir(parents=True)
    os.chmod(cell_dir, 0o755)

    in_path = cell_dir / "in.npz"
    np.savez(in_path, X_train=X_train, X_eval=X_eval, seed=np.array([seed], dtype=np.int64))
    os.chmod(in_path, 0o644)

    out_path = work / "out.npy"
    os.chown(work, AGENT_UID, AGENT_GID)
    os.chmod(work, 0o700)

    env = dict(CHILD_ENV)
    env["TMPDIR"] = str(work)
    env["HOME"] = str(work)
    call = [PY, str(runner), str(in_path), str(out_path), str(staged)]
    argv = [RUNUSER, "-u", "agent", "--"] + call if RUNUSER_OK else call

    res = _launch(argv, work, env, CELL_BUDGET_SEC, cell_dir, drop_privilege=not RUNUSER_OK)

    returned = None
    if not res["timed_out"] and res["returncode"] == 0 and out_path.is_file():
        try:
            returned = np.load(out_path, allow_pickle=False)
        except Exception:
            returned = None
    shutil.rmtree(cell_dir, ignore_errors=True)
    return returned, res




RUNUSER_OK = False


def _grade() -> dict:
    global RUNUSER_OK
    half = os.environ.get("GRADE_SPLIT", "final")
    record = {
        "half": half,
        "cell_budget_sec": CELL_BUDGET_SEC,
        "privilege_drop": "unknown",
        "cells_total": 0,
        "cells_valid": 0,
        "cells_invalid": 0,
        "invalid_reasons": {},
        "metric": 0.0,
        "reward": 0.0,
        "valid": False,
        "note": "",
    }

    GRADE_ROOT.mkdir(parents=True, exist_ok=True)
    os.chmod(GRADE_ROOT, 0o755)

    RUNUSER_OK = _drop_privilege_works()
    record["privilege_drop"] = "runuser:agent" if RUNUSER_OK else "popen-setuid:agent"

    staged_info = _stage_deliverable()
    record["deliverable"] = staged_info

    cells = list(gc.iter_cells(half))
    record["cells_total"] = len(cells)

    if not staged_info["present"]:
        record["note"] = "missing_or_empty_entry_point"
        record["cells_invalid"] = len(cells)
        record["invalid_reasons"] = {"missing_entry_point": len(cells)}
        return record

    runner = _prepare_runner()
    staged = GRADE_ROOT / "deliverable"

    writable_before = _snapshot_writable()
    scores, reasons, elapsed, per_cell = [], {}, [], {}
    for index, (key, X, seed, n_train, n_eval) in enumerate(cells):
        X_train, X_eval, X_eval_true, mask_eval = gc.cellspec.build_cell(X, seed, n_train, n_eval)
        returned, res = _run_cell(runner, staged, index, X_train, X_eval, seed)
        _scrub_writable(writable_before)
        elapsed.append(round(res["elapsed"], 3))

        if res["timed_out"]:
            score, reason = None, "timeout"
        elif res["returncode"] != 0:
            score, reason = None, "nonzero_exit"
        elif returned is None:
            score, reason = None, "no_or_unreadable_output"
        else:
            score, reason = gc.score_submission(returned, X_eval_true, mask_eval)

        scores.append(score)
        per_cell[key] = None if score is None else round(float(score), 6)
        if score is None:
            reasons[reason] = reasons.get(reason, 0) + 1
        del X_train, X_eval, X_eval_true, mask_eval, returned

    record["cells_valid"] = sum(1 for s in scores if s is not None)
    record["cells_invalid"] = sum(1 for s in scores if s is None)
    record["invalid_reasons"] = reasons
    record["metric"] = gc.aggregate(scores)
    record["reward"] = gc.graded_reward(record["metric"])
    record["valid"] = record["cells_valid"] > 0
    record["cell_elapsed_sec"] = elapsed
    record["cell_scores"] = per_cell
    if not record["valid"]:
        record["reward"] = 0.0
    return record


def _write(record: dict) -> None:
    METRIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRIC_PATH, "w") as fh:
        json.dump(record, fh, indent=2, sort_keys=True)
        fh.write("\n")


@pytest.fixture(scope="module")
def result():
    try:
        record = _grade()
    except Exception as exc:
        record = {"valid": False, "reward": 0.0, "metric": 0.0, "note": f"grader_error: {exc!r}"}
        _write(record)
        raise
    _write(record)
    return record


def test_deliverable_present(result):
    """The submitted entry point exists at the declared path and is non-empty."""
    assert result["deliverable"]["present"], (
        f"no usable {gc.ENTRY_POINT}: {result.get('note')}"
    )


def test_no_symlinks_submitted(result):
    """Symbolic links under the deliverable directory are pruned, never followed."""
    assert result["deliverable"]["pruned_symlinks"] == 0, (
        "symbolic links under /app/output were skipped; they are not copied to the verifier"
    )


def test_every_cell_ran(result):
    """Every graded cell returned a finite, correctly shaped array inside its budget."""
    assert result["cells_total"] > 0
    assert result["cells_invalid"] == 0, (
        f"{result['cells_invalid']}/{result['cells_total']} cells scored 0: "
        f"{result['invalid_reasons']}"
    )


def test_metric_recorded(result):
    """The panel metric and the mapped reward were computed and recorded."""
    assert METRIC_PATH.is_file()
    assert np.isfinite(result["metric"])
    assert 0.0 <= result["reward"] <= 1.0
