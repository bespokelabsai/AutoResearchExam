from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import grader_core as gc

TESTS_DIR = Path(__file__).resolve().parent
DELIVERABLE = Path("/app/output")
ENTRY_POINT = DELIVERABLE / "selector.py"
LOG_DIR = Path("/logs/verifier")
METRIC_PATH = LOG_DIR / "metric.json"
RUNNER = TESTS_DIR / "run_selector.py"
PYTHON = sys.executable
AGENT_UID = 1001
AGENT_GID = 1001

SPLIT = os.environ.get("GRADED_SPLIT", "final")
assert SPLIT in ("intermediate", "final"), f"bad GRADED_SPLIT {SPLIT!r}"
HIDDEN = TESTS_DIR / "hidden_data"


class InvalidSubmission(Exception):
    """Anything that makes the deliverable ungradeable. Always scores exactly 0."""


def copy_submission(dest: Path) -> None:
    """Copy /app/output without ever following a link out of it.

    Every submitted path is hostile: the whole tree is walked first and any symlink, hard link or
    non-regular file is rejected before a single byte is copied, and the copy itself is made with
    symlinks=True so nothing is dereferenced even if the check were bypassed by a race.
    """
    if not DELIVERABLE.is_dir():
        raise InvalidSubmission(f"{DELIVERABLE} is not a directory")
    if DELIVERABLE.is_symlink():
        raise InvalidSubmission(f"{DELIVERABLE} is a symlink")
    for path in DELIVERABLE.rglob("*"):
        if path.is_symlink():
            raise InvalidSubmission(f"submission contains a symlink: {path}")
        st = path.lstat()
        if path.is_dir():
            continue
        if not path.is_file():
            raise InvalidSubmission(f"submission contains a non-regular file: {path}")
        if st.st_nlink > 1:
            raise InvalidSubmission(f"submission contains a hard link: {path}")
    shutil.copytree(DELIVERABLE, dest, symlinks=True)
    os.chmod(dest, 0o755)
    for path in dest.rglob("*"):
        os.chmod(path, 0o755 if path.is_dir() else 0o644)


_PRIV_PREFIX = None


def drop_privilege_prefix() -> list:
    """Command prefix that runs the child as the unprivileged agent uid.

    Prefers `runuser -u agent`, probed once so a PAM-less image falls back instead of failing
    every grade; the fallback drops the same privileges from Python in the child. Either way the
    child is put in its own session and process group (`start_new_session` / `os.setsid`) so the
    wall-clock timeout can kill the whole tree, and candidate code never runs as root.
    """
    global _PRIV_PREFIX
    if _PRIV_PREFIX is None:
        runuser = shutil.which("runuser") or (
            "/usr/sbin/runuser" if os.path.exists("/usr/sbin/runuser") else None)
        _PRIV_PREFIX = []
        if runuser:
            probe = subprocess.run([runuser, "-u", "agent", "--", PYTHON, "-c", "import os;"
                                    "assert os.getuid() == 1001"],
                                   capture_output=True, timeout=60)
            if probe.returncode == 0:
                _PRIV_PREFIX = [runuser, "-u", "agent", "--"]
    return list(_PRIV_PREFIX)


def _demote():
    os.setsid()
    os.setgid(AGENT_GID)
    os.setgroups([])
    os.setuid(AGENT_UID)


def run_candidate(cand_dir: Path, stage: Path, features: np.ndarray, labels: np.ndarray,
                  seed: int) -> np.ndarray:
    """Run ONE select_coreset call under the published wall-clock budget. Returns raw output."""
    runner_dir = stage / "runner"
    runner_dir.mkdir()
    runner = runner_dir / RUNNER.name
    shutil.copyfile(RUNNER, runner)
    os.chmod(runner_dir, 0o755)
    os.chmod(runner, 0o644)

    io_dir = stage / "io"
    io_dir.mkdir()
    feat_p, lab_p, out_p = io_dir / "features.npy", io_dir / "labels.npy", io_dir / "selection.npy"
    np.save(feat_p, features)
    np.save(lab_p, labels)
    for p in (feat_p, lab_p):
        os.chmod(p, 0o644)
    os.chmod(stage, 0o755)
    os.chmod(io_dir, 0o777)

    prefix = drop_privilege_prefix()
    cmd = prefix + [PYTHON, str(runner), str(cand_dir), str(feat_p), str(lab_p),
                    str(gc.BUDGET), str(seed), str(out_p)]
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": str(io_dir),
        "TMPDIR": str(io_dir),
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    started = time.monotonic()
    popen_kwargs = {"cwd": str(io_dir), "env": env, "stdout": subprocess.PIPE,
                    "stderr": subprocess.PIPE, "text": True}
    if prefix:
        popen_kwargs["start_new_session"] = True
    else:
        popen_kwargs["preexec_fn"] = _demote
    proc = subprocess.Popen(cmd, **popen_kwargs)
    try:
        _, err = proc.communicate(timeout=gc.SELECTOR_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            proc.kill()
        proc.communicate()
        raise InvalidSubmission(
            f"select_coreset exceeded its {gc.SELECTOR_TIMEOUT_SEC}s budget on seed {seed}")
    elapsed = time.monotonic() - started
    if proc.returncode != 0:
        raise InvalidSubmission(
            f"select_coreset call failed (exit {proc.returncode}) after {elapsed:.1f}s: "
            f"{(err or '').strip()[-400:]}")
    if out_p.is_symlink():
        raise InvalidSubmission("the selection path was replaced by a link")
    if not out_p.is_file():
        raise InvalidSubmission("the call produced no selection")
    try:
        raw = np.load(out_p, allow_pickle=False)
    except Exception as exc:
        raise InvalidSubmission(f"selection is unreadable: {type(exc).__name__}: {exc}")
    return raw


def load_sealed():
    pool_f = np.load(HIDDEN / "pool" / "pool_features.npy")
    pool_y = np.load(HIDDEN / "pool" / "pool_labels.npy")
    draws = np.load(HIDDEN / "pool" / "draws.npy")
    ev = (np.load(HIDDEN / SPLIT / "eval_features.npy"),
          np.load(HIDDEN / SPLIT / "eval_labels.npy"),
          np.load(HIDDEN / SPLIT / "eval_attributes.npy"))
    return pool_f, pool_y, draws, ev


def grade() -> dict:
    pool_f, pool_y, draws, (ev_f, ev_y, ev_a) = load_sealed()
    work = Path("/tmp/grade_work")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    per_draw = []
    all_runs = []
    for di in range(draws.shape[0]):
        idx = draws[di]
        f, y = pool_f[idx], pool_y[idx]
        seed = gc.SELECTOR_SEEDS[di]
        stage = work / f"draw{di}"
        stage.mkdir()
        cand_dir = stage / "candidate"
        copy_submission(cand_dir)
        raw = run_candidate(cand_dir, stage, f, y, seed)
        try:
            selection = gc.validate_selection(raw, f.shape[0], y)
        except gc.SelectionError as exc:
            raise InvalidSubmission(str(exc))
        runs = gc.score_selection(f, y, selection, ev_f, ev_y, ev_a)
        per_draw.append({"draw": di, "runs": runs,
                         "wga_mean": round(float(np.mean([r["worst_group"] for r in runs])), 4)})
        all_runs.extend(runs)
        shutil.rmtree(stage, ignore_errors=True)

    metric = round(float(np.mean([r["worst_group"] for r in all_runs])), 2)
    return {
        "metric_name": "worst_group_accuracy_percent",
        "split": SPLIT,
        "n_runs": len(all_runs),
        "metric": metric,
        "per_draw_wga": [d["wga_mean"] for d in per_draw],
        "overall_accuracy_mean": round(float(np.mean([r["overall"] for r in all_runs])), 2),
        "reward": gc.graded_reward(metric),
        "valid": True,
        "detail": per_draw,
    }


def write_metric(payload: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True))


def test_worst_group_accuracy_of_submitted_selector():
    """Execute /app/output/selector.py on the sealed pool draws and score the pinned probe.

    WHAT: runs the deliverable once per sealed pool draw, validates each returned index array
    against the published contract, trains the pinned probe on the selected rows for each pinned
    probe seed, and reports the mean worst-group accuracy over the sealed evaluation half.
    WHY: the reward must come from executing candidate code on inputs it has never seen, and the
    metric must be recomputed here from sealed attribute labels the candidate cannot read.
    """
    try:
        payload = grade()
    except InvalidSubmission as exc:
        write_metric({"metric_name": "worst_group_accuracy_percent", "split": SPLIT,
                      "metric": None, "reward": 0.0, "valid": False, "reason": str(exc)})
        print(f"[{SPLIT}] submission invalid: {exc}")
        raise AssertionError(f"invalid submission: {exc}")
    except Exception as exc:
        write_metric({"metric_name": "worst_group_accuracy_percent", "split": SPLIT,
                      "metric": None, "reward": 0.0, "valid": False,
                      "reason": f"grader error: {type(exc).__name__}: {exc}"})
        raise

    write_metric(payload)
    print(f"[{SPLIT}] worst-group accuracy over {payload['n_runs']} runs: {payload['metric']:.2f}")
    assert payload["metric"] is not None
