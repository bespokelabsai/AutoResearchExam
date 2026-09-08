import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grader_core as gc

TESTS_DIR = Path(__file__).resolve().parent
SUBMISSION_DIR = Path("/app/solution")
ENTRY_POINT = "impute.py"
METRIC_PATH = Path("/logs/verifier/metric.json")
SPLIT = os.environ.get("GRADER_SPLIT", "final")
HIDDEN_DIR = TESTS_DIR / "hidden_data" / SPLIT
RUNUSER = shutil.which("runuser") or "/usr/sbin/runuser"
AGENT_UID, AGENT_GID = 1001, 1001
MAX_CONCURRENT = 4
PYTHON = sys.executable


def _copy_submission(dest):
    """Copy the submitted directory, treating every submitted path as hostile.

    Any symlink or multiply-linked regular file anywhere in the tree rejects the whole
    submission: a privileged copy that followed `impute.py -> /tests/hidden_data/...`
    would materialise verifier-owned files into the directory the score is computed from.
    """
    if not SUBMISSION_DIR.is_dir() or SUBMISSION_DIR.is_symlink():
        return False, f"{SUBMISSION_DIR} is missing or is a symlink"
    for path in [SUBMISSION_DIR, *SUBMISSION_DIR.rglob("*")]:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            return False, f"submitted tree contains a symlink: {path}"
        if stat.S_ISREG(info.st_mode) and info.st_nlink > 1:
            return False, f"submitted tree contains a hard link: {path}"
    shutil.copytree(SUBMISSION_DIR, dest, symlinks=True, dirs_exist_ok=True)
    subprocess.run(["chown", "-R", "root:root", str(dest)], check=False)
    subprocess.run(["chmod", "-R", "a-w,a+rX", str(dest)], check=False)
    if not (Path(dest) / ENTRY_POINT).is_file():
        return False, f"no {ENTRY_POINT} in the submitted directory"
    return True, ""


def _run_one(cand_dir, npz_path):
    """Execute the candidate on one sealed panel and score it here, in the root process.

    Returns (metric, note).  Every failure mode -- missing entry point, import error,
    exception, timeout, no output, unparseable output, wrong shape, wrong dtype,
    non-finite value, out-of-range value -- returns the trivial-baseline metric.
    """
    with np.load(npz_path, allow_pickle=False) as data:
        panel = np.ascontiguousarray(data["panel"], dtype=np.float64)
        eval_mask = data["eval_mask"]
        eval_true = data["eval_true"]
        seed = int(data["seed"])

    work = Path(tempfile.mkdtemp(prefix="grade_"))
    try:
        os.chmod(work, 0o755)
        runner = work / "run_candidate.py"
        shutil.copy(TESTS_DIR / "run_candidate.py", runner)
        panel_path = work / "panel.npy"
        np.save(panel_path, panel, allow_pickle=False)
        os.chmod(panel_path, 0o444)
        os.chmod(runner, 0o444)

        scratch = work / "scratch"
        scratch.mkdir()
        os.chown(scratch, AGENT_UID, AGENT_GID)
        os.chmod(scratch, 0o755)
        out_path = scratch / "out.npy"

        env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": str(scratch),
            "TMPDIR": str(scratch),
            "PYTHONHASHSEED": "0",
            "PYTHONSAFEPATH": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
        cmd = [RUNUSER, "-u", "agent", "--", PYTHON, str(runner),
               str(cand_dir), str(panel_path), str(seed), str(out_path)]
        chatter_path = work / "candidate_stdio.log"
        with open(chatter_path, "wb") as sink:
            proc = subprocess.Popen(cmd, env=env, cwd=str(scratch),
                                    stdout=sink, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
            try:
                proc.wait(timeout=gc.PER_INSTANCE_TIMEOUT_SEC)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except OSError:
                    proc.kill()
                proc.wait()
                return gc.M0, "timed out"
        if proc.returncode != 0:
            tail = chatter_path.read_bytes()[-4000:].decode("utf-8", "replace")
            lines = tail.strip().splitlines()
            return gc.M0, f"candidate exited {proc.returncode}: {lines[-1][:200] if lines else ''}"

        if not out_path.exists():
            return gc.M0, "no output file"
        info = out_path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            return gc.M0, "output path is not a regular file"
        if info.st_size > gc.MAX_OUTPUT_BYTES:
            return gc.M0, "output file too large"
        try:
            fd = os.open(out_path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, "rb") as handle:
                arr = np.load(handle, allow_pickle=False)
        except Exception as exc:
            return gc.M0, f"unreadable output: {type(exc).__name__}"

        ok, reason = gc.validate_output(arr, panel.shape)
        if not ok:
            return gc.M0, reason
        return gc.r2_imp(arr.astype(np.float64, copy=False)[eval_mask], eval_true), "ok"
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _reap_agent_processes():
    """Kill anything still running as uid 1001.

    A candidate can detach itself from the process group the per-panel timeout kills.  It
    could not forge a score even so -- /logs/verifier is root-owned 0755 -- but nothing
    agent-authored should outlive the graded run.  Done by walking /proc rather than with
    pkill, which the slim base does not ship.
    """
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid == AGENT_UID:
                os.kill(int(entry.name), signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass


def test_imputation_r2():
    """Grade the submitted imputer on the sealed panels of this split.

    WHAT: executes /app/solution/impute.py once per sealed panel and recomputes
    R^2_imp on the held-out cells from verifier-owned ground truth.
    WHY: the score must come from running the agent's code on panels it has never seen,
    never from a file it wrote or a number it reported.
    """
    instances = sorted(HIDDEN_DIR.glob("instance_*.npz"))
    assert instances, f"sealed split {HIDDEN_DIR} is empty"

    cand_root = Path(tempfile.mkdtemp(prefix="cand_"))
    os.chmod(cand_root, 0o755)
    cand_dir = cand_root / "solution"
    ok, reason = _copy_submission(cand_dir)
    if ok:
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
            results = list(pool.map(lambda p: _run_one(cand_dir, p), instances))
    else:
        results = [(gc.M0, reason)] * len(instances)
    shutil.rmtree(cand_root, ignore_errors=True)
    _reap_agent_processes()

    metrics = [m for m, _ in results]
    notes = [n for _, n in results]
    mean_metric = float(np.mean(metrics))
    reward = gc.graded_reward(mean_metric)

    METRIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRIC_PATH.write_text(json.dumps({
        "metric_name": "mean_out_of_sample_R2_imp_block_regime",
        "split": SPLIT,
        "n_instances": len(instances),
        "n_scored_ok": sum(1 for n in notes if n == "ok"),
        "metric": mean_metric,
        "reward": reward,
        "submission_ok": ok,
        "submission_note": reason if not ok else "ok",
    }, indent=2) + "\n")

    if SPLIT == "final":
        for path, (m, note) in zip(instances, results):
            print(f"{path.name}: R2_imp={m:+.4f} ({note})")
    print(f"split={SPLIT} instances={len(instances)} "
          f"scored_ok={sum(1 for n in notes if n == 'ok')} "
          f"mean_R2_imp={mean_metric:+.4f} reward={reward:.4f}")
