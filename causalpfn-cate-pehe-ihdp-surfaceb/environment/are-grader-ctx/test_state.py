from __future__ import annotations

import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grader_core as gc

TESTS_DIR = Path(__file__).resolve().parent
SPLIT = os.environ.get("GRADER_SPLIT", "final")
PANEL_PATH = TESTS_DIR / "hidden_data" / SPLIT / "panel.npz"
ARTIFACT_DIR = Path("/app/output")
LOG_DIR = Path("/logs/verifier")

PYTHON = "/usr/local/bin/python3"
RUNUSER = "/usr/sbin/runuser"
AGENT_USER = "agent"
AGENT_UID = 1001
AGENT_GID = 1001


def _write_metric(payload: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "metric.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _reap_agent_processes() -> None:
    """Kill anything still running as the candidate uid, without shelling out."""
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != AGENT_UID:
                continue
            os.kill(int(entry.name), signal.SIGKILL)
        except (OSError, ValueError):
            continue


def _run_candidate(solution_dir: Path, inputs_path: Path, work_dir: Path) -> dict:
    """Launch runner.py as uid 1001 and time it from this process, around the subprocess
    boundary. The candidate never reports its own duration and never runs in this process."""
    harness_dir = work_dir.parent / "harness"
    harness_dir.mkdir(parents=True, exist_ok=True)
    runner = harness_dir / "runner.py"
    runner.write_bytes((TESTS_DIR / "runner.py").read_bytes())
    os.chmod(harness_dir, 0o755)
    os.chmod(runner, 0o644)

    preds_path = work_dir / "predictions.npz"
    meta_path = work_dir / "meta.json"
    env_pairs = [
        "PATH=/usr/local/bin:/usr/bin:/bin",
        f"HOME={work_dir}",
        f"TMPDIR={work_dir}",
        "PYTHONSAFEPATH=1",
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONHASHSEED=0",
        "OMP_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "NUMEXPR_NUM_THREADS=1",
    ]
    cmd = [RUNUSER, "-u", AGENT_USER, "--", "/usr/bin/env", "-i", *env_pairs,
           PYTHON, str(runner), str(inputs_path), str(solution_dir),
           str(preds_path), str(meta_path)]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            cwd=str(work_dir), text=True, start_new_session=True)
    started = time.monotonic()
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=gc.GRADED_RUN_BUDGET_SEC)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except OSError:
            proc.kill()
        try:
            stdout, stderr = proc.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
    elapsed = time.monotonic() - started
    _reap_agent_processes()

    return {
        "timed_out": timed_out,
        "elapsed_sec": elapsed,
        "returncode": proc.returncode,
        "stderr_tail": (stderr or "")[-800:],
        "preds_path": preds_path,
        "meta_path": meta_path,
    }


def grade() -> dict:
    payload = {
        "split": SPLIT,
        "metric_name": "mean_pehe",
        "metric": None,
        "reward": 0.0,
        "valid": False,
        "reason": "",
        "n_realizations": 0,
        "n_ok": 0,
        "n_fallback": 0,
        "elapsed_sec": 0.0,
        "pruned_paths": [],
    }
    if not PANEL_PATH.is_file():
        payload["reason"] = f"sealed panel missing at {PANEL_PATH}"
        return payload
    if not ARTIFACT_DIR.is_dir():
        payload["reason"] = "no deliverable directory at /app/output"
        return payload

    inputs, taus = gc.load_sealed_panel(PANEL_PATH)
    payload["n_realizations"] = len(inputs)

    sandbox = Path(tempfile.mkdtemp(prefix="grade_", dir="/tmp"))
    os.chmod(sandbox, 0o755)
    solution_dir = sandbox / "solution"
    work_dir = sandbox / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    os.chown(work_dir, AGENT_UID, AGENT_GID)
    os.chmod(work_dir, 0o755)

    copy_report = gc.copy_solution_tree(ARTIFACT_DIR, solution_dir)
    payload["pruned_paths"] = copy_report["pruned"][:20]

    inputs_path = sandbox / "inputs.npz"
    gc.write_runner_inputs(inputs, inputs_path)

    run = _run_candidate(solution_dir, inputs_path, work_dir)
    payload["elapsed_sec"] = round(run["elapsed_sec"], 3)

    if run["timed_out"]:
        payload["reason"] = (f"graded run exceeded the {gc.GRADED_RUN_BUDGET_SEC:.0f} s "
                             "wall-clock budget and was killed")
        return payload

    meta = {}
    try:
        meta = json.loads(Path(run["meta_path"]).read_text())
    except Exception as exc:
        payload["reason"] = f"candidate wrote no usable run metadata: {exc}"
    if not meta.get("import_ok", False):
        payload["reason"] = payload["reason"] or (
            f"entry point could not be imported/called: {meta.get('error')}")
        return payload

    preds_path = Path(run["preds_path"])
    try:
        st = preds_path.lstat()
        if not stat.S_ISREG(st.st_mode) or st.st_size > 200_000_000:
            payload["reason"] = "predictions file is not a plain, plausibly-sized array file"
            return payload
    except OSError as exc:
        payload["reason"] = f"candidate wrote no predictions: {exc}"
        return payload

    try:
        with np.load(preds_path, allow_pickle=False) as f:
            preds = np.asarray(f["preds"], dtype=np.float64)
            status = np.asarray(f["status"], dtype=np.int64)
    except Exception as exc:
        payload["reason"] = f"predictions unreadable: {exc}"
        return payload

    if preds.shape != (len(inputs), gc.N_TEST) or status.shape != (len(inputs),):
        payload["reason"] = "prediction array has the wrong shape"
        return payload

    scored = gc.score_panel(inputs, taus, preds, status)
    if scored["n_ok"] == 0:
        payload["n_ok"] = 0
        payload["n_fallback"] = scored["n_fallback"]
        payload["reason"] = f"every call failed: {meta.get('error')}"
        return payload
    payload.update(scored)

    payload["valid"] = True
    payload["reward"] = gc.graded_reward(scored["metric"], True)
    payload["reason"] = "ok"
    return payload


def test_candidate_estimator_scores_the_sealed_panel():
    """Execute /app/output/estimator.py on the sealed panel for this split and record the
    mean PEHE and the mapped reward. Written on every path, including failure."""
    payload = grade()
    _write_metric(payload)
    metric = "n/a" if payload["metric"] is None else f"{payload['metric']:.4f}"
    print(f"split={payload['split']} realizations={payload['n_realizations']} "
          f"mean_PEHE={metric} calls_ok={payload['n_ok']} "
          f"calls_scored_as_constant={payload['n_fallback']} "
          f"run_seconds={payload['elapsed_sec']:.1f} status={payload['reason']}")
    assert payload["valid"], payload["reason"]
