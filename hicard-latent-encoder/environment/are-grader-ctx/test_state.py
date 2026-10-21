from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grader_core as gc

SPLIT = os.environ.get("GRADER_SPLIT", "final")
HIDDEN_DIR = Path("/tests/hidden_data") / SPLIT
DELIVERABLE_DIR = Path("/app/output")
LOGS_DIR = Path("/logs/verifier")
METRIC_PATH = LOGS_DIR / "metric.json"
RUNNER = Path("/opt/harness/candidate_runner.py")
PYTHON = "/usr/local/bin/python3"
RUNUSER = "/usr/sbin/runuser"
AGENT_USER = "agent"
AGENT_UID = 1001
AGENT_GID = 1001
SCRATCH = Path("/tmp/grading") / SPLIT

MAX_OUTPUT_BYTES = 64 * 1024 * 1024
CANDIDATE_AS_LIMIT = 8 * 1024**3


def _write_metric(payload: dict) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _chown_tree(root: Path) -> None:
    os.chown(root, AGENT_UID, AGENT_GID)
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            p = Path(dirpath) / name
            try:
                os.chown(p, AGENT_UID, AGENT_GID, follow_symlinks=False)
            except OSError:
                pass


def _drop_privilege() -> None:
    """Fallback only, for an image without util-linux's runuser."""
    os.setgroups([AGENT_GID])
    os.setgid(AGENT_GID)
    os.setuid(AGENT_UID)


def _command(cand_dir: Path, inputs: Path, out: Path) -> list[str]:
    """`sh -c` sets the address-space and core limits in the child without running any
    Python after fork, then execs the runner in place."""
    inner = " ".join(shlex.quote(str(a)) for a in
                     [PYTHON, "-I", RUNNER, cand_dir, inputs, out])
    script = f"ulimit -v {CANDIDATE_AS_LIMIT // 1024}; ulimit -c 0; exec {inner}"
    if os.path.exists(RUNUSER):
        return [RUNUSER, "-u", AGENT_USER, "--", "/bin/sh", "-c", script]
    return ["/bin/sh", "-c", script]


def _run_candidate(cand_dir: Path, inputs: Path, out: Path) -> tuple[bool, str, dict]:
    """Run the deliverable on one instance. Returns (ok, reason, diagnostics)."""
    if out.exists():
        out.unlink()
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": "/home/agent",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
        "TMPDIR": str(cand_dir.parent / "tmp"),
    }
    started = time.monotonic()
    proc = subprocess.Popen(
        _command(cand_dir, inputs, out),
        cwd="/opt/harness",
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        start_new_session=True,
        preexec_fn=None if os.path.exists(RUNUSER) else _drop_privilege,
    )
    timed_out = False
    try:
        output, _ = proc.communicate(timeout=gc.CANDIDATE_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        timed_out = True
        output = ""
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            output, _ = proc.communicate(timeout=30)
        except Exception:
            proc.kill()
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
    elapsed = time.monotonic() - started
    diag = {
        "wall_sec": round(elapsed, 2),
        "returncode": proc.returncode,
        "stdio_tail": (output or "")[-1500:],
    }
    if timed_out:
        return False, f"exceeded the {gc.CANDIDATE_TIMEOUT_SEC:.0f}s per-instance budget", diag
    if proc.returncode != 0:
        return False, f"deliverable exited with code {proc.returncode}", diag
    return True, "ok", diag


def _grade_instance(inst_path: Path, cand_dir: Path, io_root: Path) -> dict:
    inst = gc.load_instance(inst_path)
    parts = gc.split_arrays(inst)
    n_train = parts["X_train"].shape[0]
    n_test = parts["X_test"].shape[0]

    work = io_root / inst["name"]
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    inputs = work / "inputs.npz"
    np.savez(
        inputs,
        X_train=parts["X_train"], G_train=parts["G_train"],
        X_test=parts["X_test"], G_test=parts["G_test"],
        n_categories=np.int64(parts["n_categories"]),
        max_width=np.int64(gc.MAX_WIDTH),
    )
    os.chmod(inputs, 0o644)
    out_dir = work / "out"
    out_dir.mkdir()
    os.chown(out_dir, AGENT_UID, AGENT_GID)
    out = out_dir / "encoding.npz"

    record = {"instance": inst["name"], "onehot_mse": inst["onehot_mse"]}
    ok, reason, diag = _run_candidate(cand_dir, inputs, out)
    record.update(diag)
    if not ok:
        record.update({"valid": False, "reason": reason})
        return record

    if out.is_file() and out.stat().st_size > MAX_OUTPUT_BYTES:
        record.update({"valid": False, "reason": "output file exceeds the declared shape budget"})
        return record

    E_train, E_test, reason = gc.load_candidate_output(out, n_train, n_test)
    if E_train is None:
        record.update({"valid": False, "reason": reason})
        return record

    mse = gc.forest_mse(parts["X_train"], E_train, parts["y_train"],
                        parts["X_test"], E_test, parts["y_test"])
    record.update({
        "valid": True,
        "reason": "ok",
        "width": int(E_train.shape[1]),
        "submission_mse": mse,
        "pct_reduction": gc.pct_reduction(inst["onehot_mse"], mse),
    })
    return record


def _grade() -> dict:
    payload = {
        "split": SPLIT,
        "metric_name": "mean percentage reduction in held-out MSE vs the one-hot reference",
        "valid": False,
        "reason": "",
        "metric": None,
        "reward": 0.0,
        "per_instance": [],
    }
    instances = sorted(HIDDEN_DIR.glob("instance_*.npz"))
    if not instances:
        payload["reason"] = f"no sealed instances under {HIDDEN_DIR}"
        return payload
    payload["n_instances"] = len(instances)

    entry = DELIVERABLE_DIR / gc.ENTRY_POINT
    if not entry.is_file() or entry.stat().st_size == 0:
        payload["reason"] = f"missing or empty entry point at {entry}"
        return payload

    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    cand_dir = SCRATCH / "candidate"
    io_root = SCRATCH / "io"
    io_root.mkdir(parents=True)
    (SCRATCH / "tmp").mkdir()
    shutil.copytree(DELIVERABLE_DIR, cand_dir, symlinks=True)
    _chown_tree(cand_dir)
    _chown_tree(SCRATCH / "tmp")
    os.chmod(SCRATCH, 0o755)
    os.chmod(io_root, 0o755)

    records = []
    for path in instances:
        record = _grade_instance(path, cand_dir, io_root)
        records.append(record)
        if not record["valid"]:
            break
    payload["per_instance"] = records
    bad = [r for r in records if not r["valid"]]
    if bad or len(records) != len(instances):
        payload["reason"] = (f"instance {bad[0]['instance']} invalid: {bad[0]['reason']}"
                             if bad else "grading stopped early")
        return payload

    pcts = [float(r["pct_reduction"]) for r in records]
    metric = float(np.mean(pcts))
    payload.update({
        "valid": True,
        "reason": "ok",
        "metric": metric,
        "reward": gc.graded_reward(metric, True),
    })
    return payload


def test_submitted_encoder() -> None:
    """Execute /app/output/encoder.py on the sealed half of this grader and score it.

    WHAT: builds each sealed instance's encoding by running the deliverable as uid 1001
    under the declared per-instance budget, then measures the held-out MSE of the fixed
    forest on [covariates | encoding] against the one-hot reference MSE cached with each
    instance. WHY: the reward has to come from executing the submitted code on inputs
    that did not exist in the agent image, not from reading an artifact.
    """
    try:
        payload = _grade()
    except Exception as exc:
        payload = {"split": SPLIT, "valid": False, "metric": None, "reward": 0.0,
                   "reason": f"grader fault: {type(exc).__name__}: {exc}"}
        _write_metric(payload)
        raise
    _write_metric(payload)

    if SPLIT == "intermediate":
        print(f"graded_instances={payload.get('n_instances', 0)}")
        if payload["valid"]:
            print(f"mean_pct_mse_reduction_vs_onehot={payload['metric']:.4f}")
        else:
            print(f"invalid_submission: {payload['reason']}")
    else:
        for rec in payload.get("per_instance", []):
            print(json.dumps(rec, sort_keys=True))
        print(f"metric={payload['metric']} reward={payload['reward']}")

    assert payload["valid"], payload["reason"]
