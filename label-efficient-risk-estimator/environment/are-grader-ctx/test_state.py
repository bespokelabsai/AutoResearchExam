import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grader_core as GC

PY = "/usr/local/bin/python3"
TESTS_DIR = Path(__file__).resolve().parent
RUNNER_SRC = TESTS_DIR / "candidate_runner.py"
DELIVERABLE_DIR = Path("/app/output")
ENTRY_NAME = "estimator.py"
STAGE = Path("/grade_run")
LOGS = Path("/logs/verifier")
AGENT_UID, AGENT_GID = 1001, 1001
MAX_SUBMISSION_BYTES = 256 * 1024 * 1024
MAX_SUBMISSION_FILES = 2000

_HDR = struct.Struct("<c")
_I32 = struct.Struct("<i")
_F64 = struct.Struct("<d")


def harvest_submission():
    """Copy /app/output into a root-owned staging tree, refusing every link.

    Every submitted path is treated as hostile: symlinks and hard links are rejected
    before anything is copied (checked at every depth), and the copy itself never
    follows a link. A privileged copy that dereferenced a submitted symlink would
    materialise sealed verifier files into the tree the score is computed from.
    """
    dest = STAGE / "submission"
    if dest.exists():
        shutil.rmtree(dest)
    if DELIVERABLE_DIR.is_symlink():
        return None, f"{DELIVERABLE_DIR} is a symlink"
    if not DELIVERABLE_DIR.is_dir():
        return None, f"no deliverable directory at {DELIVERABLE_DIR}"
    total, count = 0, 0
    for root, dirs, files in os.walk(DELIVERABLE_DIR, followlinks=False):
        for name in list(dirs) + list(files):
            p = Path(root) / name
            st = p.lstat()
            if p.is_symlink():
                return None, f"symlink in the deliverable tree: {p}"
            if p.is_file() and st.st_nlink > 1:
                return None, f"hard link in the deliverable tree: {p}"
            if p.is_file():
                total += st.st_size
                count += 1
            elif not p.is_dir():
                return None, f"not a regular file or directory: {p}"
    if count == 0:
        return None, "the deliverable directory is empty"
    if total > MAX_SUBMISSION_BYTES or count > MAX_SUBMISSION_FILES:
        return None, f"deliverable too large: {count} files, {total} bytes"
    shutil.copytree(DELIVERABLE_DIR, dest, symlinks=True)
    for root, dirs, files in os.walk(dest):
        os.chmod(root, 0o755)
        for name in files:
            os.chmod(Path(root) / name, 0o644)
    entry = dest / ENTRY_NAME
    if entry.is_symlink() or not entry.is_file() or entry.stat().st_size == 0:
        return None, f"missing or empty entry point {DELIVERABLE_DIR / ENTRY_NAME}"
    return dest, None


def seal_filesystem():
    """Leave uid 1001 no writable directory, so no candidate state can survive a seed."""
    for path in ("/tmp", "/var/tmp", "/dev/shm", "/home/agent", "/app", "/app/output"):
        try:
            os.chown(path, 0, 0)
            os.chmod(path, 0o755)
        except OSError:
            pass


def _launch_runner(cell_dir, submission_dir, budget, cseed, n_seeds, seed0, child_fd):
    argv = [PY, "-I", str(cell_dir / "candidate_runner.py"),
            "--pool", str(cell_dir / "pool.npz"),
            "--submission-dir", str(submission_dir),
            "--entry", str(Path(submission_dir) / ENTRY_NAME),
            "--budget", str(budget), "--seeds", str(n_seeds),
            "--seed-start", str(seed0),
            "--cell-seed", str(cseed),
            "--per-seed-timeout", str(GC.PER_SEED_TIMEOUT_S),
            "--sock-fd", str(child_fd)]
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/home/agent",
           "TMPDIR": "/nonexistent", "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONSAFEPATH": "1", "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "NUMEXPR_NUM_THREADS": "1", "LC_ALL": "C"}
    err = subprocess.DEVNULL
    if os.environ.get("GRADE_DEBUG"):
        err = open(Path(cell_dir) / "candidate_stderr.log", "wb")
    prefix = []
    for cand, args in (("/usr/sbin/runuser", ["-u", "agent", "--"]),
                       ("/usr/bin/setpriv", [f"--reuid={AGENT_UID}",
                                             f"--regid={AGENT_GID}", "--clear-groups", "--"])):
        if os.path.exists(cand):
            prefix = [cand] + args
            break
    kwargs = {}
    if not prefix:
        kwargs = {"user": AGENT_UID, "group": AGENT_GID, "extra_groups": []}
    return subprocess.Popen(prefix + argv, env=env, start_new_session=True,
                            pass_fds=(child_fd,), stdout=subprocess.DEVNULL,
                            stderr=err, cwd=str(cell_dir), **kwargs)


def _serve(sock, labels, budget, cseed, seeds, deadline):
    """Label oracle. Enforces the budget and the one-shot, in-order seed protocol."""
    n_pool = labels.shape[0]
    estimates, used, answered, perm = {}, 0, True, None
    expected = list(seeds)
    pos, current = 0, None
    violation = None
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                violation = "cell wall-clock budget exhausted"
                break
            sock.settimeout(remaining)
            head = sock.recv(1)
            if not head:
                break
            if head == b"S":
                s = _I32.unpack(_recv_exact(sock, 4))[0]
                if pos >= len(expected) or s != expected[pos]:
                    violation = f"out-of-order seed handshake: {s} after {current}"
                    break
                current, used, answered = s, 0, False
                pos += 1
                perm = GC.seed_permutation(cseed, s, n_pool)
                sock.sendall(b"\x01")
            elif head == b"L":
                j = _I32.unpack(_recv_exact(sock, 4))[0]
                if answered or used >= budget or not (0 <= j < n_pool):
                    sock.sendall(_I32.pack(-1))
                else:
                    used += 1
                    sock.sendall(_I32.pack(int(labels[perm[j]])))
            elif head == b"R":
                s = _I32.unpack(_recv_exact(sock, 4))[0]
                v = _F64.unpack(_recv_exact(sock, 8))[0]
                if s != current or answered:
                    violation = f"duplicate or stray result for seed {s}"
                    sock.sendall(b"\x01")
                    break
                estimates[s], answered = v, True
                sock.sendall(b"\x01")
            elif head == b"F":
                _recv_exact(sock, 4)
                answered = True
                sock.sendall(b"\x01")
            elif head == b"Q":
                _recv_exact(sock, 4)
                break
            else:
                violation = f"unknown opcode {head!r}"
                break
    except (socket.timeout, ConnectionError, OSError, struct.error) as exc:
        if violation is None and len(estimates) < len(expected):
            violation = f"label channel ended early ({type(exc).__name__})"
    return estimates, violation


def _recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("candidate closed the label channel")
        buf += chunk
    return buf


def grade_cell(job):
    """Runs in a root worker process. Returns the cell's ratio and diagnostics.

    Any failure inside this worker is reported as a worst-case cell rather than being
    allowed to abort the grade, so the reward path always reaches compute_reward.py.
    """
    try:
        return _grade_cell(job)
    except Exception as exc:
        return {"instance": job[1], "budget": job[2], "risk": None,
                "ratio": GC.RATIO_CAP, "n_ok": 0, "n_seeds": job[4],
                "violation": f"scorer error: {type(exc).__name__}: {exc}"}


def _grade_cell(job):
    inst_dir, instance_id, budget, submission_dir, n_seeds, seed0 = job
    inst_dir = Path(inst_dir)
    target = np.load(inst_dir / "target_probs.npy").astype(np.float64)
    surrogate = np.load(inst_dir / "surrogate_probs.npy").astype(np.float64)
    labels = np.load(inst_dir / "labels.npy").astype(np.int64)
    cseed = GC.cell_seed(instance_id, budget)
    risk = GC.exact_risk(target, labels)
    seeds = list(range(seed0, seed0 + n_seeds))
    ref_se = GC.reference_squared_errors(target, labels, budget, cseed, seeds)

    cell_dir = Path(tempfile.mkdtemp(prefix=f"cell_{instance_id}_{budget}_", dir=str(STAGE)))
    os.chmod(cell_dir, 0o755)
    np.savez(cell_dir / "pool.npz", target_probs=target, surrogate_probs=surrogate)
    shutil.copy2(RUNNER_SRC, cell_dir / "candidate_runner.py")
    os.chmod(cell_dir / "pool.npz", 0o644)
    os.chmod(cell_dir / "candidate_runner.py", 0o644)

    parent_sock, child_sock = socket.socketpair()
    proc = None
    try:
        proc = _launch_runner(cell_dir, submission_dir, budget, cseed, n_seeds, seed0,
                              child_sock.fileno())
        child_sock.close()
        estimates, violation = _serve(parent_sock, labels, budget, cseed, seeds,
                                      time.monotonic() + GC.PER_CELL_TIMEOUT_S)
    finally:
        if os.environ.get("GRADE_DEBUG"):
            try:
                tail = (cell_dir / "candidate_stderr.log").read_bytes()[-4000:]
                if tail:
                    print(f"[stderr {instance_id} M={budget}]\n{tail.decode(errors='replace')}")
            except OSError:
                pass
        try:
            parent_sock.close()
        except OSError:
            pass
        if proc is not None:
            try:
                os.killpg(os.getpgid(proc.pid), 9)
            except OSError:
                pass
            try:
                proc.wait(timeout=30)
            except Exception:
                pass
        shutil.rmtree(cell_dir, ignore_errors=True)

    agent_se = np.full(n_seeds, GC.SE_CAP, dtype=np.float64)
    if violation is None:
        for s, v in estimates.items():
            k = s - seed0
            if 0 <= k < n_seeds and np.isfinite(v):
                agent_se[k] = min(GC.SE_CAP, (float(v) - risk) ** 2)
    n_ok = int(np.sum(agent_se < GC.SE_CAP))
    return {"instance": instance_id, "budget": budget, "risk": risk,
            "ratio": GC.cell_ratio(agent_se, ref_se), "n_ok": n_ok,
            "n_seeds": n_seeds, "violation": violation}


def _write_metric(payload):
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "metric.json").write_text(json.dumps(payload, indent=2, sort_keys=True))


def test_reward():
    """Execute the candidate estimator over every sealed cell and write metric.json.

    WHAT: drives the candidate's /app/output/estimator.py through the full adaptive
    acquisition protocol on the sealed pools, computes the median relative
    risk-estimation error against the verifier's own uniform-random reference, and maps
    it to the graded reward.
    WHY: the reward has to come from EXECUTING candidate code on inputs that were never
    in the agent image, never from reading a file the candidate wrote.
    """
    split = os.environ.get("GRADE_SPLIT", "final")
    hidden = TESTS_DIR / "hidden_data" / split
    n_seeds = int(os.environ.get("GRADE_SEEDS", GC.N_SEEDS))
    seed0 = int(os.environ.get("GRADE_SEED_OFFSET", 0))
    STAGE.mkdir(parents=True, exist_ok=True)
    os.chmod(STAGE, 0o755)

    submission_dir, reason = harvest_submission()
    if submission_dir is None:
        _write_metric({"valid": False, "reason": reason, "reward": 0.0,
                       "metric": None, "split": split})
        print(f"INVALID SUBMISSION: {reason}")
        assert False, reason

    seal_filesystem()
    instances = sorted(p for p in hidden.iterdir() if p.is_dir())
    assert instances, f"no sealed instances under {hidden}"
    jobs = [(str(p), p.name, b, str(submission_dir), n_seeds, seed0)
            for p in instances for b in GC.BUDGETS]

    workers = min(8, len(jobs))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        cells = list(pool.map(grade_cell, jobs))

    metric = GC.aggregate(c["ratio"] for c in cells)
    reward = GC.graded_reward(metric, valid=True)
    _write_metric({"valid": True, "split": split, "metric": metric, "reward": reward,
                   "n_seeds": n_seeds, "seed_offset": seed0,
                   "cells": cells})
    if split != "intermediate":
        for c in sorted(cells, key=lambda c: (c["instance"], c["budget"])):
            print(f"  {c['instance']:>16s}  M={c['budget']:>3d}  "
                  f"relative_error={c['ratio']:.4f}  seeds_ok={c['n_ok']}/{c['n_seeds']}"
                  + (f"  [{c['violation']}]" if c["violation"] else ""))
    print(f"median relative risk-estimation error = {metric:.4f} "
          f"over {len(cells)} cells")
    assert np.isfinite(metric)


if __name__ == "__main__":
    test_reward()
