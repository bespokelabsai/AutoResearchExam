import json
import os
import random
import select
import shutil
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

_SEED = int(os.environ.get("PYTHONHASHSEED", "0"))
random.seed(_SEED)
np.random.seed(_SEED)
try:
    import torch

    torch.manual_seed(_SEED)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
except ImportError:
    pass

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR))

import grader_core as G

PYTHON = "/usr/local/bin/python3"
DELIVERABLE_SRC = Path("/app/output")
RUN_DIR = Path("/tmp/mdrun")
LOG_DIR = Path("/logs/verifier")
METRIC_PATH = LOG_DIR / "metric.json"
AGENT_USER = "agent"
AGENT_UID = 1001
AGENT_GID = 1001


def _tool(name, candidates):
    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return shutil.which(name)


SETSID = _tool("setsid", ["/usr/bin/setsid", "/bin/setsid"])
RUNUSER = _tool("runuser", ["/usr/sbin/runuser", "/sbin/runuser", "/usr/bin/runuser"])


def _drop_to_agent():
    """Fallback privilege drop, used when runuser/setsid are absent from the image.

    Does exactly what `setsid runuser -u agent` does: a new session, so the process
    group can be killed as a unit, then uid/gid 1001 with no supplementary groups.
    """
    os.setsid()
    os.setgroups([])
    os.setgid(AGENT_GID)
    os.setuid(AGENT_UID)


class WorkerMaskSource:
    """Drives the low-privilege predictor process and enforces its time budgets."""

    def __init__(self, deliverable, worker_script, stderr_path):
        env = {
            "PATH": "/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin",
            "HOME": "/home/agent",
            "TMPDIR": str(RUN_DIR / "tmp"),
            "PYTHONPATH": "",
            "PYTHONSAFEPATH": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
        self._stderr = open(stderr_path, "wb")
        target = [PYTHON, str(worker_script), str(deliverable)]
        if SETSID and RUNUSER:
            argv = [SETSID, RUNUSER, "-u", AGENT_USER, "--"] + target
            preexec = None
        else:
            argv = target
            preexec = _drop_to_agent
        self.proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            cwd=str(RUN_DIR),
            env=env,
            close_fds=True,
            preexec_fn=preexec,
        )
        self.alive = True
        self._deadline = 0.0
        self.spent = 0.0


    def _kill(self):
        self.alive = False
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                self.proc.kill()
            except ProcessLookupError:
                pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass

    def _read_exact(self, count):
        fd = self.proc.stdout.fileno()
        chunks = []
        got = 0
        while got < count:
            remaining = self._deadline - time.monotonic()
            if remaining <= 0:
                self._kill()
                raise G.MaskError("time budget exhausted")
            ready, _, _ = select.select([fd], [], [], min(remaining, 5.0))
            if not ready:
                continue
            piece = os.read(fd, count - got)
            if not piece:
                self._kill()
                raise G.MaskError("predictor process exited")
            chunks.append(piece)
            got += len(piece)
        return b"".join(chunks)

    def _send(self, payload):
        try:
            self.proc.stdin.write(payload)
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            self._kill()
            raise G.MaskError("predictor process exited")

    def wait_ready(self, budget):
        self._deadline = time.monotonic() + budget
        started = time.monotonic()
        reply = self._read_exact(1)
        self.init_seconds = time.monotonic() - started
        if reply != b"I":
            self._kill()
            raise G.MaskError("construction failed")


    def begin_episode(self):
        self._deadline = time.monotonic() + G.EPISODE_BUDGET_S

    def _call(self, opcode, obs):
        if not self.alive:
            raise G.MaskError("predictor process is gone")
        started = time.monotonic()
        try:
            self._send(opcode + np.ascontiguousarray(obs, dtype=np.float32).tobytes())
            head = self._read_exact(1)
            if head == b"E":
                raise G.MaskError("predictor raised or broke the mask contract")
            if head != b"K":
                self._kill()
                raise G.MaskError("protocol violation")
            if opcode == b"P":
                raw = self._read_exact(G.N_ACTIONS)
                buf = np.frombuffer(raw, dtype=np.uint8)
                if buf.max(initial=0) > 1:
                    self._kill()
                    raise G.MaskError("mask bytes out of range")
                return buf.astype(bool)
            return None
        finally:
            self.spent += time.monotonic() - started

    def reset(self, obs):
        self._call(b"R", obs)

    def predict(self, obs):
        return self._call(b"P", obs)

    def close(self):
        if self.alive:
            try:
                self._send(b"Q")
                self.proc.wait(timeout=10)
            except (G.MaskError, subprocess.TimeoutExpired):
                self._kill()
            self.alive = False
        try:
            self._stderr.close()
        except OSError:
            pass


def _copy_no_symlinks(src, dst):
    """Copy a tree, skipping symlinks entirely.

    A symlink in the artifact would otherwise be dereferenced by this root process and
    its target copied into a directory the predictor process can read, which is a route
    out of the sealed verifier tree.
    """
    dst.mkdir(parents=True, exist_ok=True)
    for entry in sorted(src.iterdir()):
        if entry.is_symlink():
            continue
        if entry.is_dir():
            _copy_no_symlinks(entry, dst / entry.name)
        elif entry.is_file():
            shutil.copyfile(entry, dst / entry.name)


def _chown_tree(path):
    os.chown(path, AGENT_UID, AGENT_GID)
    for root, dirs, files in os.walk(path):
        for name in dirs + files:
            os.chown(os.path.join(root, name), AGENT_UID, AGENT_GID)


def _stage_deliverable():
    """Copy the artifact into a scratch tree the agent uid owns, and place the runner."""
    if RUN_DIR.exists():
        shutil.rmtree(RUN_DIR, ignore_errors=True)
    RUN_DIR.mkdir(parents=True)
    (RUN_DIR / "tmp").mkdir()
    target = RUN_DIR / "deliverable"
    if DELIVERABLE_SRC.is_dir():
        _copy_no_symlinks(DELIVERABLE_SRC, target)
    else:
        target.mkdir()
    worker = RUN_DIR / "predict_worker.py"
    shutil.copyfile(TESTS_DIR / "predict_worker.py", worker)
    worker.chmod(0o755)
    _chown_tree(target)
    os.chown(RUN_DIR / "tmp", AGENT_UID, AGENT_GID)
    RUN_DIR.chmod(0o755)
    return target, worker


def _blank_result(reason):
    return {
        "metric_name": G.METRIC_NAME,
        "metric": 0.0,
        "valid": False,
        "invalid_reason": reason,
        "mean_return": 0.0,
        "n_episodes": 0,
        "reward": 0.0,
        "m0": G.M0,
        "baseline_score": G.BASELINE_SCORE,
    }


def _seal_violations():
    """Paths in the sealed tree that uid 1001 could reach. Must be empty before grading."""
    bad = []
    for path in (TESTS_DIR, G.HIDDEN_DIR, TESTS_DIR / "mdenv.py", TESTS_DIR / "grader_core.py"):
        try:
            mode = stat.S_IMODE(path.stat().st_mode)
        except OSError as exc:
            bad.append("%s: %s" % (path, exc))
            continue
        if mode & 0o077:
            bad.append("%s mode %o" % (path, mode))
    return bad


def _grade():
    violations = _seal_violations()
    if violations:
        return _blank_result("verifier tree not sealed: %s" % "; ".join(violations))

    entry = DELIVERABLE_SRC / "predictor.py"
    if not entry.is_file():
        return _blank_result("no /app/output/predictor.py")
    if entry.stat().st_size == 0:
        return _blank_result("/app/output/predictor.py is empty")

    seeds = G.eval_seeds()
    policy = G.load_policy()
    deliverable, worker = _stage_deliverable()
    source = WorkerMaskSource(deliverable, worker, RUN_DIR / "worker_stderr.log")
    try:
        try:
            source.wait_ready(G.INIT_BUDGET_S)
        except G.MaskError as exc:
            return _blank_result("construction: %s" % exc)

        returns = []
        statuses = {}
        started = time.monotonic()
        for seed in seeds:
            if time.monotonic() - started > G.TOTAL_BUDGET_S:
                statuses["global_budget_exhausted"] = statuses.get("global_budget_exhausted", 0) + 1
                returns.append(0.0)
                continue
            if not source.alive:
                statuses["worker_gone"] = statuses.get("worker_gone", 0) + 1
                returns.append(0.0)
                continue
            value, status = G.run_episode(policy, seed, source)
            returns.append(float(value))
            key = status if status == "ok" else status.split(":", 1)[0] + "_error"
            statuses[key] = statuses.get(key, 0) + 1
        arr = np.asarray(returns, dtype=np.float64)
        metric = float(arr.mean())
        return {
            "metric_name": G.METRIC_NAME,
            "metric": metric,
            "valid": True,
            "invalid_reason": None,
            "mean_return": metric,
            "median_return": float(np.median(arr)),
            "min_return": float(arr.min()),
            "max_return": float(arr.max()),
            "n_episodes": int(arr.size),
            "episode_returns": [round(float(v), 4) for v in arr],
            "episode_statuses": statuses,
            "init_seconds": round(getattr(source, "init_seconds", 0.0), 3),
            "predictor_seconds": round(source.spent, 3),
            "grading_seconds": round(time.monotonic() - started, 3),
            "reward": G.graded_reward(metric, True),
            "m0": G.M0,
            "baseline_score": G.BASELINE_SCORE,
            "reward_map": "rational_squash",
        }
    finally:
        source.close()


@pytest.fixture(scope="module")
def result():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        payload = _grade()
    except BaseException as exc:
        payload = _blank_result("verifier error: %r" % exc)
    payload["reward"] = min(1.0, max(0.0, float(payload.get("reward", 0.0))))
    METRIC_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def test_verifier_tree_is_sealed():
    """The sealed grader tree and the eval ground truth must be unreadable to uid 1001.

    This is the boundary the two-process split rests on, so it is checked rather than
    assumed: if the mode ever regressed, the predictor process could read the eval map
    seeds and the validity oracle instead of predicting them. `_grade` applies the same
    check as a fail-closed precondition; this test is what makes a regression visible in
    the report rather than only as a zero.
    """
    assert _seal_violations() == []


def test_deliverable_constructed(result):
    """The submitted predictor imported and constructed inside its declared budget.

    A failure here is the invalid path: the reward is already 0 and the run produced no
    episodes, so nothing downstream is meaningful.
    """
    assert result["valid"], result["invalid_reason"]
    assert result["init_seconds"] <= G.INIT_BUDGET_S


def test_all_sealed_episodes_were_played(result):
    """Every one of the sealed maps in this split contributed exactly one episode return.

    Guards against a partially-completed sweep silently averaging over fewer maps, which
    would make the metric incomparable between submissions.
    """
    assert result["n_episodes"] == len(G.eval_seeds()) == G.N_EVAL_EPISODES


def test_reward_is_bounded_and_recorded(result):
    """The reward written for the platform is finite and inside [0, 1].

    compute_reward.py clamps as well; this fails the report loudly if the map itself
    ever produced something outside the contract.
    """
    reward = result["reward"]
    assert np.isfinite(reward) and 0.0 <= reward <= 1.0
    assert METRIC_PATH.is_file()
