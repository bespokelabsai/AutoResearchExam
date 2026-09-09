from __future__ import annotations

import ctypes
import json
import os
import select
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import envs as envs_mod

N_ENVS = 64
N_EVAL_STATES = 32
TOTAL_ENV_STEPS = 400_000
N_CHECKPOINTS = 40
CPU_SECONDS = 45.0
WALL_SECONDS = 150.0

_HDR = struct.Struct("<cI")
_CLK = os.sysconf("SC_CLK_TCK")


@dataclass(frozen=True)
class Budget:
    total_env_steps: int = TOTAL_ENV_STEPS
    cpu_seconds: float = CPU_SECONDS
    n_checkpoints: int = N_CHECKPOINTS
    wall_seconds: float = WALL_SECONDS


class ProtocolError(RuntimeError):
    pass


def _write_blocking(fd: int, kind: bytes, payload: bytes = b"") -> None:
    """Unguarded write, used only for the handshake before any limit can bite."""
    buf = _HDR.pack(kind, len(payload)) + payload
    while buf:
        buf = buf[os.write(fd, buf) :]


def _stat_cpu(entry: str) -> tuple[int, float]:
    """(parent pid, utime+stime+cutime+cstime in seconds) for one /proc entry."""
    with open(f"/proc/{entry}/stat", "rb") as fh:
        fields = fh.read().rsplit(b")", 1)[1].split()
    cpu = (int(fields[11]) + int(fields[12]) + int(fields[13]) + int(fields[14])) / _CLK
    return int(fields[1]), cpu


def _child_cpu_seconds(pid: int) -> float:
    """CPU consumed by the launched process itself, including reaped descendants."""
    try:
        return _stat_cpu(str(pid))[1]
    except (OSError, IndexError, ValueError):
        return 0.0


_PR_SET_CHILD_SUBREAPER = 36


def _become_subreaper() -> None:
    """Make this process the reaper for every orphan below it.

    A run's process that calls ``setsid`` or double-forks would otherwise be reparented to
    init and vanish from any parent-chain walk; with this set it is reparented here instead,
    so it stays inside :func:`_descendants` for both the CPU meter and the final SIGKILL.
    """
    try:
        ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)
    except (OSError, AttributeError):
        pass


def _descendants(root: int) -> dict[int, float]:
    """{pid: cpu seconds} for every live process below ``root`` in the parent chain.

    The launched process is a privilege-dropping wrapper that forks, so reading its own
    /proc entry alone would meter the wrapper and not the training procedure.  Walking the
    parent chain rather than matching a process-group id means a worker that leaves the
    group (``os.setsid``) is still counted, and :func:`_become_subreaper` keeps double-forked
    orphans in the walk too.
    """
    ppid: dict[int, int] = {}
    cpu: dict[int, float] = {}
    try:
        entries = os.listdir("/proc")
    except OSError:
        return {}
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            ppid[int(entry)], cpu[int(entry)] = _stat_cpu(entry)
        except (OSError, IndexError, ValueError):
            continue
    kids: dict[int, list[int]] = {}
    for pid, parent in ppid.items():
        kids.setdefault(parent, []).append(pid)
    out: dict[int, float] = {}
    stack = list(kids.get(root, []))
    while stack:
        pid = stack.pop()
        out[pid] = cpu[pid]
        stack.extend(kids.get(pid, []))
    return out


def _subtree_cpu_seconds(root: int) -> float:
    """CPU consumed by every live process below ``root``, reaped descendants included."""
    return sum(_descendants(root).values())






_AUDIT_ARCH_X86_64 = 0xC000003E
_SECCOMP_DENY = (
    29, 30, 31, 67,
    64, 65, 66,
    68, 69, 70, 71,
    240, 241, 242, 243, 244, 245,
    49,
    248, 249, 250,
    101, 310, 311, 438,
    129, 297, 424,
)


class _SockFilter(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint16), ("jt", ctypes.c_uint8), ("jf", ctypes.c_uint8), ("k", ctypes.c_uint32)]


class _SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_uint16), ("filter", ctypes.POINTER(_SockFilter))]


def _install_syscall_filter() -> None:
    """Refuse the cross-process IPC syscalls with EPERM; kill the process on a foreign ABI."""
    ld_w_abs, jeq_k, jge_k, ret_k = 0x20, 0x15, 0x35, 0x06
    allow, kill, eperm = 0x7FFF0000, 0x80000000, 0x00050000 | 1
    ins = [
        (ld_w_abs, 0, 0, 4), (jeq_k, 1, 0, _AUDIT_ARCH_X86_64), (ret_k, 0, 0, kill),
        (ld_w_abs, 0, 0, 0), (jge_k, 0, 1, 0x40000000), (ret_k, 0, 0, kill),
    ]
    for nr in _SECCOMP_DENY:
        ins += [(jeq_k, 0, 1, nr), (ret_k, 0, 0, eperm)]
    ins.append((ret_k, 0, 0, allow))
    arr = (_SockFilter * len(ins))(*[_SockFilter(*i) for i in ins])
    prog = _SockFprog(len(ins), arr)
    libc = ctypes.CDLL(None, use_errno=True)


    if libc.prctl(38, 1, 0, 0, 0) != 0 or libc.prctl(22, 2, ctypes.byref(prog), 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "seccomp filter install failed")


def _decode_actions(payload: bytes, rows: int) -> np.ndarray:
    want = rows * envs_mod.ACT_DIM * 4
    if len(payload) != want:
        raise ProtocolError(f"action payload is {len(payload)} bytes, expected {want}")
    a = np.frombuffer(payload, dtype="<f4").reshape(rows, envs_mod.ACT_DIM)
    if not np.all(np.isfinite(a)):
        raise ProtocolError("action contains a non-finite value")
    if float(np.max(np.abs(a))) > 1.0 + 1e-4:
        raise ProtocolError("action outside [-1, 1]")
    return np.clip(a.astype(np.float64), -1.0, 1.0)


class _Session:
    """One metered training run against one environment and one seed."""

    def __init__(
        self,
        spec: envs_mod.EnvSpec,
        run_seed: int,
        budget: Budget,
        eval_init: np.ndarray,
        solution_dir: str,
        harness_dir: str,
        python_exe: str,
        launcher: list[str] | None = None,
        scratch_dir: str | None = None,
    ):
        self.spec = spec
        self.budget = budget
        self.train_env = envs_mod.VecEnv(spec, n_envs=N_ENVS, seed=run_seed)
        self.eval_env = envs_mod.VecEnv(
            spec, n_envs=eval_init.shape[0], seed=0, init_states=eval_init
        )
        self.steps = 0
        self.per_ckpt = budget.total_env_steps // budget.n_checkpoints
        self.next_ckpt = 1
        self.returns: list[float | None] = [None] * budget.n_checkpoints
        self.policy_ready = False
        self.status = "ok"
        self.cpu_base = 0.0
        self._metered = False
        self._cpu_seen = 0.0
        self._cpu_probe_t = 0.0
        self._group_probe_t = 0.0

        c2p_r, c2p_w = os.pipe()
        p2c_r, p2c_w = os.pipe()
        cmd = list(launcher or []) + [
            python_exe,
            os.path.join(harness_dir, "child_main.py"),
            solution_dir,
            str(p2c_r),
            str(c2p_w),
        ]
        env = {
            "PATH": "/usr/local/bin:/usr/local/sbin:/usr/bin:/usr/sbin:/bin:/sbin",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONSAFEPATH": "1",




            "TMPDIR": scratch_dir or solution_dir,
            "HOME": scratch_dir or "/tmp",
        }
        _become_subreaper()
        self.proc = subprocess.Popen(
            cmd,
            pass_fds=(p2c_r, c2p_w),
            env=env,
            cwd=scratch_dir or solution_dir,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
            preexec_fn=_install_syscall_filter,
        )
        os.close(p2c_r)
        os.close(c2p_w)
        self.rx = c2p_r
        self.tx = p2c_w
        self.t0 = time.monotonic()
        _write_blocking(
            self.tx,
            b"I",
            json.dumps(
                {
                    "n_envs": N_ENVS,
                    "obs_dim": envs_mod.OBS_DIM,
                    "act_dim": envs_mod.ACT_DIM,
                    "max_episode_steps": envs_mod.MAX_EPISODE_STEPS,
                    "total_env_steps": budget.total_env_steps,
                    "cpu_seconds": budget.cpu_seconds,
                    "n_checkpoints": budget.n_checkpoints,
                    "seed": int(run_seed),
                }
            ).encode(),
        )



    def _cpu_used(self) -> float:
        """Monotone estimate of the run's CPU seconds, throttled to stay cheap."""
        now = time.monotonic()
        if now - self._cpu_probe_t >= 0.05:
            self._cpu_probe_t = now
            seen = _child_cpu_seconds(self.proc.pid)
            if now - self._group_probe_t >= 0.5:
                self._group_probe_t = now
                seen = max(seen, _subtree_cpu_seconds(os.getpid()))
            self._cpu_seen = max(self._cpu_seen, seen)
        return self._cpu_seen

    def _check_limits(self) -> None:
        if time.monotonic() - self.t0 > self.budget.wall_seconds:
            raise ProtocolError("wall-clock budget exhausted")
        if self._cpu_used() - self.cpu_base > self.budget.cpu_seconds:
            raise ProtocolError("cpu budget exhausted")



    def _send(self, kind: bytes, payload: bytes = b"") -> None:
        buf = _HDR.pack(kind, len(payload)) + payload
        while buf:
            self._check_limits()
            if not select.select([], [self.tx], [], 0.5)[1]:
                continue
            try:
                buf = buf[os.write(self.tx, buf) :]
            except OSError as exc:
                raise ProtocolError(f"write to child failed: {exc}") from exc

    def _recv_exact(self, n: int) -> bytes:
        out = bytearray()
        while len(out) < n:
            self._check_limits()
            if not select.select([self.rx], [], [], 0.5)[0]:
                continue
            try:
                chunk = os.read(self.rx, n - len(out))
            except OSError as exc:
                raise ProtocolError(f"read from child failed: {exc}") from exc
            if not chunk:
                raise ProtocolError("child closed the pipe")
            out += chunk
        return bytes(out)

    def _recv(self) -> tuple[bytes, bytes]:
        kind, length = _HDR.unpack(self._recv_exact(_HDR.size))
        if length > 1 << 22:
            raise ProtocolError("oversized frame")
        return kind, self._recv_exact(length)



    def _query_actions(self, obs: np.ndarray) -> np.ndarray:
        self._send(b"Q", np.ascontiguousarray(obs, dtype="<f4").tobytes())
        kind, payload = self._recv()
        if kind != b"A":
            raise ProtocolError(f"expected an evaluation answer, got {kind!r}")
        return _decode_actions(payload, obs.shape[0])

    def _evaluate(self) -> float:
        obs = self.eval_env.reset()
        total = np.zeros(self.eval_env.n_envs, dtype=np.float64)
        for _ in range(envs_mod.MAX_EPISODE_STEPS):
            self._check_limits()
            act = self._query_actions(obs)
            obs, reward, _ = self.eval_env.step(act)
            total += reward
        return float(np.mean(total))

    def _run_due_checkpoints(self) -> None:
        while self.next_ckpt <= self.budget.n_checkpoints and self.steps >= self.next_ckpt * self.per_ckpt:
            if self.policy_ready:
                self.returns[self.next_ckpt - 1] = self._evaluate()
            self.next_ckpt += 1



    def run(self) -> dict[str, Any]:
        try:
            self._serve()
        except ProtocolError as exc:
            self.status = str(exc)
        except Exception as exc:
            self.status = f"harness error: {type(exc).__name__}: {exc}"
        finally:
            self._cpu_probe_t = 0.0
            self._group_probe_t = 0.0
            self.cpu = max(0.0, self._cpu_used() - self.cpu_base)
            self._shutdown()
        return {
            "checkpoint_returns": self.returns,
            "status": self.status,
            "env_steps": self.steps,
            "child_cpu_seconds": round(self.cpu, 3),
            "wall_seconds": round(time.monotonic() - self.t0, 3),
        }

    def _serve(self) -> None:
        exhausted = False
        while True:
            kind, payload = self._recv()
            first = not self._metered
            if first:



                self._metered = True
                self._cpu_probe_t = 0.0
                self._group_probe_t = 0.0
                self.cpu_base = self._cpu_used()
                self.t0 = time.monotonic()
            if kind == b"R":
                if payload:
                    raise ProtocolError("reset takes no payload")
                obs = self.train_env.reset()
                self._send(b"O", np.ascontiguousarray(obs, dtype="<f4").tobytes())
            elif kind == b"S":
                if exhausted or self.steps >= self.budget.total_env_steps:
                    exhausted = True
                    self._send(b"E")
                    continue
                act = _decode_actions(payload, N_ENVS)
                obs, reward, done = self.train_env.step(act)
                self.steps += N_ENVS
                body = (
                    np.ascontiguousarray(obs, dtype="<f4").tobytes()
                    + np.ascontiguousarray(reward, dtype="<f4").tobytes()
                    + np.ascontiguousarray(done, dtype=np.uint8).tobytes()
                )



                self._run_due_checkpoints()
                self._send(b"T", body)
            elif kind == b"G":


                if not first:
                    raise ProtocolError("duplicate G")
                self._send(b"K")
            elif kind == b"P":
                self.policy_ready = True
                self._send(b"K")
            elif kind == b"D":
                self.steps = self.budget.total_env_steps
                self._run_due_checkpoints()
                return
            else:
                raise ProtocolError(f"unexpected message {kind!r}")

    def _kill_subtree(self) -> None:
        """SIGKILL every process below this one and reap what it adopted as subreaper.

        Repeats until the walk is empty, so a process forked between one snapshot and the
        kill is caught on the next pass and nothing survives into, or is metered against,
        the next run.
        """
        deadline = time.monotonic() + 10.0
        while True:
            pids = _descendants(os.getpid())
            if not pids:
                return
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
                try:
                    os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    pass
            if time.monotonic() > deadline:
                return
            time.sleep(0.01)

    def _shutdown(self) -> None:
        try:
            _write_blocking(self.tx, b"X")
        except OSError:
            pass
        for fd in (self.tx, self.rx):
            try:
                os.close(fd)
            except OSError:
                pass
        self._kill_subtree()
        try:
            self.stderr_tail = (self.proc.stderr.read() or b"")[-1500:].decode("utf-8", "replace")
        except Exception:
            self.stderr_tail = ""
        finally:
            try:
                self.proc.stderr.close()
            except Exception:
                pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def run_one(
    spec: envs_mod.EnvSpec,
    run_seed: int,
    eval_init: np.ndarray,
    solution_dir: str,
    harness_dir: str,
    python_exe: str = sys.executable,
    budget: Budget | None = None,
    launcher: list[str] | None = None,
    scratch_dir: str | None = None,
) -> dict[str, Any]:
    """Execute one metered training run and return its checkpoint return series.

    ``scratch_dir`` is the run's private working directory.  When it is not supplied a fresh
    one is created and removed here, so a run never inherits state from an earlier run.
    """
    owned = scratch_dir is None
    if owned:
        scratch_dir = tempfile.mkdtemp(prefix="run_")
        os.chmod(scratch_dir, 0o700)
    session = _Session(
        spec=spec,
        run_seed=run_seed,
        budget=budget or Budget(),
        eval_init=eval_init,
        solution_dir=solution_dir,
        harness_dir=harness_dir,
        python_exe=python_exe,
        launcher=launcher,
        scratch_dir=scratch_dir,
    )
    try:
        result = session.run()
    finally:
        if owned:
            shutil.rmtree(scratch_dir, ignore_errors=True)
    result["stderr_tail"] = getattr(session, "stderr_tail", "")
    return result


def random_action_return(spec: envs_mod.EnvSpec, eval_init: np.ndarray, seed: int) -> float:
    """Mean undiscounted return of uniform-random actions on the evaluation states."""
    env = envs_mod.VecEnv(spec, n_envs=eval_init.shape[0], seed=0, init_states=eval_init)
    rng = np.random.default_rng(seed)
    obs = env.reset()
    total = np.zeros(env.n_envs)
    for _ in range(envs_mod.MAX_EPISODE_STEPS):
        obs, reward, _ = env.step(rng.uniform(-1.0, 1.0, size=(env.n_envs, envs_mod.ACT_DIM)))
        total += reward
    return float(np.mean(total))
