from __future__ import annotations

import ctypes
import json
import os
import resource
import select
import shutil
import signal
import stat
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np



sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sopcc

CLOCK_TICKS = os.sysconf("SC_CLK_TCK")
CPU_POLL_INTERVAL = 2.0
PR_SET_CHILD_SUBREAPER = 36

_LOCK = threading.Lock()
_WORKERS: set[int] = set()
_ESCAPED_CPU = 0.0


@dataclass
class Sandbox:
    """Resource and privilege policy applied to every candidate process."""

    cpu_seconds: float = 75.0
    timeout: float = 150.0
    address_space_bytes: int = 6 * 1024 ** 3
    file_size_bytes: int = 64 * 1024 ** 2
    open_files: int = 256
    run_as_uid: int | None = None
    run_as_gid: int | None = None
    new_session: bool = True
    env: dict = field(default_factory=dict)


class WorkerGone(Exception):
    """The candidate process died, hung, or stopped speaking the protocol."""


class _LineChannel:
    """Deadline-bounded line reads straight off a pipe fd."""

    def __init__(self, fd: int):
        self.fd = fd
        self.buf = b""

    def readline(self, deadline: float) -> bytes:
        while b"\n" not in self.buf:
            left = deadline - time.monotonic()
            if left <= 0:
                raise WorkerGone("wall-clock budget exhausted")
            ready, _, _ = select.select([self.fd], [], [], min(left, 0.5))
            if not ready:
                continue
            chunk = os.read(self.fd, 1 << 16)
            if not chunk:
                raise WorkerGone("candidate process closed the channel")
            self.buf += chunk
        line, _, self.buf = self.buf.partition(b"\n")
        return line


def _write_line(fd: int, payload: bytes, deadline: float) -> None:
    view = memoryview(payload + b"\n")
    while view:
        left = deadline - time.monotonic()
        if left <= 0:
            raise WorkerGone("wall-clock budget exhausted")
        _, ready, _ = select.select([], [fd], [], min(left, 0.5))
        if not ready:
            continue
        try:
            written = os.write(fd, view)
        except (BrokenPipeError, OSError) as exc:
            raise WorkerGone(f"candidate process is not reading: {exc}") from exc
        view = view[written:]


def _become_subreaper() -> None:
    """Adopt orphaned descendants of every candidate instead of leaving them to init.

    A candidate that forks and lets the parent exit would otherwise hand the orphan to pid 1,
    outside any tree walked below. With this set, such an orphan is re-parented to this process,
    where `_tree` recognises it (a child of ours that is not a worker we started) and it is
    charged, killed and reaped along with the candidate's own tree.
    """
    try:
        ctypes.CDLL(None, use_errno=True).prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)
    except (OSError, AttributeError):
        pass


def _scan() -> dict[int, tuple[int, float]]:
    """pid -> (ppid, CPU seconds) for every live process.

    The CPU figure is utime + stime + cutime + cstime: the process's own threads plus every
    child it has already reaped, so a helper that burns CPU and exits between two scans is
    still charged to whoever reaped it.
    """
    procs: dict[int, tuple[int, float]] = {}
    try:
        names = os.listdir("/proc")
    except OSError:
        return procs
    for name in names:
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat", "rb") as fh:
                raw = fh.read()
        except OSError:
            continue
        cut = raw.rfind(b")")
        if cut < 0:
            continue
        fields = raw[cut + 2:].split()
        if len(fields) < 15:
            continue
        try:
            procs[int(name)] = (int(fields[1]),
                                sum(int(f) for f in fields[11:15]) / CLOCK_TICKS)
        except ValueError:
            continue
    return procs


def _tree(procs: dict[int, tuple[int, float]], root: int) -> tuple[set[int], set[int]]:
    """(`root` plus its live descendants, adopted orphans plus theirs), both by parent pid.

    Parentage is what setsid(2) cannot change, which is why the session id is not used. An
    adopted orphan (a child of this process that is not a worker it started) cannot be
    attributed to one instance, so it is charged to every instance: the candidate is the same
    code in each of them. Call with `_LOCK` held.
    """
    children: dict[int, list[int]] = {}
    for pid, (ppid, _) in procs.items():
        children.setdefault(ppid, []).append(pid)

    def walk(roots) -> set[int]:
        members: set[int] = set()
        stack = list(roots)
        while stack:
            pid = stack.pop()
            if pid in members:
                continue
            members.add(pid)
            stack.extend(children.get(pid, []))
        return members

    own = walk([root])
    orphans = walk(pid for pid in children.get(os.getpid(), []) if pid not in _WORKERS) - own
    return own, orphans


def _charge(procs: dict[int, tuple[int, float]], own: set[int], orphans: set[int]) -> float:
    return sum(procs[pid][1] for pid in own | orphans if pid in procs) + _ESCAPED_CPU


def _tree_cpu(root: int) -> float:
    """CPU seconds the instance rooted at `root` is charged for right now."""
    with _LOCK:
        procs = _scan()
        return _charge(procs, *_tree(procs, root))


def _end_instance(proc) -> float:
    """Kill the instance's tree, reap what this process adopted, and return the final charge."""
    global _ESCAPED_CPU
    with _LOCK:
        procs = _scan()
        own, orphans = _tree(procs, proc.pid)
        charge = _charge(procs, own, orphans)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        for _ in range(2):
            for pid in set().union(*_tree(_scan(), proc.pid)):
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
        try:
            proc.stdin.close()
            proc.stdout.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=10)
        except Exception:
            pass
        _WORKERS.discard(proc.pid)



        deadline = time.monotonic() + 5.0
        while True:
            pending = 0
            for pid, (ppid, cpu) in _scan().items():
                if ppid != os.getpid() or pid in _WORKERS:
                    continue
                try:
                    reaped, _ = os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    continue
                if not reaped:
                    pending += 1
                elif pid not in own:
                    _ESCAPED_CPU += cpu
            if not pending or time.monotonic() >= deadline:
                break
            time.sleep(0.05)
    return charge


def validate_solution_dir(path: str, max_bytes: int) -> tuple[bool, str, int]:
    """Reject a deliverable tree that is missing, oversized, or contains links.

    A symlink or a hard link inside a tree the harness copies is an attempt to make a
    privileged copy materialise a file from somewhere else, so it is refused outright
    rather than followed.
    """
    if not os.path.isdir(path):
        return False, f"no deliverable directory at {path}", 0
    if os.path.islink(path):
        return False, "deliverable directory is a symlink", 0
    total = 0
    for root, dirs, files in os.walk(path, followlinks=False):
        for name in dirs + files:
            full = os.path.join(root, name)
            info = os.lstat(full)
            if stat.S_ISLNK(info.st_mode):
                return False, f"symlink in deliverable: {os.path.relpath(full, path)}", total
            if stat.S_ISREG(info.st_mode):
                if info.st_nlink > 1:
                    return False, f"hard link in deliverable: {os.path.relpath(full, path)}", total
                total += info.st_size
            elif not stat.S_ISDIR(info.st_mode):
                return False, f"non-regular file in deliverable: {os.path.relpath(full, path)}", total
            if total > max_bytes:
                return False, f"deliverable exceeds {max_bytes} bytes", total
    if not os.path.isfile(os.path.join(path, "policy.py")):
        return False, f"no entry point at {os.path.join(path, 'policy.py')}", total
    return True, "ok", total


def stage_candidate(src: str, dst: str, own_uid: int | None = None) -> str:
    """Copy a validated deliverable to a location the candidate cannot rewrite mid-grade.

    `src` is treated as hostile and must already have passed `validate_solution_dir`; the
    copy keeps symlinks as symlinks rather than following them. The caller is responsible
    for `dst`'s parent being traversable by the uid the candidate runs as.
    """
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True)
    for root, dirs, files in os.walk(dst):
        for name in dirs:
            os.chmod(os.path.join(root, name), 0o755)
        for name in files:
            os.chmod(os.path.join(root, name), 0o644)
    os.chmod(dst, 0o755)
    if own_uid is not None:
        for root, dirs, files in os.walk(dst):
            for name in dirs + files:
                os.chown(os.path.join(root, name), own_uid, own_uid)
        os.chown(dst, own_uid, own_uid)
    return dst


def _instance_payload(instance: dict) -> dict:
    return {
        "coords": instance["coords"].tolist(),
        "rewards": instance["rewards"].tolist(),
        "dist": instance["dist"].tolist(),
        "start": int(instance["start"]),
        "goal": int(instance["goal"]),
        "budget": float(instance["budget"]),
        "p_fail": float(instance["p_fail"]),
        "cost_model": str(instance["cost_model"]),
    }


def run_instance(
    sandbox: Sandbox,
    candidate_dir: str,
    worker_path: str,
    python_exe: str,
    instance: dict,
    noise: np.ndarray,
    seed: int,
    episodes: int = sopcc.EPISODES_PER_INSTANCE,
) -> dict:
    """Grade one instance: `episodes` episodes of one candidate process."""
    import subprocess

    limits = {
        "cpu_seconds": float(sandbox.cpu_seconds),
        "address_space_bytes": int(sandbox.address_space_bytes),
        "file_size_bytes": int(sandbox.file_size_bytes),
        "open_files": int(sandbox.open_files),
    }
    cmd = [python_exe, "-I", "-B", worker_path, candidate_dir, json.dumps(limits)]
    privilege = {}
    if sandbox.run_as_uid is not None:
        privilege = {"user": int(sandbox.run_as_uid),
                     "group": int(sandbox.run_as_gid if sandbox.run_as_gid is not None
                                  else sandbox.run_as_uid),
                     "extra_groups": []}



    scratch = tempfile.mkdtemp(prefix="sopcc-run-")
    if sandbox.run_as_uid is not None:
        os.chown(scratch, int(sandbox.run_as_uid),
                 int(sandbox.run_as_gid if sandbox.run_as_gid is not None
                     else sandbox.run_as_uid))
    os.chmod(scratch, 0o700)
    env = {
        "PATH": "/usr/local/bin:/usr/local/sbin:/usr/bin:/bin",
        "HOME": scratch,
        "TMPDIR": scratch,
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    env.update(sandbox.env)

    errlog = open(os.path.join(scratch, "stderr.log"), "w+b")
    deadline = time.monotonic() + float(sandbox.timeout)
    report = {
        "seed": int(seed),
        "episodes": [],
        "cpu_seconds": 0.0,
        "wall_seconds": 0.0,
        "abort": None,
        "stderr_tail": "",
    }
    started = time.monotonic()
    proc = None
    _become_subreaper()
    try:
        with _LOCK:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=errlog,
                cwd="/",
                env=env,
                close_fds=True,
                start_new_session=bool(sandbox.new_session),
                **privilege,
            )
            _WORKERS.add(proc.pid)
        try:
            resource.prlimit(proc.pid, resource.RLIMIT_CPU,
                             (int(sandbox.cpu_seconds) + 1, int(sandbox.cpu_seconds) + 3))
        except (PermissionError, ProcessLookupError, OSError):
            pass
        reader = _LineChannel(proc.stdout.fileno())
        writer = proc.stdin.fileno()

        next_poll = [time.monotonic()]

        def _exchange(payload: dict) -> dict:
            _write_line(writer, json.dumps(payload).encode(), deadline)
            now = time.monotonic()
            if now >= next_poll[0]:
                next_poll[0] = now + CPU_POLL_INTERVAL
                cpu = _tree_cpu(proc.pid)
                report["cpu_seconds"] = max(report["cpu_seconds"], cpu)
                if cpu > sandbox.cpu_seconds:
                    raise WorkerGone(f"cpu budget exhausted ({cpu:.1f}s)")
            line = reader.readline(deadline)
            try:
                return json.loads(line)
            except (ValueError, UnicodeDecodeError) as exc:
                raise WorkerGone(f"unparseable reply: {exc}") from exc

        ack = _exchange({"cmd": "init", "seed": int(seed),
                         "instance": _instance_payload(instance)})
        if not ack.get("ok"):
            report["abort"] = f"{ack.get('stage', 'init')}: {ack.get('error', 'refused')}"
            raise WorkerGone(report["abort"])

        for episode in range(int(episodes)):
            def choose(obs):
                message = {
                    "cmd": "act",
                    "obs": {
                        "current": int(obs["current"]),
                        "visited": [bool(v) for v in obs["visited"]],
                        "remaining_budget": float(obs["remaining_budget"]),
                        "collected_reward": float(obs["collected_reward"]),
                        "step": int(obs["step"]),
                    },
                }
                answer = _exchange(message)
                if not answer.get("ok"):
                    raise sopcc.InvalidAction(
                        f"policy raised {answer.get('error', 'an exception')}")
                action = answer.get("action")
                if action is None:
                    raise sopcc.InvalidAction(
                        f"action is {answer.get('kind', 'None')}, not an int")
                return int(action)

            outcome = sopcc.run_episode(instance, noise[episode], choose)
            report["episodes"].append(outcome)
    except WorkerGone as exc:
        report["abort"] = report["abort"] or str(exc)
    except Exception as exc:
        report["abort"] = f"{type(exc).__name__}: {exc}"
    finally:
        if proc is not None:
            report["cpu_seconds"] = max(report["cpu_seconds"], _end_instance(proc))
        report["wall_seconds"] = time.monotonic() - started
        try:
            errlog.seek(0)
            report["stderr_tail"] = errlog.read(4096).decode("utf-8", "replace")
        except OSError:
            pass
        errlog.close()
        shutil.rmtree(scratch, ignore_errors=True)

    while len(report["episodes"]) < int(episodes):
        report["episodes"].append(
            {"delivered": 0.0, "failed": True,
             "reason": f"aborted: {report['abort'] or 'unknown'}", "visits": 0,
             "traversals": 0})
    return report


def grade_panel(
    panel,
    sandbox: Sandbox,
    candidate_dir: str,
    worker_path: str,
    python_exe: str | None = None,
    workers: int = 8,
    seed_base: int = 1000,
    episodes: int = sopcc.EPISODES_PER_INSTANCE,
    retry_budget: int = 4,
) -> dict:
    """Grade a whole panel, at most `workers` instances at a time, and aggregate the metric."""
    python_exe = python_exe or sys.executable
    reports: list[dict | None] = [None] * len(panel)
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(panel)))) as pool:
        futures = {
            pool.submit(
                run_instance, sandbox, candidate_dir, worker_path, python_exe,
                panel[i]["instance"], panel[i]["noise"], seed_base + i, episodes,
            ): i
            for i in range(len(panel))
        }
        for future, i in futures.items():
            reports[i] = future.result()




    retries = []
    for i, report in enumerate(reports):
        if retry_budget <= 0:
            break
        abort = report["abort"] or ""
        infra = ("closed the channel" in abort) or ("is not reading" in abort)
        if not infra or report["wall_seconds"] > 0.25 * sandbox.timeout:
            continue
        retry_budget -= 1
        retries.append({"instance": i, "first_abort": abort})
        reports[i] = run_instance(
            sandbox, candidate_dir, worker_path, python_exe,
            panel[i]["instance"], panel[i]["noise"], seed_base + i, episodes)

    delivered, failed, draws = [], [], []
    for i, report in enumerate(reports):
        for outcome in report["episodes"]:
            delivered.append(outcome["delivered"])
            failed.append(outcome["failed"])
            draws.append(panel[i].get("draw", 0))
    summary = sopcc.panel_metric(delivered, failed)
    delivered_arr = np.asarray(delivered, dtype=np.float64)
    failed_arr = np.asarray(failed, dtype=bool)
    draw_arr = np.asarray(draws, dtype=np.int64)
    per_draw = {}
    for draw in sorted(set(draws)):
        mask = draw_arr == draw
        per_draw[int(draw)] = sopcc.panel_metric(delivered_arr[mask], failed_arr[mask])
    summary["per_draw"] = per_draw
    summary["draw_metric_mean"] = float(
        np.mean([per_draw[d]["metric"] for d in sorted(per_draw)])) if per_draw else 0.0
    summary["instances"] = len(panel)
    summary["retried_instances"] = retries
    summary["aborted_instances"] = sum(1 for r in reports if r["abort"])
    summary["max_cpu_seconds"] = max((r["cpu_seconds"] for r in reports), default=0.0)
    summary["max_wall_seconds"] = max((r["wall_seconds"] for r in reports), default=0.0)
    summary["reports"] = reports
    return summary
