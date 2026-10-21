from __future__ import annotations

import os
import shutil
import stat
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import envs as envs_mod
import panel as panel_mod



RUN_UID_BASE = 2000
MAX_SOLUTION_BYTES = 64 * 1024 * 1024
MAX_SOLUTION_FILES = 2000


class DeliverableError(RuntimeError):
    """The submitted deliverable is missing or does not satisfy the stated contract."""


def audit_solution_tree(src: str) -> None:
    """Reject a submission we must not copy: symlinks, hard links, specials, or bulk."""
    if not os.path.isdir(src) or os.path.islink(src):
        raise DeliverableError("/app/solution is missing or is not a real directory")
    total = 0
    count = 0
    for root, dirs, files in os.walk(src, followlinks=False):
        for name in dirs + files:
            p = os.path.join(root, name)
            st = os.lstat(p)
            if stat.S_ISLNK(st.st_mode):
                raise DeliverableError(f"symlink in the deliverable: {p}")
            if stat.S_ISREG(st.st_mode):
                if st.st_nlink > 1:
                    raise DeliverableError(f"hard link in the deliverable: {p}")
                total += st.st_size
                count += 1
            elif not stat.S_ISDIR(st.st_mode):
                raise DeliverableError(f"special file in the deliverable: {p}")
    if total > MAX_SOLUTION_BYTES:
        raise DeliverableError(f"deliverable is {total} bytes, over the {MAX_SOLUTION_BYTES} cap")
    if count > MAX_SOLUTION_FILES:
        raise DeliverableError(f"deliverable holds {count} files, over the cap")
    entry = os.path.join(src, "trainer.py")
    if not os.path.isfile(entry) or os.path.getsize(entry) == 0:
        raise DeliverableError("/app/solution/trainer.py is missing or empty")
    import ast

    try:
        tree = ast.parse(open(entry, "rb").read())
    except SyntaxError as exc:
        raise DeliverableError(f"trainer.py does not parse: {exc}") from exc
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "train":
            if len(node.args.args) >= 4:
                return
            raise DeliverableError("train() must accept (env, seed, budget, report)")
    raise DeliverableError("trainer.py defines no module-level function named 'train'")


def stage(src: str, workdir: str) -> tuple[str, str]:
    """Copy the deliverable and the child-side harness into root-owned, read-only scratch."""
    audit_solution_tree(src)
    sol = os.path.join(workdir, "solution")
    shutil.copytree(src, sol, symlinks=True, ignore_dangling_symlinks=False)
    harness = os.path.join(workdir, "harness")
    os.makedirs(harness, exist_ok=True)
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "child_main.py"), harness)




    for root, dirs, files in os.walk(sol):
        for name in files:
            os.chmod(os.path.join(root, name), 0o444)
        for name in dirs:
            os.chmod(os.path.join(root, name), 0o555)
    os.chmod(sol, 0o555)
    if os.geteuid() == 0:
        for root, dirs, files in os.walk(sol):
            for name in dirs + files:
                os.lchown(os.path.join(root, name), 0, 0)
        os.chown(sol, 0, 0)
        os.chmod(workdir, 0o755)
        os.chmod(harness, 0o755)
        os.chmod(os.path.join(harness, "child_main.py"), 0o644)
    return sol, harness


def harden_shared_writable_paths() -> dict[str, str]:
    """Close every world-writable path a run could use as a cross-run cache.

    Called by the root grader before any candidate code starts.  Each run is handed a private
    scratch directory; these are the paths that would otherwise persist between runs.
    """
    out: dict[str, str] = {}
    for path in ("/tmp", "/var/tmp", "/dev/shm", "/dev/mqueue", "/home/agent", "/run/lock"):
        try:
            os.chown(path, 0, 0)
            os.chmod(path, 0o755)
            out[path] = "sealed"
        except OSError as exc:
            out[path] = type(exc).__name__
    return out


def _job(args: tuple[Any, ...]) -> dict[str, Any]:
    env_json, eval_init, run_seed, sol, harness, python_exe, launcher, name, uid = args
    spec = envs_mod.EnvSpec.from_json(env_json)
    scratch = tempfile.mkdtemp(prefix="run_")
    try:
        os.chmod(scratch, 0o700)
        if os.geteuid() == 0:
            os.chown(scratch, uid, uid)
        res = panel_mod.run_one(
            spec,
            run_seed,
            np.asarray(eval_init, dtype=np.float64),
            sol,
            harness,
            python_exe=python_exe,
            launcher=[a.format(uid=uid) for a in launcher],
            scratch_dir=scratch,
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    res["env"] = name
    res["seed"] = run_seed
    return res


def execute_panel(
    panel: dict[str, Any],
    solution_src: str,
    workdir: str | None = None,
    workers: int = 8,
    python_exe: str = "/usr/local/bin/python3",
    launcher: tuple[str, ...] = (),
    timeout: float | None = None,
) -> list[dict[str, Any]]:
    """Run the whole panel.  ``launcher`` may carry a ``{uid}`` placeholder, filled per run.  ``timeout`` bounds the panel as a whole; each individual run is
    already bounded by :mod:`panel`'s own CPU and wall meters.  Runs that do not finish inside
    the panel deadline are recorded with no checkpoints, which scores them as failures rather
    than discarding the whole grade."""
    tmp = workdir or tempfile.mkdtemp(prefix="panel_")
    try:
        sol, harness = stage(solution_src, tmp)
        jobs = [
            (env["spec"], env["eval_init"], seed, sol, harness, python_exe, launcher, env["name"], RUN_UID_BASE + i)
            for i, (env, seed) in enumerate((e, s) for e in panel["envs"] for s in panel["run_seeds"])
        ]
        out: list[dict[str, Any]] = []
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        ex = ProcessPoolExecutor(max_workers=workers)
        try:
            futures = {ex.submit(_job, j): j for j in jobs}
            for fut in as_completed(futures, timeout=None):
                r = fut.result()
                r.pop("stderr_tail", None)
                out.append(r)
                if deadline is not None and time.monotonic() > deadline:
                    for other, job in futures.items():
                        if other.done():
                            continue
                        other.cancel()
                        out.append(
                            {
                                "checkpoint_returns": [None] * panel_mod.N_CHECKPOINTS,
                                "status": "panel deadline exceeded",
                                "env": job[7],
                                "seed": job[2],
                                "child_cpu_seconds": None,
                            }
                        )
                    break
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
        return out
    finally:
        if workdir is None:
            for root, dirs, files in os.walk(tmp):
                for name in dirs:
                    try:
                        os.chmod(os.path.join(root, name), 0o700)
                    except OSError:
                        pass
            shutil.rmtree(tmp, ignore_errors=True)
