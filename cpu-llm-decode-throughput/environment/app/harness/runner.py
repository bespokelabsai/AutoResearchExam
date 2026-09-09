from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time

READY_POLL_S = 0.002




def _child(engine_path, model_dir, warmup_path, results_path, ready_path):
    import importlib.util

    os.environ["HOME"] = os.getcwd()

    import torch

    torch.set_num_threads(len(os.sched_getaffinity(0)))
    torch.manual_seed(0)

    sys.path.insert(0, os.path.dirname(os.path.abspath(engine_path)))
    spec = importlib.util.spec_from_file_location("graded_engine", engine_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    engine = module.Engine(model_dir)
    with open(warmup_path) as fh:
        engine.generate(json.load(fh))

    with open(ready_path, "w") as fh:
        fh.write("ready")
        fh.flush()
        os.fsync(fh.fileno())

    payload = sys.stdin.buffer.read()
    requests = json.loads(payload.decode("utf-8"))

    result = engine.generate(requests)

    with open(results_path, "w") as fh:
        json.dump(result, fh)
        fh.flush()
        os.fsync(fh.fileno())
    sys.stderr.flush()
    os._exit(0)



def _kill_uid(uid):
    """SIGKILL every process still running as `uid`.

    A descendant that calls setsid() leaves the process group and survives killpg; this
    is the backstop, so nothing a finished run left behind can consume cores during the
    next one.
    """
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            if os.stat(os.path.join("/proc", entry)).st_uid != uid:
                continue
            os.kill(int(entry), signal.SIGKILL)
        except (OSError, ProcessLookupError):
            continue




class RunOutcome:
    """What one timed run produced. ``ok`` false means the run scores nothing."""

    def __init__(self, ok, elapsed_s, outputs, detail):
        self.ok = ok
        self.elapsed_s = elapsed_s
        self.outputs = outputs
        self.detail = detail

    def __repr__(self):
        return f"RunOutcome(ok={self.ok}, elapsed_s={self.elapsed_s}, detail={self.detail!r})"


def run_timed(
    engine_path,
    model_dir,
    warmup_path,
    workload,
    run_dir,
    load_budget_s,
    run_budget_s,
    cmd_prefix=(),
    env_extra=None,
    runner_path=None,
    reap_uid=None,
):
    """Run one engine over one workload in a fresh process and time only generate().

    ``cmd_prefix`` lets a privileged caller wrap the launch (e.g. drop to another uid),
    and ``runner_path`` lets it point the child at a copy of this file that the reduced
    uid can actually read. Returns a :class:`RunOutcome`; every failure returns ok=False.
    """
    results_path = os.path.join(run_dir, "results.json")
    ready_path = os.path.join(run_dir, "ready.marker")
    log_path = os.path.join(run_dir, "engine.log")
    for stale in (results_path, ready_path):
        if os.path.exists(stale):
            os.unlink(stale)

    env = {
        "PATH": "/usr/local/bin:/usr/local/sbin:/usr/bin:/usr/sbin:/bin:/sbin",
        "HOME": run_dir,
        "TMPDIR": run_dir,
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
        "OMP_NUM_THREADS": str(len(os.sched_getaffinity(0))),
        "MKL_NUM_THREADS": str(len(os.sched_getaffinity(0))),
        "OPENBLAS_NUM_THREADS": str(len(os.sched_getaffinity(0))),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    env.update(env_extra or {})

    argv = list(cmd_prefix) + [
        sys.executable,
        "-s",
        "-P",
        runner_path or os.path.abspath(__file__),
        "--child",
        engine_path,
        model_dir,
        warmup_path,
        results_path,
        ready_path,
    ]
    payload = json.dumps(workload).encode("utf-8")

    log = open(log_path, "wb")
    proc = subprocess.Popen(
        argv,
        stdin=subprocess.PIPE,
        stdout=log,
        stderr=log,
        cwd=run_dir,
        env=env,
        start_new_session=True,
    )

    pgid = proc.pid

    def _reap():
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        if reap_uid is not None:
            _kill_uid(reap_uid)
        log.close()

    deadline = time.monotonic() + load_budget_s
    while not os.path.exists(ready_path):
        if proc.poll() is not None:
            _reap()
            return RunOutcome(False, None, None, "engine exited before it was ready")
        if time.monotonic() > deadline:
            _reap()
            return RunOutcome(False, None, None, "engine exceeded the load budget")
        time.sleep(READY_POLL_S)

    def _feed():
        try:
            proc.stdin.write(payload)
            proc.stdin.close()
        except (BrokenPipeError, OSError, ValueError):
            pass

    started = time.perf_counter()
    threading.Thread(target=_feed, daemon=True).start()
    try:
        returncode = proc.wait(timeout=run_budget_s)
    except subprocess.TimeoutExpired:
        _reap()
        return RunOutcome(False, None, None, "engine exceeded the wall-clock budget")
    elapsed = time.perf_counter() - started

    try:
        with open(results_path) as fh:
            outputs = json.load(fh)
    except (OSError, ValueError) as exc:
        outputs = None
        detail = f"unreadable results: {exc}"
    else:
        detail = ""
    _reap()

    if returncode != 0:
        return RunOutcome(False, elapsed, None, f"engine exited with code {returncode}")
    if outputs is None:
        return RunOutcome(False, elapsed, None, detail)
    return RunOutcome(True, elapsed, outputs, "")


if __name__ == "__main__":
    if len(sys.argv) == 7 and sys.argv[1] == "--child":
        _child(*sys.argv[2:])
    else:
        raise SystemExit("runner.py is launched by the harness, not directly")
