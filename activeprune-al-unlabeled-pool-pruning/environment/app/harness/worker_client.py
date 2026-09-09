from __future__ import annotations

import json
import os
import selectors
import signal
import subprocess
import time

IMPORT_BUDGET_SEC = 60.0
CALL_BUDGET_SEC = 20.0
ADDRESS_SPACE_LIMIT = 4 * 1024 ** 3


class PrunerFailure(RuntimeError):
    """The submitted pruner crashed, timed out, or broke the protocol."""


class PruneWorker:
    def __init__(self, python_exe, worker_script, module_dir, entry_module, pool_texts_path,
                 scratch_dir, launch_prefix=None, new_session=os.setsid,
                 stderr_path=None, call_budget=CALL_BUDGET_SEC,
                 import_budget=IMPORT_BUDGET_SEC,
                 address_space_limit=ADDRESS_SPACE_LIMIT):
        self.python_exe = python_exe
        self.worker_script = worker_script
        self.module_dir = module_dir
        self.entry_module = entry_module
        self.pool_texts_path = pool_texts_path
        self.scratch_dir = scratch_dir
        self.launch_prefix = list(launch_prefix) if launch_prefix else []
        self.new_session = new_session
        self.stderr_path = stderr_path
        self.call_budget = call_budget
        self.import_budget = import_budget
        self.address_space_limit = address_space_limit
        self.proc = None
        self._selector = None
        self._stderr = None
        self.timings = []

    def _argv(self):
        return self.launch_prefix + [self.python_exe, self.worker_script, self.module_dir,
                                     self.entry_module, self.pool_texts_path]

    def _child_setup(self):
        limit = self.address_space_limit
        new_session = self.new_session

        def apply():
            new_session()
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        return apply

    def start(self):
        env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": self.scratch_dir,
            "TMPDIR": self.scratch_dir,
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
        self._stderr = open(self.stderr_path, "ab") if self.stderr_path \
            else open(os.devnull, "wb")
        self.proc = subprocess.Popen(
            self._argv(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._stderr, cwd=self.scratch_dir, env=env, text=True,
            bufsize=1, preexec_fn=self._child_setup())
        self._selector = selectors.DefaultSelector()
        self._selector.register(self.proc.stdout, selectors.EVENT_READ)
        hello = self._read(self.import_budget, "module import")
        if hello.get("status") != "ready":
            detail = hello.get("detail", hello)
            self.kill()
            raise PrunerFailure(f"module did not import: {detail}")

    def kill(self):
        if self.proc is None:
            return
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                self.proc.kill()
            except ProcessLookupError:
                pass
        try:
            self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pass
        if self._selector is not None:
            self._selector.close()
            self._selector = None
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None
        self.proc = None

    def close(self):
        if self.proc is None:
            return
        try:
            self.proc.stdin.write(json.dumps({"op": "stop"}) + "\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=10)
        except Exception:
            pass
        self.kill()

    def _read(self, budget, what):
        deadline = time.monotonic() + budget
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.kill()
                raise PrunerFailure(f"{what} exceeded its {budget:g}s budget")
            if not self._selector.select(timeout=min(remaining, 1.0)):
                if self.proc.poll() is not None:
                    self.kill()
                    raise PrunerFailure(f"{what}: process exited without replying")
                continue
            line = self.proc.stdout.readline()
            if line == "":
                self.kill()
                raise PrunerFailure(f"{what}: channel closed without a reply")
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError as exc:
                self.kill()
                raise PrunerFailure(f"{what}: unparseable reply ({exc})") from None

    def call(self, available, labeled_indices, labeled_labels, keep, iteration, rng_seed):
        if self.proc is None:
            raise PrunerFailure("worker is not running")
        request = {"op": "prune", "available": list(map(int, available)),
                   "labeled_indices": list(map(int, labeled_indices)),
                   "labeled_labels": list(map(int, labeled_labels)),
                   "keep": int(keep), "iteration": int(iteration),
                   "rng_seed": int(rng_seed)}
        started = time.monotonic()
        try:
            self.proc.stdin.write(json.dumps(request) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError) as exc:
            self.kill()
            raise PrunerFailure(f"pruner is gone before iteration {iteration}: {exc}") from None
        reply = self._read(self.call_budget, f"prune call at iteration {iteration}")
        self.timings.append(time.monotonic() - started)
        if reply.get("status") != "ok":
            detail = reply.get("detail", reply)
            self.kill()
            raise PrunerFailure(f"pruner raised at iteration {iteration}: {detail}")
        selection = reply.get("selection")
        if isinstance(selection, dict):
            self.kill()
            raise PrunerFailure(
                f"pruner returned an unusable type at iteration {iteration}: "
                f"{selection.get('bad_type')}")
        return selection
