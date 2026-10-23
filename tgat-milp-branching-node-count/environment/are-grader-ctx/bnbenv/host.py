from __future__ import annotations

import importlib.util
import io
import os
import sys
import traceback

import numpy as np

from . import protocol

BLOCKED_MODULES = (
    "pyscipopt",
    "gurobipy",
    "mip",
    "pulp",
    "highspy",
    "ortools",
    "cplex",
    "docplex",
    "xpress",
    "cylp",
)

MAX_ERROR_CHARS = 400

NATIVE_ROOTS = ("/usr/local/lib/", "/usr/lib/", "/lib/")
NATIVE_SUFFIXES = (".so", ".pyd", ".dylib")


def _audit(event, args):
    path = None
    if event == "import" and len(args) > 1:
        path = args[1]
    elif event in ("ctypes.dlopen", "ctypes.LoadLibrary") and args:
        path = args[0]
    if not isinstance(path, str) or not path:
        return
    if path.endswith(NATIVE_SUFFIXES) or ".so." in path:
        if not path.startswith(NATIVE_ROOTS):
            raise PermissionError(
                f"loading the native module {path} is not allowed inside the "
                "branching policy process")


class _BlockSolvers:
    """meta_path finder that refuses exactly the modules listed above."""

    def find_module(self, fullname, path=None):
        return None

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".", 1)[0]
        if root in BLOCKED_MODULES:
            raise ImportError(
                f"import of {root!r} is blocked inside the branching policy "
                "process")
        return None


def _load_policy_module(submission_dir, entry_point):
    path = os.path.join(submission_dir, entry_point)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"missing entry point {path}")
    if os.path.getsize(path) == 0:
        raise ValueError(f"empty entry point {path}")
    if submission_dir not in sys.path:
        sys.path.insert(0, submission_dir)
    spec = importlib.util.spec_from_file_location("agent_policy", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["agent_policy"] = module
    spec.loader.exec_module(module)
    return module


def _freeze(obs):
    for value in obs.values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    return obs


def _short(exc):
    text = f"{type(exc).__name__}: {exc}"
    return text[:MAX_ERROR_CHARS].replace("\n", " ")


def main():
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    import numpy.linalg
    sys.meta_path.insert(0, _BlockSolvers())
    sys.addaudithook(_audit)
    inp = io.open(0, "rb", closefd=False)
    out = io.open(int(os.environ["BNB_REPLY_FD"]), "wb", closefd=False)

    module = None
    policy = None
    while True:
        try:
            frame = protocol.read_obs(inp)
        except Exception:
            return 1
        if frame is None:
            return 0
        cmd = frame.get("cmd")
        if cmd == "quit":
            return 0
        if cmd == "load":
            try:
                module = _load_policy_module(frame["submission_dir"],
                                             frame["entry_point"])
                loader = getattr(module, "load", None)
                if loader is not None:
                    loader()
            except BaseException as exc:
                traceback.print_exc(file=sys.stderr)
                protocol.write_decision(out, "err " + _short(exc))
                return 0
            protocol.write_decision(out, "ready")
        elif cmd == "new":
            try:
                policy = module.Policy()
            except BaseException as exc:
                traceback.print_exc(file=sys.stderr)
                protocol.write_decision(out, "err " + _short(exc))
                return 0
            protocol.write_decision(out, "ok")
        elif cmd == "select":
            obs = _freeze(frame["obs"])
            n_cands = len(obs["cand_idx"])
            try:
                choice = policy.select(obs)
            except BaseException as exc:
                traceback.print_exc(file=sys.stderr)
                protocol.write_decision(out, "err " + _short(exc))
                return 0
            if isinstance(choice, bool) or not isinstance(
                    choice, (int, np.integer)):
                protocol.write_decision(
                    out, f"err select returned {type(choice).__name__}, "
                         "expected int")
                return 0
            index = int(choice)
            if not 0 <= index < n_cands:
                protocol.write_decision(
                    out, f"err select returned {index}, outside "
                         f"[0, {n_cands})")
                return 0
            protocol.write_decision(out, f"i {index}")
        else:
            protocol.write_decision(out, "err unknown command")
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
