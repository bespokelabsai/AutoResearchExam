import json
import os
import sys
import time

import numpy as np

_dump = json.dump
_exit = os._exit


def encode(obj):
    """Best-effort, lossless-for-integers JSON encoding of an untrusted return value."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [encode(x) for x in obj.tolist()]
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [encode(x) for x in obj]
    try:
        return [encode(x) for x in list(obj)]
    except Exception:
        return {"__unencodable__": type(obj).__name__}


def main() -> None:
    work, candidate = sys.argv[1], sys.argv[2]
    out_path = os.path.join(work, "result.json")
    payload = {"ok": False, "error": "runner did not complete"}
    try:
        with open(os.path.join(work, "args.json"), "r") as fh:
            args = json.load(fh)
        points = np.ascontiguousarray(np.load(os.path.join(work, "points.npy")), dtype=np.float64)

        entry = os.path.join(candidate, "select_subset.py")
        import importlib.util

        spec = importlib.util.spec_from_file_location("candidate_select_subset", entry)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {entry}")
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, candidate)
        spec.loader.exec_module(module)
        fn = getattr(module, "select")

        t0 = time.monotonic()
        returned = fn(points, int(args["k"]), int(args["seed"]), float(args["time_budget_s"]))
        call_s = time.monotonic() - t0
        payload = {"ok": True, "call_s": call_s, "result": encode(returned)}
    except BaseException as exc:
        payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:400]}

    try:
        with open(out_path, "w") as fh:
            _dump(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
    except BaseException:
        _exit(3)
    _exit(0)


if __name__ == "__main__":
    main()
