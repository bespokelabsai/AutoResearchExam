from __future__ import annotations

import importlib.util
import json
import os
import random
import sys
import time

import numpy as np

N_TEST = 75
STATUS_OK, STATUS_RAISED, STATUS_BAD_SHAPE, STATUS_NON_FINITE = 0, 1, 2, 3


def _pin_determinism() -> None:
    """Mirror the OMP/MKL/OPENBLAS pins the launcher's env dict already sets: torch has its
    own thread pool that a thread-count env var alone does not reliably bind, and any candidate
    that draws from the global RNGs without its own seed needs a fixed starting state."""
    random.seed(0)
    np.random.seed(0)
    try:
        import torch
        torch.manual_seed(0)
        torch.set_num_threads(1)
    except ImportError:
        pass


def main() -> int:
    _pin_determinism()
    inputs_path, solution_dir, out_path, meta_path = sys.argv[1:5]
    meta = {"import_ok": False, "error": None, "n_calls": 0, "elapsed_sec": 0.0}

    def dump_meta():
        with open(meta_path, "w") as fh:
            json.dump(meta, fh)

    data = np.load(inputs_path, allow_pickle=False)
    n = int(data["n_realizations"])
    preds = np.zeros((n, N_TEST), dtype=np.float64)
    status = np.full(n, STATUS_RAISED, dtype=np.int64)

    entry_point = os.path.join(solution_dir, "estimator.py")
    sys.path.insert(0, solution_dir)
    try:
        spec = importlib.util.spec_from_file_location("candidate_estimator", entry_point)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fn = getattr(module, "estimate_cate")
        if not callable(fn):
            raise TypeError("estimate_cate is not callable")
        meta["import_ok"] = True
    except BaseException as exc:
        meta["error"] = f"{type(exc).__name__}: {exc}"[:500]
        dump_meta()
        np.savez(out_path, preds=preds, status=status)
        return 0

    t0 = time.monotonic()
    for k in range(n):
        X_train = np.array(data[f"X_train_{k:03d}"], dtype=np.float64)
        t_train = np.array(data[f"t_train_{k:03d}"], dtype=np.float64)
        y_train = np.array(data[f"y_train_{k:03d}"], dtype=np.float64)
        X_test = np.array(data[f"X_test_{k:03d}"], dtype=np.float64)
        try:
            out = fn(X_train, t_train, y_train, X_test)
            arr = np.asarray(out, dtype=np.float64)
            if arr.shape != (X_test.shape[0],):
                status[k] = STATUS_BAD_SHAPE
                continue
            if not np.all(np.isfinite(arr)):
                status[k] = STATUS_NON_FINITE
                continue
            preds[k, : arr.shape[0]] = arr
            status[k] = STATUS_OK
        except BaseException as exc:
            status[k] = STATUS_RAISED
            if meta["error"] is None:
                meta["error"] = f"call {k}: {type(exc).__name__}: {exc}"[:500]
        finally:
            meta["n_calls"] = k + 1

    meta["elapsed_sec"] = time.monotonic() - t0
    np.savez(out_path, preds=preds, status=status)
    dump_meta()
    return 0


if __name__ == "__main__":
    sys.exit(main())
