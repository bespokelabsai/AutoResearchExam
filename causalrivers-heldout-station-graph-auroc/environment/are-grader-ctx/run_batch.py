import importlib.util
import json
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

K_NODES = 5
IMPORT_FAILED = 3
NO_INPUT = 4

torch.set_num_threads(1)


def load_candidate(module_path):
    sys.path.insert(0, str(Path(module_path).resolve().parent))
    spec = importlib.util.spec_from_file_location("candidate_predict", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["candidate_predict"] = module
    spec.loader.exec_module(module)
    fn = getattr(module, "score_edges", None)
    if not callable(fn):
        raise AttributeError("module defines no callable score_edges")
    return fn


def main():
    spec_path, in_path, out_path, status_path = sys.argv[1:5]
    status = {"import_ok": False, "calls": [], "error": ""}

    def write_status():
        Path(status_path).write_text(json.dumps(status))

    try:
        spec = json.loads(Path(spec_path).read_text())
        series = np.load(in_path, mmap_mode="r")
    except Exception:
        status["error"] = traceback.format_exc(limit=3)[-2000:]
        write_status()
        return NO_INPUT

    seed = int(spec["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    n = int(series.shape[0])
    try:
        score_edges = load_candidate(spec["module"])
    except BaseException:
        status["error"] = traceback.format_exc(limit=6)[-4000:]
        write_status()
        return IMPORT_FAILED
    status["import_ok"] = True

    freq = int(spec["freq_minutes"])
    out = np.full((n, K_NODES, K_NODES), np.nan, dtype=np.float64)
    for i in range(n):
        window = np.ascontiguousarray(series[i])
        try:
            matrix = score_edges(window, freq)
            arr = np.asarray(matrix, dtype=np.float64)
            if arr.shape != (K_NODES, K_NODES):
                raise ValueError(f"score_edges returned shape {arr.shape}, expected (5, 5)")
            out[i] = arr
            status["calls"].append("ok")
        except BaseException:
            status["calls"].append("error")
            if len(status["error"]) < 4000:
                status["error"] += traceback.format_exc(limit=3)[-1000:]
    np.save(out_path, out)
    write_status()
    return 0


if __name__ == "__main__":
    sys.exit(main())
