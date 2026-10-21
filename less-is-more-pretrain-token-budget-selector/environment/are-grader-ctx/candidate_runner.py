import importlib.util
import json
import random
import sys
import traceback

import numpy as np
import torch


def _transport(item):
    """ints travel as ints; anything else travels as a marker the grader drops."""
    if isinstance(item, bool):
        return {"__bad__": "bool"}
    if isinstance(item, int):
        return int(item)
    typ = type(item).__name__
    try:
        return int(item.__index__())
    except Exception:
        return {"__bad__": typ}


def main() -> int:
    deliverable_dir, pool_dir, budget_s, workdir, out_path = sys.argv[1:6]
    result = {"ok": False, "error": None, "value": None, "value_type": None}
    try:
        spec = importlib.util.spec_from_file_location(
            "candidate_select", f"{deliverable_dir}/select.py")
        if spec is None or spec.loader is None:
            raise ImportError("select.py is not importable")
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        module = importlib.util.module_from_spec(spec)
        sys.modules["candidate_select"] = module
        sys.path.insert(0, deliverable_dir)
        spec.loader.exec_module(module)
        fn = getattr(module, "select", None)
        if fn is None or not callable(fn):
            raise AttributeError("select.py defines no callable named 'select'")
        value = fn(pool_dir, int(budget_s), workdir)
        result["value_type"] = type(value).__name__
        if isinstance(value, tuple):
            value = list(value)
        if not isinstance(value, list):
            raise TypeError(f"select() returned {type(value).__name__}, expected list")
        result["value"] = [_transport(v) for v in value]
        result["ok"] = True
    except BaseException:
        result["error"] = traceback.format_exc()[-4000:]
    try:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh)
    except Exception:
        return 4
    return 0 if result["ok"] else 3


if __name__ == "__main__":
    sys.exit(main())
