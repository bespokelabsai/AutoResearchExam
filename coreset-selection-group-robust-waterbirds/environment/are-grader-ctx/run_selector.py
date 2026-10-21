import importlib.util
import sys
from pathlib import Path

import numpy as np


def load_entry_point(cand_dir: str):
    path = Path(cand_dir) / "selector.py"
    if not path.is_file():
        print(f"no entry point at {path}", file=sys.stderr)
        return None
    sys.path.insert(0, cand_dir)
    spec = importlib.util.spec_from_file_location("candidate_selector", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["candidate_selector"] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    cand_dir, feat_p, lab_p, budget, seed, out_p = sys.argv[1:7]
    try:
        module = load_entry_point(cand_dir)
    except Exception as exc:
        print(f"import failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if module is None:
        return 2
    fn = getattr(module, "select_coreset", None)
    if not callable(fn):
        print("selector.py defines no callable select_coreset", file=sys.stderr)
        return 2

    features = np.load(feat_p)
    labels = np.load(lab_p)
    try:
        result = fn(features, labels, int(budget), int(seed))
    except Exception as exc:
        print(f"select_coreset raised: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    try:
        np.save(out_p, np.asarray(result))
    except Exception as exc:
        print(f"return value not saveable: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
