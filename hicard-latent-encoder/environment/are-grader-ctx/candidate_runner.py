from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ENTRY_POINT = "encoder.py"
ENTRY_CLASS = "Encoder"


def main() -> int:
    cand_dir, inputs_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

    with np.load(inputs_path, allow_pickle=False) as z:
        X_train = z["X_train"].astype(np.float64)
        G_train = z["G_train"].astype(np.int64)
        X_test = z["X_test"].astype(np.float64)
        G_test = z["G_test"].astype(np.int64)
        n_categories = int(z["n_categories"])
        max_width = int(z["max_width"])

    entry = Path(cand_dir) / ENTRY_POINT
    sys.path.insert(0, str(cand_dir))
    spec = importlib.util.spec_from_file_location("submitted_encoder", entry)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {entry}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["submitted_encoder"] = module
    spec.loader.exec_module(module)

    encoder_cls = getattr(module, ENTRY_CLASS)
    encoder = encoder_cls(n_categories, max_width)
    encoder.fit(X_train, G_train)
    E_train = encoder.transform(X_train, G_train)
    E_test = encoder.transform(X_test, G_test)

    for label, E in (("E_tr", E_train), ("E_te", E_test)):
        if not isinstance(E, np.ndarray):
            raise TypeError(f"{label}: transform returned {type(E).__name__}, not numpy.ndarray")

    np.savez(out_path, E_tr=E_train, E_te=E_test)
    return 0


if __name__ == "__main__":
    sys.exit(main())
