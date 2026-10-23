import importlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

SANDBOX = Path(__file__).resolve().parent
TRAIN_SEED = 0
EVAL_NAMES = ("a", "b", "c")


def main():
    status = {"ok": False, "error": None, "seconds": {}}
    try:
        torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
        sys.path.append(str(SANDBOX / "code"))
        model = importlib.import_module("hash_model")

        train_features = np.load(SANDBOX / "inputs" / "train_features.npy")
        train_labels = np.load(SANDBOX / "inputs" / "train_labels.npy")

        t0 = time.monotonic()
        state = model.train(train_features, train_labels, TRAIN_SEED)
        status["seconds"]["train"] = time.monotonic() - t0

        for name in EVAL_NAMES:
            features = np.load(SANDBOX / "inputs" / f"eval_{name}.npy")
            t0 = time.monotonic()
            codes = model.encode(state, features)
            status["seconds"][f"encode_{name}"] = time.monotonic() - t0
            np.save(SANDBOX / "out" / f"codes_{name}.npy", np.asarray(codes))

        status["ok"] = True
    except BaseException as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"[:600]
        traceback.print_exc()
    finally:
        (SANDBOX / "out" / "status.json").write_text(json.dumps(status))


if __name__ == "__main__":
    main()
