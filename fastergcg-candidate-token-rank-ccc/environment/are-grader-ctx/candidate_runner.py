import json
import os
import random
import sys
import time
import traceback

T0 = time.monotonic()

import numpy as np

BUDGET_SEC = 300.0
N_CANDIDATES = 64
EMBEDDING_PATH = "/opt/assets/embedding_matrix.npy"


def _fail(out_dir, reason, detail=""):
    with open(os.path.join(out_dir, "status.json"), "w") as fh:
        json.dump({"ok": False, "reason": reason, "detail": detail[-4000:],
                   "elapsed_sec": round(time.monotonic() - T0, 3)}, fh)
    sys.stderr.write(f"candidate_runner: {reason}\n{detail}\n")


def main() -> int:
    features_path, deliverable_dir, out_dir = sys.argv[1:4]
    os.makedirs(out_dir, exist_ok=True)

    random.seed(0)
    np.random.seed(0)

    try:
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
    except Exception:
        pass

    embedding = np.load(EMBEDDING_PATH)
    embedding.setflags(write=False)
    z = np.load(features_path)
    n_states = int(z["step"].shape[0])

    sys.path.insert(0, deliverable_dir)
    try:
        import ranker
        rank_fn = getattr(ranker, "rank")
        if not callable(rank_fn):
            raise TypeError("ranker.rank is not callable")
    except Exception:
        _fail(out_dir, "import_failed", traceback.format_exc())
        return 2

    identity = np.arange(N_CANDIDATES, dtype=np.float64)
    preds = np.zeros((n_states, N_CANDIDATES), dtype=np.float64)
    fallbacks, errors = [], []
    for i in range(n_states):
        if time.monotonic() - T0 > BUDGET_SEC:
            _fail(out_dir, "budget_exhausted",
                  f"after {i} of {n_states} calls")
            return 3
        features = {
            "grad_row": z["grad_row"][i],
            "candidate_ids": z["candidate_ids"][i],
            "current_token_id": int(z["current_token_id"][i]),
            "current_loss": float(z["current_loss"][i]),
            "prefix_ids": z["prefix_ids"][i][: int(z["prefix_len"][i])].tolist(),
            "post_ids": z["post_ids"][i][: int(z["post_len"][i])].tolist(),
            "target_ids": z["target_ids"][i][: int(z["target_len"][i])].tolist(),
            "step": int(z["step"][i]),
            "embedding_matrix": embedding,
        }
        try:
            out = rank_fn(features)
            arr = np.asarray(out)
            if arr.shape != (N_CANDIDATES,):
                raise ValueError(f"shape {arr.shape}, expected ({N_CANDIDATES},)")
            if arr.dtype.kind not in "fiub":
                raise TypeError(f"dtype {arr.dtype} is not a real number type")
            arr = arr.astype(np.float64)
            if not np.all(np.isfinite(arr)):
                raise ValueError("non-finite score")
            preds[i] = arr
        except Exception as exc:
            preds[i] = identity
            fallbacks.append(i)
            if len(errors) < 5:
                errors.append(f"snapshot {i}: {exc!r}")

    np.save(os.path.join(out_dir, "predictions.npy"), preds)
    with open(os.path.join(out_dir, "status.json"), "w") as fh:
        json.dump({"ok": True, "n_states": n_states,
                   "fallback_calls": len(fallbacks),
                   "fallback_indices": fallbacks[:80],
                   "first_errors": errors,
                   "elapsed_sec": round(time.monotonic() - T0, 3)}, fh)
    print(f"candidate_runner: {n_states} calls, {len(fallbacks)} fell back, "
          f"{time.monotonic() - T0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        try:
            _fail(sys.argv[3], "runner_crashed", traceback.format_exc())
        except Exception:
            traceback.print_exc()
        sys.exit(4)
