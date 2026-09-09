#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

DEV_CSV = Path("/app/data/dev.csv")
FEATURES = [f"f{i}" for i in range(9)]
LABEL = "y"

N_TRAIN = 1_785
N_EVAL = 800
N_UNLABELED = 2_413


def _take(y: np.ndarray, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=seed)
    return next(sss.split(np.zeros(len(y)), y))


def make_local_panel(draw: int, dev_csv: Path = DEV_CSV) -> dict:
    """Build local draw number `draw`.  Deterministic in `draw`."""
    dev = pd.read_csv(dev_csv)
    y = dev[LABEL].to_numpy()

    tr_idx, rest = _take(y, N_TRAIN, 100_000 + draw)
    ev_rel, rest2 = _take(y[rest], N_EVAL, 200_000 + draw)
    ev_idx = rest[ev_rel]
    unl_rel, _ = _take(y[rest[rest2]], N_UNLABELED, 300_000 + draw)
    unl_idx = rest[rest2][unl_rel]

    train_X = dev.iloc[tr_idx][FEATURES].reset_index(drop=True)
    train_y = dev.iloc[tr_idx][LABEL].to_numpy(dtype=np.int8)
    eval_X = dev.iloc[ev_idx][FEATURES].reset_index(drop=True)
    eval_y = dev.iloc[ev_idx][LABEL].to_numpy(dtype=np.int8)

    pool_X = pd.concat(
        [dev.iloc[unl_idx][FEATURES], eval_X], ignore_index=True
    )
    rng = np.random.default_rng(400_000 + draw)
    pool_X = pool_X.iloc[rng.permutation(len(pool_X))].reset_index(drop=True)

    return {
        "train_X": train_X,
        "train_y": train_y,
        "pool_X": pool_X,
        "eval_X": eval_X,
        "eval_y": eval_y,
        "seed": draw,
    }


def main() -> int:
    n_draws = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    sys.path.insert(0, "/app/output")
    import solution

    aucs = []
    for draw in range(n_draws):
        p = make_local_panel(draw)
        scores = np.asarray(
            solution.fit_predict(
                p["train_X"], p["train_y"], p["pool_X"], p["eval_X"], p["seed"]
            ),
            dtype=np.float64,
        )
        if scores.shape != (N_EVAL,):
            raise SystemExit(f"draw {draw}: expected shape ({N_EVAL},), got {scores.shape}")
        auc = roc_auc_score(p["eval_y"], scores)
        aucs.append(auc)
        print(f"draw {draw:2d}  roc_auc={auc:.4f}", flush=True)
    print(f"mean over {n_draws} draws: {float(np.mean(aucs)):.4f} "
          f"(min {min(aucs):.4f}, max {max(aucs):.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
