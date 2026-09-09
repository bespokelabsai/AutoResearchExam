from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cellspec

HIDDEN_ROOT = Path("/tests/hidden_data")

DELIVERABLE_DIR = Path("/app/output")
ENTRY_POINT = DELIVERABLE_DIR / "impute.py"
ENTRY_FUNCTION = "impute"
CELL_BUDGET_SEC = 30.0
MAX_DELIVERABLE_FILES = 2000
MAX_DELIVERABLE_BYTES = 512 * 1024 * 1024

M0 = 0.00
BASELINE_SCORE = 0.30


def graded_reward(metric: float) -> float:
    """rational_squash: 0 at the trivial baseline, 0.5 at the baseline score, no cutoff."""
    x_ref = BASELINE_SCORE - M0
    assert x_ref > 0, "malformed task: the baseline score does not beat the trivial baseline"
    if not np.isfinite(metric):
        return 0.0
    u = max(0.0, float(metric) - M0) / x_ref
    return u / (1.0 + u)


def load_manifest(half: str) -> dict:
    with open(HIDDEN_ROOT / half / "manifest.json") as fh:
        return json.load(fh)


def iter_cells(half: str):
    """Yield (cell_key, X_raw, seed, n_train, n_eval) for every graded cell of `half`."""
    manifest = load_manifest(half)
    for entry in manifest["datasets"]:
        with np.load(HIDDEN_ROOT / half / entry["file"], allow_pickle=False) as data:
            X = np.ascontiguousarray(data["X"], dtype=np.float64)
        for k, seed in enumerate(entry["seeds"]):
            yield (f"{entry['id']}:{k}", X, int(seed), entry["n_train"], entry["n_eval"])


def score_submission(X_pred_raw, X_eval_true, mask_eval):
    """Validate an untrusted returned array, then score it. Returns (score, reason).

    `score` is None when the array violates the contract. The reasons are deliberately
    coarse: they are aggregated into the public grader's stdout, and a reason carrying the
    expected shape would hand the agent the dimensions of a sealed dataset.
    """
    arr = X_pred_raw
    if not isinstance(arr, np.ndarray):
        return None, "not_an_array"
    if arr.dtype.kind != "f":
        return None, "wrong_dtype"
    if arr.shape != X_eval_true.shape:
        return None, "wrong_shape"
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        return None, "non_finite"
    return cellspec.imputation_r2(arr, X_eval_true, mask_eval), "ok"


def aggregate(cell_scores):
    """Unweighted mean over cells. An invalid cell contributes 0.0."""
    if not cell_scores:
        return 0.0
    return float(np.mean([0.0 if s is None else float(s) for s in cell_scores]))


def deliverable_files(root: Path):
    """Every regular, non-symlink file under `root`, deepest-first order irrelevant.

    Symlinks are neither followed nor copied at any depth: a submitted
    `weights.npy -> /tests/hidden_data/f1.npz` must never materialise anywhere the
    grader or the candidate process can read it.
    """
    kept, skipped_links = [], []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        for name in list(dirnames):
            if os.path.islink(os.path.join(dirpath, name)):
                dirnames.remove(name)
                skipped_links.append(os.path.join(dirpath, name))
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            if os.path.islink(full):
                skipped_links.append(full)
                continue
            if not os.path.isfile(full):
                continue
            kept.append(full)
    return kept, skipped_links
