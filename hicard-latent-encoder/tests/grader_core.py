from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor

MAX_WIDTH = 8
CANDIDATE_TIMEOUT_SEC = 120.0
ENTRY_POINT = "encoder.py"
ENTRY_CLASS = "Encoder"

FOREST_KWARGS = {"n_estimators": 100, "random_state": 0, "n_jobs": 8}

M0 = 3.0
BASELINE_SCORE = 27.0


def graded_reward(metric: float, valid: bool) -> float:
    """rational_squash: 0 at the trivial baseline, 0.5 at the anchor, asymptotic to 1."""
    if not valid or metric is None or not math.isfinite(float(metric)):
        return 0.0
    x_ref = BASELINE_SCORE - M0
    assert x_ref > 0, "malformed task: the anchor does not beat the trivial baseline"
    u = max(0.0, float(metric) - M0) / x_ref
    return u / (1.0 + u)


def load_instance(path: str | Path) -> dict:
    """Load a sealed instance. `allow_pickle=False`: these are trusted files, but the
    loader stays strict so the same helper is safe on any .npz."""
    with np.load(str(path), allow_pickle=False) as z:
        inst = {
            "X": z["X"].astype(np.float64, copy=True),
            "G": z["G"].astype(np.int64, copy=True),
            "y": z["y"].astype(np.float64, copy=True),
            "train_idx": z["train_idx"].astype(np.int64, copy=True),
            "test_idx": z["test_idx"].astype(np.int64, copy=True),
            "n_categories": int(z["n_categories"]),
            "onehot_mse": float(z["onehot_mse"]),
        }
    inst["name"] = Path(path).stem
    return inst


def split_arrays(inst: dict) -> dict:
    tr, te = inst["train_idx"], inst["test_idx"]
    return {
        "X_train": np.ascontiguousarray(inst["X"][tr]),
        "G_train": np.ascontiguousarray(inst["G"][tr]),
        "y_train": np.ascontiguousarray(inst["y"][tr]),
        "X_test": np.ascontiguousarray(inst["X"][te]),
        "G_test": np.ascontiguousarray(inst["G"][te]),
        "y_test": np.ascontiguousarray(inst["y"][te]),
        "n_categories": inst["n_categories"],
    }


def _design(X: np.ndarray, E: np.ndarray | None) -> np.ndarray:
    Xf = np.ascontiguousarray(X, dtype=np.float32)
    if E is None or E.shape[1] == 0:
        return Xf
    return np.hstack([Xf, np.ascontiguousarray(E, dtype=np.float32)])


def forest_mse(X_train, E_train, y_train, X_test, E_test, y_test) -> float:
    """Held-out MSE of the fixed forest trained on [X | E]."""
    model = RandomForestRegressor(**FOREST_KWARGS)
    model.fit(_design(X_train, E_train), np.asarray(y_train, dtype=np.float64))
    pred = model.predict(_design(X_test, E_test))
    resid = np.asarray(y_test, dtype=np.float64) - np.asarray(pred, dtype=np.float64)
    return float(np.mean(resid * resid))


def onehot_matrix(G: np.ndarray, n_categories: int) -> np.ndarray:
    out = np.zeros((G.shape[0], n_categories), dtype=np.float32)
    out[np.arange(G.shape[0]), np.asarray(G, dtype=np.int64)] = 1.0
    return out


def onehot_reference_mse(parts: dict) -> float:
    """The reference encoding: the full one-hot indicator of the categorical column.

    Used to fill each instance's cached `onehot_mse` at build time with the same
    forest and the same design-matrix construction the graded run uses.
    """
    n_cat = parts["n_categories"]
    return forest_mse(
        parts["X_train"], onehot_matrix(parts["G_train"], n_cat), parts["y_train"],
        parts["X_test"], onehot_matrix(parts["G_test"], n_cat), parts["y_test"],
    )


def pct_reduction(mse_reference: float, mse_submission: float) -> float:
    return 100.0 * (mse_reference - mse_submission) / mse_reference


def validate_encoding(E, n_rows: int, label: str) -> tuple[bool, str]:
    if not isinstance(E, np.ndarray):
        return False, f"{label}: not a numpy array"
    if E.ndim != 2:
        return False, f"{label}: rank {E.ndim}, expected 2"
    if E.dtype != np.float64:
        return False, f"{label}: dtype {E.dtype}, expected float64"
    if E.shape[0] != n_rows:
        return False, f"{label}: {E.shape[0]} rows, expected {n_rows}"
    if not (0 <= E.shape[1] <= MAX_WIDTH):
        return False, f"{label}: width {E.shape[1]}, expected 0..{MAX_WIDTH}"
    if E.size and not np.all(np.isfinite(E)):
        return False, f"{label}: contains a non-finite entry"
    if E.size and float(np.max(np.abs(E))) > float(np.finfo(np.float32).max):
        return False, f"{label}: entry outside the float32 range"
    return True, "ok"


def load_candidate_output(path: str | Path, n_train: int, n_test: int):
    """Parse the privilege-dropped run's output as untrusted data."""
    p = Path(path)
    if not p.is_file():
        return None, None, "no output file"
    try:
        with np.load(str(p), allow_pickle=False) as z:
            if "E_tr" not in z.files or "E_te" not in z.files:
                return None, None, "output missing E_tr/E_te"
            E_tr = z["E_tr"]
            E_te = z["E_te"]
    except Exception as exc:
        return None, None, f"unreadable output: {type(exc).__name__}"
    ok, reason = validate_encoding(E_tr, n_train, "E_train")
    if not ok:
        return None, None, reason
    ok, reason = validate_encoding(E_te, n_test, "E_test")
    if not ok:
        return None, None, reason
    if E_tr.shape[1] != E_te.shape[1]:
        return None, None, "E_train and E_test have different widths"
    return E_tr, E_te, "ok"
