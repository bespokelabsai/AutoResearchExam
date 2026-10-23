from __future__ import annotations

import numpy as np
from sklearn.preprocessing import QuantileTransformer

MISSING_RATE = 0.50

QT_N_QUANTILES = 1000
QT_SUBSAMPLE = 100_000
QT_RANDOM_STATE = 0


def build_cell(X, seed, n_train, n_eval):
    """Build one cell from the raw feature matrix `X` (float64, no missing values).

    Rows are permuted with `seed` and then *partitioned*: the first `n_train` rows of the
    permutation become the training rows and the next `n_eval` become the evaluation rows,
    so the two are disjoint by construction. Independent MCAR masks at MISSING_RATE are
    drawn for each matrix from the same generator. A QuantileTransformer with a normal
    output distribution is fitted on the *incomplete* training matrix and applied to both.

    Returns (X_train, X_eval, X_eval_true, mask_eval):
      X_train     (n_train, d) float64, gaussianized, np.nan at masked entries
      X_eval      (n_eval, d)  float64, gaussianized, np.nan at masked entries
      X_eval_true (n_eval, d)  float64, gaussianized, complete -- the ground truth
      mask_eval   (n_eval, d)  bool, True where X_eval is missing
    """
    X = np.asarray(X, dtype=np.float64)
    n_rows = X.shape[0]
    if n_train + n_eval > n_rows:
        raise ValueError(f"cell needs {n_train + n_eval} rows, matrix has {n_rows}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_rows)
    rows_train = perm[:n_train]
    rows_eval = perm[n_train : n_train + n_eval]

    raw_train = X[rows_train]
    raw_eval = X[rows_eval]

    mask_train = rng.random(raw_train.shape) < MISSING_RATE
    mask_eval = rng.random(raw_eval.shape) < MISSING_RATE

    incomplete_train = raw_train.copy()
    incomplete_train[mask_train] = np.nan

    qt = QuantileTransformer(
        output_distribution="normal",
        n_quantiles=min(QT_N_QUANTILES, n_train),
        subsample=QT_SUBSAMPLE,
        random_state=QT_RANDOM_STATE,
        copy=True,
    )
    qt.fit(incomplete_train)

    X_train = np.ascontiguousarray(qt.transform(incomplete_train), dtype=np.float64)
    X_eval_true = np.ascontiguousarray(qt.transform(raw_eval), dtype=np.float64)
    X_eval = X_eval_true.copy()
    X_eval[mask_eval] = np.nan

    return X_train, X_eval, X_eval_true, mask_eval


def imputation_r2(X_pred, X_true, mask):
    """Per-feature R^2 over masked entries, averaged unweighted over features.

    For feature j, over the entries where `mask` is True:
        R2_j = 1 - sum((true - pred)^2) / sum((true - mean(true))^2)
    Features with fewer than two masked entries, or with a degenerate ground-truth
    variance over those entries, are skipped. Returns the mean of the kept features.
    """
    X_pred = np.asarray(X_pred, dtype=np.float64)
    X_true = np.asarray(X_true, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)

    scores = []
    for j in range(X_true.shape[1]):
        sel = mask[:, j]
        if int(sel.sum()) < 2:
            continue
        truth = X_true[sel, j]
        pred = X_pred[sel, j]
        ss_tot = float(np.sum((truth - truth.mean()) ** 2))
        if not np.isfinite(ss_tot) or ss_tot <= 1e-12:
            continue
        ss_res = float(np.sum((truth - pred) ** 2))
        scores.append(1.0 - ss_res / ss_tot)

    if not scores:
        raise ValueError("no scorable feature in this cell")
    return float(np.mean(scores))
