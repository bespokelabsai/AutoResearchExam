import numpy as np


def impute(X_train: np.ndarray, X_eval: np.ndarray, seed: int) -> np.ndarray:
    column_mean = np.nanmean(X_train, axis=0)
    column_mean = np.where(np.isfinite(column_mean), column_mean, 0.0)

    filled = np.array(X_eval, dtype=np.float64, copy=True)
    missing = ~np.isfinite(filled)
    filled[missing] = np.take(column_mean, np.nonzero(missing)[1])
    return filled
