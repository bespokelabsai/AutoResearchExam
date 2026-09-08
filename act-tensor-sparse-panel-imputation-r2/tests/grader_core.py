import math

import numpy as np

VALUE_LO = -1.0
VALUE_HI = 1.0
PER_INSTANCE_TIMEOUT_SEC = 240.0
MAX_OUTPUT_BYTES = 64_000_000

M0 = 0.0
BASELINE_SCORE = 0.7047
HIGHER_IS_BETTER = True


def r2_imp(pred_at_eval, truth_at_eval):
    """Paper Eq. 19: R^2_imp pooled over one instance's held-out cells.

    1 - sum_m (x_m - xtilde_m)^2 / sum_m (x_m - xbar)^2, with xbar the mean of the
    held-out TRUE values.
    """
    truth = np.asarray(truth_at_eval, dtype=np.float64)
    pred = np.asarray(pred_at_eval, dtype=np.float64)
    denom = float(np.sum((truth - truth.mean()) ** 2))
    if not math.isfinite(denom) or denom <= 0.0:
        raise ValueError("degenerate held-out set: zero variance")
    resid = float(np.sum((truth - pred) ** 2))
    return 1.0 - resid / denom


def validate_output(arr, expected_shape):
    """Return (ok, reason).  Mirrors the contract stated in instruction.md."""
    if arr is None:
        return False, "no output produced"
    if not isinstance(arr, np.ndarray):
        return False, f"output is {type(arr).__name__}, not a numpy array"
    if arr.dtype.kind != "f":
        return False, f"output dtype {arr.dtype} is not a real floating dtype"
    if arr.shape != expected_shape:
        return False, f"output shape {arr.shape} != {expected_shape}"
    values = arr.astype(np.float64, copy=False)
    if not np.all(np.isfinite(values)):
        return False, "output contains NaN or inf"
    if float(values.min()) < VALUE_LO or float(values.max()) > VALUE_HI:
        return False, f"output outside [{VALUE_LO}, {VALUE_HI}]"
    return True, ""


def graded_reward(metric, m0=M0, baseline_score=BASELINE_SCORE,
                  higher_is_better=HIGHER_IS_BETTER,
                  valid=True):
    """rational_squash: 0 at the trivial baseline, 0.5 at the baseline score, asymptotic
    to 1, monotone, never negative, with no cutoff."""
    if not valid or metric is None or not math.isfinite(metric):
        return 0.0
    sigma = 1.0 if higher_is_better else -1.0
    x = sigma * (metric - m0)
    x_ref = sigma * (baseline_score - m0)
    assert x_ref > 0, "malformed task: the baseline score does not beat the trivial baseline"
    u = max(0.0, x) / x_ref
    return u / (1.0 + u)
