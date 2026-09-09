import numpy as np

CODE_BITS = 16
TOP_K = 5000
QUERY_CHUNK = 250

M0 = 0.435
BASELINE_SCORE = 0.7903


class InvalidSubmission(Exception):
    """The deliverable produced nothing gradeable. Routes to reward exactly 0."""


def validate_codes(array, n_rows, what):
    """Enforce the code contract stated in instruction.md: an (n_rows, 16) int8 array whose
    every element is -1 or +1. Anything else is an invalid submission, not a low score."""
    if not isinstance(array, np.ndarray):
        raise InvalidSubmission(f"{what}: not an array")
    if array.shape != (n_rows, CODE_BITS):
        raise InvalidSubmission(f"{what}: shape {array.shape}, expected {(n_rows, CODE_BITS)}")
    if array.dtype != np.int8:
        raise InvalidSubmission(f"{what}: dtype {array.dtype}, expected int8")
    bad = np.setdiff1d(np.unique(array), np.array([-1, 1], dtype=np.int8))
    if bad.size:
        raise InvalidSubmission(f"{what}: values outside {{-1, +1}}: {bad[:4].tolist()}")
    return array


def map_at_k(query_codes, db_codes, query_labels, db_labels, top_k=TOP_K):
    """mAP@top_k. Returns (mAP, diagnostics)."""
    q = query_codes.astype(np.float32)
    d = db_codes.astype(np.float32)
    ql = query_labels.astype(np.float32)
    dl = db_labels.astype(np.float32)
    n_q, n_db = q.shape[0], d.shape[0]
    k = min(top_k, n_db)
    ranks = np.arange(1, k + 1, dtype=np.float64)

    aps = np.empty(n_q, dtype=np.float64)
    rel_frac = np.empty(n_q, dtype=np.float64)
    for start in range(0, n_q, QUERY_CHUNK):
        stop = min(start + QUERY_CHUNK, n_q)
        dist = ((CODE_BITS - q[start:stop] @ d.T) / 2).astype(np.int16)
        relevant = (ql[start:stop] @ dl.T) > 0
        order = np.argsort(dist, axis=1, kind="stable")[:, :k]
        rel = np.take_along_axis(relevant, order, axis=1)
        hits = np.cumsum(rel, axis=1, dtype=np.float64)
        total = hits[:, -1]
        prec = hits / ranks
        with np.errstate(invalid="ignore", divide="ignore"):
            ap = np.where(total > 0, (prec * rel).sum(axis=1) / np.maximum(total, 1.0), 0.0)
        aps[start:stop] = ap
        rel_frac[start:stop] = relevant.mean(axis=1)

    diagnostics = {
        "n_query": int(n_q),
        "n_database": int(n_db),
        "top_k": int(k),
        "ap_mean": float(aps.mean()),
        "ap_median": float(np.median(aps)),
        "ap_p10": float(np.quantile(aps, 0.10)),
        "ap_p90": float(np.quantile(aps, 0.90)),
        "query_relevant_fraction_mean": float(rel_frac.mean()),
        "distinct_query_codes": int(np.unique(query_codes, axis=0).shape[0]),
        "distinct_db_codes": int(np.unique(db_codes, axis=0).shape[0]),
        "db_bit_imbalance_max": float(np.abs(db_codes.astype(np.float64).mean(axis=0)).max()),
    }
    return float(aps.mean()), diagnostics


def graded_reward(metric, valid=True):
    """rational_squash, monotone in the metric, no cutoff, bounded in [0, 1)."""
    if not valid or metric is None or not np.isfinite(metric):
        return 0.0
    x_ref = BASELINE_SCORE - M0
    assert x_ref > 0, "malformed task: the reference result does not beat the trivial baseline"
    u = max(0.0, float(metric) - M0) / x_ref
    return u / (1.0 + u)
