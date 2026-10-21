import numpy as np

DIM = 3


def star_discrepancy_3d(pts: np.ndarray) -> float:
    """Exact L_inf star discrepancy of a point set in [0,1]^3."""
    pts = np.ascontiguousarray(np.asarray(pts, dtype=np.float64))
    if pts.ndim != 2 or pts.shape[1] != DIM or pts.shape[0] == 0:
        raise ValueError("expected a non-empty (k, 3) array")
    if not np.isfinite(pts).all() or pts.min() < 0.0 or pts.max() > 1.0:
        raise ValueError("points must lie in [0,1]^3")
    k = pts.shape[0]

    grids, idx = [], []
    for d in range(DIM):
        g = np.unique(np.concatenate([pts[:, d], np.array([1.0])]))
        grids.append(g)
        idx.append(np.searchsorted(g, pts[:, d]))

    shape = tuple(len(g) for g in grids)
    hist = np.zeros(shape, dtype=np.int64)
    np.add.at(hist, (idx[0], idx[1], idx[2]), 1)
    closed = hist.cumsum(0).cumsum(1).cumsum(2)
    op = np.zeros(shape, dtype=np.int64)
    op[1:, 1:, 1:] = closed[:-1, :-1, :-1]

    vol = grids[0][:, None, None] * grids[1][None, :, None] * grids[2][None, None, :]
    over = closed / k - vol
    under = vol - op / k
    return float(max(over.max(), under.max()))


def baseline_indices(seed: int, n: int, k: int) -> np.ndarray:
    """The trivial baseline: a uniformly random k-subset, ignoring the coordinates."""
    return np.random.default_rng(int(seed)).choice(int(n), int(k), replace=False)


def validate_indices(raw, n: int, k: int) -> np.ndarray:
    """Turn an untrusted decoded JSON value into k distinct indices, or raise.

    Accepts a flat sequence of k integral values (ints, or floats that are exactly
    integral). Anything else - wrong length, duplicates, out-of-range, non-integral,
    booleans, nested structures - is rejected.
    """
    if not isinstance(raw, list):
        raise ValueError(f"result is {type(raw).__name__}, expected a sequence")
    if len(raw) != k:
        raise ValueError(f"returned {len(raw)} indices, expected exactly {k}")
    out = np.empty(k, dtype=np.int64)
    for i, v in enumerate(raw):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"index {i} is {type(v).__name__}, expected an integer")
        if isinstance(v, float):
            if not np.isfinite(v) or float(v) != int(v):
                raise ValueError(f"index {i} = {v!r} is not integral")
            v = int(v)
        if v < 0 or v >= n:
            raise ValueError(f"index {i} = {v} is outside [0, {n})")
        out[i] = v
    if len(np.unique(out)) != k:
        raise ValueError("returned indices are not distinct")
    return out


def graded_reward(metric: float, m0: float) -> float:
    """single_anchor_ratio, lower-is-better: rho = m0 / m, V = max(0, 1 - 1/rho).

    m0 is the panel mean of the trivial random-subset baseline, recomputed from the
    sealed instances at grade time. A submission at or worse than the baseline scores
    exactly 0; there is no cutoff above it. Neither anchor nor this arithmetic appears
    in instruction.md.
    """
    if not np.isfinite(metric) or not np.isfinite(m0) or m0 <= 0.0 or metric <= 0.0:
        return 0.0
    value = 1.0 - (float(metric) / float(m0))
    return float(value) if value > 1e-9 else 0.0
