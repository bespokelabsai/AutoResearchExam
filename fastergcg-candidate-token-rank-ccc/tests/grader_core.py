from __future__ import annotations

import numpy as np

N_CANDIDATES = 64

DEGENERATE_REWARD = 0.10
HIGHER_IS_BETTER = True

ANCHORS = {
    "final": {"m_floor": 0.06948660714285715,
              "m0": 0.1397647664835165,
              "baseline_score": 0.55},
    "intermediate": {"m_floor": 0.12324805402930404,
                     "m0": 0.13551682692307693,
                     "baseline_score": 0.5702},
}


def ordinal_ranks(values: np.ndarray) -> np.ndarray:
    """1..n ascending ranks; ties broken by ascending array position."""
    v = np.asarray(values, dtype=np.float64)
    order = np.argsort(v, kind="stable")
    out = np.empty(v.shape[0], dtype=np.float64)
    out[order] = np.arange(1, v.shape[0] + 1, dtype=np.float64)
    return out


def ccc(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's concordance correlation coefficient, population moments."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    cov = float(np.mean((x - x.mean()) * (y - y.mean())))
    denom = float(x.var() + y.var() + (x.mean() - y.mean()) ** 2)
    if denom <= 0.0:
        return 0.0
    return 2.0 * cov / denom


def pooled_ccc(pred: np.ndarray, truth: np.ndarray) -> float:
    """CCC over the pooled per-snapshot rank vectors of pred against truth."""
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if pred.shape != truth.shape or pred.ndim != 2 \
            or pred.shape[1] != N_CANDIDATES:
        raise ValueError(f"bad prediction shape {pred.shape} vs {truth.shape}")
    xs = np.concatenate([ordinal_ranks(row) for row in pred])
    ys = np.concatenate([ordinal_ranks(row) for row in truth])
    return ccc(xs, ys)


def assert_disjoint_behaviours(keys_a, keys_b) -> None:
    """Raise if two panels share a behaviour key (grouped-leakage guard)."""
    shared = sorted(set(keys_a) & set(keys_b))
    if shared:
        raise ValueError(f"panels share behaviour keys: {shared}")


def graded_reward(metric: float, panel: str = "final", *,
                  c: float = DEGENERATE_REWARD,
                  higher_is_better: bool = HIGHER_IS_BETTER) -> float:
    """rational_squash with the sub-m0 degenerate band. Bounded in [0, 1]."""
    anchors = ANCHORS[panel]
    m0, baseline_score, m_floor = anchors["m0"], anchors["baseline_score"], anchors["m_floor"]
    sigma = 1.0 if higher_is_better else -1.0
    x = sigma * (metric - m0)
    x_ref = sigma * (baseline_score - m0)
    if not x_ref > 0:
        raise ValueError("malformed task: the reference does not beat m0")

    def two_anchor() -> float:
        u = max(0.0, x) / x_ref
        return u / (1.0 + u)

    if c <= 0.0:
        return two_anchor()
    g = sigma * (m0 - m_floor)
    if g <= 0:
        return two_anchor()
    if x < 0:
        v = max(0.0, sigma * (metric - m_floor)) / g
        return c * v
    u = x / x_ref
    return c + (1.0 - c) * (u / (1.0 + u))
