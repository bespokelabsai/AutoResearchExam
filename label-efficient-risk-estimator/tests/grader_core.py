import hashlib
import math

import numpy as np

BUDGETS = (50, 100, 200, 400)
N_SEEDS = 4000
PER_SEED_TIMEOUT_S = 5.0
PER_CELL_TIMEOUT_S = 900.0
SE_CAP = 100.0
RATIO_CAP = 1.0e4

M0 = 1.0
BASELINE_SCORE = 0.68
HIGHER_IS_BETTER = False

_SALT = "2508-09093/label-efficient-risk-estimator/v1"


def cell_seed(instance_id: str, budget: int) -> int:
    """Deterministic per-cell seed. Identical on both sides of the socket."""
    h = hashlib.sha256(f"{_SALT}|{instance_id}|{budget}".encode()).digest()
    return int.from_bytes(h[:8], "big")


def seed_permutation(cseed: int, seed: int, n_pool: int) -> np.ndarray:
    """Row order the candidate sees for this (cell, seed). Defeats any {row -> label} cache."""
    return np.random.default_rng([cseed, seed, 0x5EED]).permutation(n_pool)


def exact_risk(target_probs: np.ndarray, labels: np.ndarray) -> float:
    """R = (1/N) sum_i -log target_probs[i, y_i] over the WHOLE pool."""
    p = target_probs[np.arange(labels.shape[0]), labels]
    return float(np.mean(-np.log(np.clip(p, 1e-300, None))))


def reference_squared_errors(target_probs, labels, budget, cseed, seeds):
    """Uniform-random acquisition + plain sample mean, on the same seeds and the same
    permuted view the candidate is given. This is the metric's denominator."""
    n_pool = labels.shape[0]
    risk = exact_risk(target_probs, labels)
    nll = -np.log(np.clip(target_probs[np.arange(n_pool), labels], 1e-300, None))
    out = np.empty(len(seeds), dtype=np.float64)
    for k, s in enumerate(seeds):
        perm = seed_permutation(cseed, s, n_pool)
        nll_perm = nll[perm]
        rng = np.random.default_rng([cseed, s])
        idx = rng.choice(n_pool, size=budget, replace=False)
        out[k] = (float(np.mean(nll_perm[idx])) - risk) ** 2
    return out


def cell_ratio(agent_se: np.ndarray, ref_se: np.ndarray) -> float:
    """median-over-seeds squared error of the candidate, divided by the reference's."""
    num = float(np.median(agent_se))
    den = float(np.median(ref_se))
    if not math.isfinite(den) or den <= 0.0:
        den = 1e-12
    return float(min(RATIO_CAP, num / den))


def aggregate(cell_ratios) -> float:
    """The graded metric: the median of the per-cell ratios."""
    vals = np.asarray(list(cell_ratios), dtype=np.float64)
    if vals.size == 0 or not np.all(np.isfinite(vals)):
        return float(RATIO_CAP)
    return float(np.median(vals))


def graded_reward(metric: float, valid: bool) -> float:
    """rational_squash on two task-derived anchors. Clamped again in compute_reward.py."""
    if not valid or metric is None or not math.isfinite(metric):
        return 0.0
    sigma = 1.0 if HIGHER_IS_BETTER else -1.0
    x = sigma * (metric - M0)
    x_ref = sigma * (BASELINE_SCORE - M0)
    assert x_ref > 0, "malformed task: the anchor does not beat the trivial baseline"
    u = max(0.0, x) / x_ref
    return u / (1.0 + u)
