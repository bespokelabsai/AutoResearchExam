from __future__ import annotations

import numpy as np

BUDGET = 240
PER_CLASS_QUOTA = 120
FEATURE_DIM = 512
N_CLASSES = 2
SELECTOR_TIMEOUT_SEC = 150
PROBE_SEEDS = (11, 22, 33, 44, 55)
SELECTOR_SEEDS = (10007, 20011, 30013, 41011, 51001, 61001, 71003)

PROBE_EPOCHS = 100
PROBE_BATCH = 32
PROBE_LR = 0.05
PROBE_MOMENTUM = 0.9
PROBE_WEIGHT_DECAY = 1e-4
PROBE_INIT_SCALE = 0.01
STANDARDIZE_EPS = 1e-6


M0 = 59.0
BASELINE_SCORE = 68.5


def graded_reward(metric: float) -> float:
    """rational_squash: 0 at the trivial baseline, 0.5 at the source anchor, no cutoff above."""
    x_ref = BASELINE_SCORE - M0
    assert x_ref > 0, "malformed task: the anchor does not beat the trivial baseline"
    u = max(0.0, float(metric) - M0) / x_ref
    return u / (1.0 + u)


class SelectionError(ValueError):
    """The candidate's return value violates the published selection contract."""


def validate_selection(raw, n_pool: int, pool_labels: np.ndarray) -> np.ndarray:
    """Check an untrusted selector output against the published contract.

    Returns the validated int64 index array, or raises SelectionError. Every rejection here is
    a reward-0 outcome for the whole grade, exactly as instruction.md states.
    """
    if not isinstance(raw, np.ndarray):
        raise SelectionError(f"selection is {type(raw).__name__}, expected numpy.ndarray")
    if raw.dtype != np.int64:
        raise SelectionError(f"selection dtype is {raw.dtype}, expected int64")
    if raw.ndim != 1 or raw.shape[0] != BUDGET:
        raise SelectionError(f"selection shape is {raw.shape}, expected ({BUDGET},)")
    if np.unique(raw).size != BUDGET:
        raise SelectionError("selection contains duplicate indices")
    if raw.min() < 0 or raw.max() >= n_pool:
        raise SelectionError(f"selection indices outside [0, {n_pool})")
    picked = pool_labels[raw]
    for cls in range(N_CLASSES):
        got = int((picked == cls).sum())
        if got != PER_CLASS_QUOTA:
            raise SelectionError(
                f"selection has {got} indices with label {cls}, expected {PER_CLASS_QUOTA}")
    return raw.astype(np.int64, copy=True)


def _standardize_fit(x: np.ndarray):
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.maximum(sd, STANDARDIZE_EPS)
    return mu, sd


def train_probe(x: np.ndarray, y: np.ndarray, seed: int):
    """Pinned multinomial logistic probe: mini-batch SGD with momentum on standardized features.

    Standardization statistics come from the SELECTED rows only, so no evaluation-split
    information enters training.
    """
    mu, sd = _standardize_fit(x)
    xs = ((x - mu) / sd).astype(np.float64)
    n, d = xs.shape
    onehot = np.zeros((n, N_CLASSES), dtype=np.float64)
    onehot[np.arange(n), y] = 1.0

    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, PROBE_INIT_SCALE, size=(d, N_CLASSES))
    b = np.zeros(N_CLASSES, dtype=np.float64)
    vw = np.zeros_like(w)
    vb = np.zeros_like(b)

    for _ in range(PROBE_EPOCHS):
        order = rng.permutation(n)
        for start in range(0, n, PROBE_BATCH):
            idx = order[start:start + PROBE_BATCH]
            xb, tb = xs[idx], onehot[idx]
            logits = xb @ w + b
            logits -= logits.max(axis=1, keepdims=True)
            expl = np.exp(logits)
            probs = expl / expl.sum(axis=1, keepdims=True)
            diff = (probs - tb) / idx.size
            gw = xb.T @ diff + PROBE_WEIGHT_DECAY * w
            gb = diff.sum(axis=0)
            vw = PROBE_MOMENTUM * vw + gw
            vb = PROBE_MOMENTUM * vb + gb
            w -= PROBE_LR * vw
            b -= PROBE_LR * vb
    return {"mu": mu, "sd": sd, "w": w, "b": b}


def predict(probe, x: np.ndarray) -> np.ndarray:
    xs = (x - probe["mu"]) / probe["sd"]
    return np.argmax(xs @ probe["w"] + probe["b"], axis=1)


def group_accuracies(probe, x: np.ndarray, y: np.ndarray, a: np.ndarray) -> dict:
    """Per-(label, attribute) accuracy in percent, plus the worst of the four."""
    pred = predict(probe, x)
    out = {}
    for lab in range(N_CLASSES):
        for att in (0, 1):
            mask = (y == lab) & (a == att)
            out[f"y{lab}_a{att}"] = 100.0 * float((pred[mask] == y[mask]).mean())
    out["overall"] = 100.0 * float((pred == y).mean())
    out["worst_group"] = min(out[f"y{lab}_a{att}"]
                             for lab in range(N_CLASSES) for att in (0, 1))
    return out


def score_selection(pool_feat: np.ndarray, pool_labels: np.ndarray, selection: np.ndarray,
                    eval_feat: np.ndarray, eval_labels: np.ndarray, eval_att: np.ndarray) -> list:
    """Train the pinned probe on the selected rows once per probe seed and evaluate each."""
    x = pool_feat[selection]
    y = pool_labels[selection]
    runs = []
    for seed in PROBE_SEEDS:
        probe = train_probe(x, y, seed)
        stats = group_accuracies(probe, eval_feat, eval_labels, eval_att)
        stats["probe_seed"] = seed
        runs.append(stats)
    return runs
