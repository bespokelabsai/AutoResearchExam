import json
import math
from pathlib import Path

import numpy as np

K_NODES = 5
N_PAIRS = K_NODES * (K_NODES - 1)

M0 = 0.740
BASELINE_SCORE = 0.80

HIDDEN = Path(__file__).resolve().parent / "hidden_data"


def load_half(half):
    """Return the panel of one graded half: batches, sealed edge set, series location.

    The series matrix is station-major, shape (station_count, T) float32, so that slicing the
    five stations of a sample is a contiguous read.
    """
    if half not in ("intermediate", "final"):
        raise ValueError(f"unknown grading half: {half!r}")
    panel = json.loads((HIDDEN / half / "panel.json").read_text())
    batches = [[[int(c) for c in sample] for sample in batch] for batch in panel["batches"]]
    n = sum(len(b) for b in batches)
    if n != int(panel["n_samples"]):
        raise ValueError("panel sample count does not match its batches")
    for batch in batches:
        seen = set()
        for sample in batch:
            if len(sample) != K_NODES:
                raise ValueError("a panel sample does not have 5 stations")
            if seen & set(sample):
                raise ValueError("stations repeat inside a batch")
            seen |= set(sample)
    station_count = int(panel["station_count"])
    if max(c for batch in batches for sample in batch for c in sample) >= station_count:
        raise ValueError("a panel sample references a station outside the sealed matrix")
    edges = {(int(u), int(v)) for u, v in panel["edges"]}
    for batch in batches:
        for sample in batch:
            positives = int(sample_labels(sample, edges).sum())
            if not 0 < positives < N_PAIRS:
                raise ValueError(f"panel sample has {positives} of {N_PAIRS} positive pairs, "
                                 "so its AUROC is undefined")
    return {"batches": batches, "edges": edges, "n_samples": n,
            "station_count": station_count,
            "series_path": HIDDEN / "sealed_series.npy"}


def average_ranks(values):
    """Ranks 1..n with ties averaged (the tie handling AUROC needs)."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    ranks = np.empty(values.shape[0], dtype=np.float64)
    i = 0
    while i < sorted_vals.shape[0]:
        j = i
        while j + 1 < sorted_vals.shape[0] and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def auroc(scores, labels):
    """Tie-aware AUROC (Mann-Whitney U). NaN when one class is absent."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    n_pos = int(labels.sum())
    n_neg = int(labels.shape[0] - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = average_ranks(scores)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def sample_labels(sample, edges):
    """Off-diagonal edge indicators for one subgraph, in presentation order."""
    lab = np.zeros((K_NODES, K_NODES), dtype=bool)
    for i, u in enumerate(sample):
        for j, v in enumerate(sample):
            if i != j and (u, v) in edges:
                lab[i, j] = True
    return lab


def offdiag(matrix):
    mask = ~np.eye(K_NODES, dtype=bool)
    return np.asarray(matrix)[mask]


def sample_score(matrix, sample, edges):
    """AUROC for one subgraph. An unusable matrix scores exactly 0.0, never partial credit."""
    labels = offdiag(sample_labels(sample, edges))
    if matrix is None:
        return 0.0, "missing"
    try:
        mat = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError):
        return 0.0, "uncastable"
    if mat.shape != (K_NODES, K_NODES):
        return 0.0, "bad_shape"
    scores = offdiag(mat)
    if not np.all(np.isfinite(scores)):
        return 0.0, "non_finite"
    value = auroc(scores, labels)
    if not math.isfinite(value):
        return 0.0, "degenerate_labels"
    return value, "ok"


def panel_metric(matrices, batches, edges):
    """Mean per-subgraph AUROC over one graded half.

    `matrices` is a list parallel to `batches`, each element either None (the batch produced
    no usable output) or an array of shape (len(batch), 5, 5).
    """
    per_sample, reasons = [], {}
    for batch, mats in zip(batches, matrices):
        for k, sample in enumerate(batch):
            mat = None if mats is None else mats[k]
            value, why = sample_score(mat, sample, edges)
            per_sample.append(value)
            reasons[why] = reasons.get(why, 0) + 1
    per_sample = np.asarray(per_sample, dtype=np.float64)
    return {
        "n_samples": int(per_sample.shape[0]),
        "mean_auroc": float(per_sample.mean()) if per_sample.size else 0.0,
        "std_auroc": float(per_sample.std(ddof=1)) if per_sample.size > 1 else 0.0,
        "sem_auroc": (float(per_sample.std(ddof=1) / math.sqrt(per_sample.shape[0]))
                      if per_sample.size > 1 else 0.0),
        "call_outcomes": reasons,
    }


def graded_reward(metric, valid):
    """rational_squash: 0 at M0, 0.5 at BASELINE_SCORE, monotone, uncapped, asymptotic to 1."""
    if not valid:
        return 0.0
    x_ref = BASELINE_SCORE - M0
    if not x_ref > 0:
        raise AssertionError("malformed task: the published result does not beat the baseline")
    if not math.isfinite(metric):
        return 0.0
    u = max(0.0, float(metric) - M0) / x_ref
    return u / (1.0 + u)
