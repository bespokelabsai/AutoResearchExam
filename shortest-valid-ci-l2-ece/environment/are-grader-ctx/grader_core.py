from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import panel

ALPHA = 0.10
REPLICATIONS = 2000
COVERAGE_GATE = 0.88
PER_CALL_CAP_SEC = 2.0

M0 = 0.0593
M_REF = 0.0095


def graded_reward(metric: float | None) -> float:
    """Map the metric (lower is better) to a reward in [0, 1].

    Two-anchor log-ratio squash on the improvement factor rho = M0 / metric:
        u = max(0, log(rho)) / log(M0 / M_REF),   reward = u / (1 + u).
    The strongest degenerate predictor earns exactly 0, anything worse also earns 0,
    the reference construction earns about 0.5, and the reward keeps rising without a
    cutoff, approaching 1 asymptotically.
    """
    if metric is None or not math.isfinite(metric):
        return 0.0
    if metric <= 1e-12:
        return 1.0
    rho = M0 / metric
    if rho <= 1.0:
        return 0.0
    u = math.log(rho) / math.log(M0 / M_REF)
    return u / (1.0 + u)




def load_panel(directory: str | Path) -> dict:
    doc = json.loads((Path(directory) / "panel.json").read_text())
    return doc


def replication_rngs(master_seed: int, setting_index: int, rep: int):
    """Deterministic (data rng, candidate seed) for one replication."""
    data_ss = np.random.SeedSequence(entropy=master_seed, spawn_key=(setting_index, rep, 0))
    cand_ss = np.random.SeedSequence(entropy=master_seed, spawn_key=(setting_index, rep, 1))
    cand_seed = int(cand_ss.generate_state(1, dtype=np.uint32)[0])
    return np.random.default_rng(data_ss), cand_seed


def theta_only(spec: dict, master_seed: int, setting_index: int, rep: int) -> float:
    """The true squared calibration error of a replication, without drawing its data."""
    rng, _ = replication_rngs(master_seed, setting_index, rep)
    return float(panel.theta_for(spec, rng))


def make_replication(spec: dict, master_seed: int, setting_index: int, rep: int):
    """Draw one replication: returns (Z, Y, theta_true, candidate_seed)."""
    rng, cand_seed = replication_rngs(master_seed, setting_index, rep)
    theta = panel.theta_for(spec, rng)
    Z, Y = panel.draw_replication(spec, theta, rng)
    return Z, Y, float(theta), cand_seed




def _length_and_cover(entry, k: int, theta: float):
    """Relative-length contribution and coverage flag for one replication.

    Any invalid outcome - a failed import, an exception, a timeout, a non-finite or
    mis-ordered return, or a value outside [0, k] - contributes the trivial length k
    and does not cover.
    """
    if not isinstance(entry, dict) or entry.get("status") != "ok":
        return float(k), False, entry.get("status", "missing") if isinstance(entry, dict) else "missing"
    lo, hi = entry.get("lo"), entry.get("hi")
    if not (isinstance(lo, (int, float)) and isinstance(hi, (int, float))):
        return float(k), False, "not_a_number"
    lo, hi = float(lo), float(hi)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return float(k), False, "non_finite"
    if not (0.0 <= lo <= hi <= float(k)):
        return float(k), False, "out_of_range"
    return hi - lo, bool(lo <= theta <= hi), "ok"


def score_panel(settings: list[dict], thetas: dict, results: dict, replications: int) -> dict:
    """Aggregate per-setting coverage and relative length into the metric.

    `results[(si, rep)]` is the dict a candidate call produced (or is missing).
    """
    per_setting = []
    status_counts: dict[str, int] = {}
    for si, spec in enumerate(settings):
        k = int(spec["k"])
        total_len = 0.0
        covered = 0
        for rep in range(replications):
            entry = results.get((si, rep))
            length, cover, status = _length_and_cover(entry, k, thetas[(si, rep)])
            total_len += length
            covered += int(cover)
            status_counts[status] = status_counts.get(status, 0) + 1
        coverage = covered / replications
        rel_len = total_len / (replications * k)
        valid = coverage >= COVERAGE_GATE
        per_setting.append(
            {
                "index": si,
                "coverage": coverage,
                "relative_length": rel_len,
                "coverage_gate_met": bool(valid),
                "score": rel_len if valid else 1.0,
            }
        )
    metric = float(np.mean([s["score"] for s in per_setting]))
    return {
        "metric": metric,
        "per_setting": per_setting,
        "status_counts": status_counts,
        "replications": replications,
        "coverage_gate": COVERAGE_GATE,
    }
