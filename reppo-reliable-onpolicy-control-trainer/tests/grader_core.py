from __future__ import annotations

import json
import math
from typing import Any

TAU = 0.9
HOLD_CHECKPOINTS = 10
N_CHECKPOINTS = 40
METRIC_NAME = "reliability_performance_score"
SCORING_VERSION = "1.1.0"


def normalised_scores(checkpoint_returns, r_rand: float, r_ref: float) -> list[float]:
    """s_k = clip((R_k - R_rand) / (R_ref - R_rand), 0, 1); a missing checkpoint is 0."""
    denom = float(r_ref) - float(r_rand)
    if denom <= 0.0:
        raise ValueError("malformed panel: R_ref does not exceed R_rand")
    out = []
    for x in checkpoint_returns:
        if x is None or not math.isfinite(float(x)):
            out.append(0.0)
        else:
            out.append(min(1.0, max(0.0, (float(x) - float(r_rand)) / denom)))
    return out


def run_is_reliable(scores: list[float]) -> bool:
    """True iff the run holds s >= TAU over the last HOLD_CHECKPOINTS checkpoints."""
    tail = scores[-HOLD_CHECKPOINTS:]
    return len(tail) == HOLD_CHECKPOINTS and all(s >= TAU for s in tail)


def reliable_success_fraction(runs: list[dict[str, Any]]) -> float:
    if not runs:
        return 0.0
    return sum(1 for r in runs if r["reliable"]) / len(runs)


def performance_scores(checkpoint_returns, r_rand: float, r_ref: float) -> list[float]:
    """Fraction of the remaining cost removed beyond the reliability threshold."""
    threshold = float(r_rand) + TAU * (float(r_ref) - float(r_rand))
    if not all(math.isfinite(x) for x in (r_rand, r_ref, threshold)):
        raise ValueError("malformed panel: non-finite return anchor")
    if r_ref <= r_rand or threshold >= 0.0:
        raise ValueError("malformed panel: expected R_rand < R_ref and threshold < 0")
    return [
        0.0 if x is None or not math.isfinite(float(x))
        else min(1.0, max(0.0, (float(x) - threshold) / -threshold))
        for x in checkpoint_returns
    ]


def graded_reward(metric: float, valid: bool) -> float:
    """Return the composite directly; benchmark difficulty mapping is separate."""
    if not valid or not math.isfinite(float(metric)):
        return 0.0
    return min(1.0, max(0.0, float(metric)))


def score_panel(panel: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine reliability and mean performance over the complete panel."""
    by_env = {e["name"]: e for e in panel["envs"]}
    expected = {(e["name"], seed) for e in panel["envs"] for seed in panel["run_seeds"]}
    actual = [(r["env"], r["seed"]) for r in results]
    if not expected or len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("incomplete or duplicate panel results")
    runs = []
    for r in results:
        env = by_env[r["env"]]
        returns = r["checkpoint_returns"]
        if len(returns) != N_CHECKPOINTS:
            raise ValueError("expected 40 checkpoints per run; missing checkpoints must be None")
        returns = [
            float(x) if x is not None and math.isfinite(float(x)) else None
            for x in returns
        ]
        performance = performance_scores(returns, env["r_rand"], env["r_ref"])
        scores = normalised_scores(returns, env["r_rand"], env["r_ref"])
        runs.append(
            {
                "env": r["env"],
                "seed": r["seed"],
                "status": r["status"],
                "final_s": scores[-1],
                "max_s": max(scores),
                "reliable": run_is_reliable(scores),
                "performance_improvement": sum(performance[-HOLD_CHECKPOINTS:]) / HOLD_CHECKPOINTS,
                "child_cpu_seconds": r.get("child_cpu_seconds"),
            }
        )
    reliability = reliable_success_fraction(runs)
    performance = sum(r["performance_improvement"] for r in runs) / len(runs)
    metric = 0.5 * reliability + 0.5 * performance
    per_env: dict[str, float] = {}
    for r in runs:
        per_env.setdefault(r["env"], 0.0)
        per_env[r["env"]] += 1.0 / max(1, sum(1 for q in runs if q["env"] == r["env"])) * (
            1.0 if r["reliable"] else 0.0
        )
    return {
        "metric_name": METRIC_NAME,
        "scoring_version": SCORING_VERSION,
        "metric": metric,
        "reliable_success_fraction": reliability,
        "mean_performance_improvement": performance,
        "n_runs": len(runs),
        "n_reliable": sum(1 for r in runs if r["reliable"]),
        "per_env_fraction": {k: round(v, 4) for k, v in per_env.items()},
        "mean_final_s": round(sum(r["final_s"] for r in runs) / max(1, len(runs)), 4),
        "runs": runs,
    }


def load_panel(path: str) -> dict[str, Any]:
    with open(path) as fh:
        return json.load(fh)
