from __future__ import annotations

import json
from typing import Any

TAU = 0.9
HOLD_CHECKPOINTS = 10


M0 = 0.0
BASELINE_SCORE = 0.8


def normalised_scores(checkpoint_returns, r_rand: float, r_ref: float) -> list[float]:
    """s_k = clip((R_k - R_rand) / (R_ref - R_rand), 0, 1); a missing checkpoint is 0."""
    denom = float(r_ref) - float(r_rand)
    if denom <= 0.0:
        raise ValueError("malformed panel: R_ref does not exceed R_rand")
    out = []
    for x in checkpoint_returns:
        if x is None:
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


def graded_reward(metric: float, valid: bool) -> float:
    """rational_squash: 0 at the trivial baseline, 0.5 at the baseline score, no cutoff."""
    if not valid:
        return 0.0
    x_ref = BASELINE_SCORE - M0
    if x_ref <= 0.0:
        raise ValueError("malformed task: the baseline score does not beat the trivial baseline")
    u = max(0.0, float(metric) - M0) / x_ref
    return u / (1.0 + u)


def score_panel(panel: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    """Turn raw per-run checkpoint returns into the metric and the reward."""
    by_env = {e["name"]: e for e in panel["envs"]}
    runs = []
    for r in results:
        env = by_env[r["env"]]
        scores = normalised_scores(r["checkpoint_returns"], env["r_rand"], env["r_ref"])
        runs.append(
            {
                "env": r["env"],
                "seed": r["seed"],
                "status": r["status"],
                "final_s": scores[-1],
                "max_s": max(scores),
                "reliable": run_is_reliable(scores),
                "child_cpu_seconds": r.get("child_cpu_seconds"),
            }
        )
    metric = reliable_success_fraction(runs)
    per_env: dict[str, float] = {}
    for r in runs:
        per_env.setdefault(r["env"], 0.0)
        per_env[r["env"]] += 1.0 / max(1, sum(1 for q in runs if q["env"] == r["env"])) * (
            1.0 if r["reliable"] else 0.0
        )
    return {
        "metric_name": "reliable_success_fraction",
        "metric": metric,
        "n_runs": len(runs),
        "n_reliable": sum(1 for r in runs if r["reliable"]),
        "per_env_fraction": {k: round(v, 4) for k, v in per_env.items()},
        "mean_final_s": round(sum(r["final_s"] for r in runs) / max(1, len(runs)), 4),
        "runs": runs,
    }


def load_panel(path: str) -> dict[str, Any]:
    with open(path) as fh:
        return json.load(fh)
