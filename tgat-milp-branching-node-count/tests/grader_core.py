from __future__ import annotations

import json
import math
from pathlib import Path

NODE_CAP = 20000

M0_BY_PANEL = {"intermediate": 398.89, "final": 595.15}
M0_DEFAULT = 487.0
BASELINE_SCORE = 126.0


def m0_for(panel):
    return M0_BY_PANEL.get(panel, M0_DEFAULT)

MAX_SUBMISSION_BYTES = 200 * 1024 * 1024


def shifted_geomean(nodes):
    """exp(mean(log(n + 1))) - 1 over the run node counts."""
    values = [float(n) for n in nodes]
    if not values:
        raise ValueError("no runs to aggregate")
    return math.exp(sum(math.log(v + 1.0) for v in values) / len(values)) - 1.0


def reward_from_metric(metric, m0=M0_DEFAULT):
    """Map a node-count metric to a reward in [0, 1]; lower metric is better."""
    if metric is None or not math.isfinite(metric) or metric < 0:
        return 0.0
    x_ref = math.log1p(m0) - math.log1p(BASELINE_SCORE)
    assert x_ref > 0, "malformed task: the baseline score does not beat the trivial rule"
    x = math.log1p(m0) - math.log1p(metric)
    u = max(0.0, x) / x_ref
    return u / (1.0 + u)


def load_panel(panel_dir):
    """Read one sealed panel: instance seeds and solver seed shifts."""
    spec = json.loads((Path(panel_dir) / "panel.json").read_text())
    seeds = [int(s) for s in spec["seeds"]]
    shifts = [int(k) for k in spec["shifts"]]
    if not seeds or not shifts:
        raise ValueError(f"empty panel in {panel_dir}")
    return seeds, shifts
