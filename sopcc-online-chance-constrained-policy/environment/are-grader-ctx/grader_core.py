from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sopcc

HIDDEN = Path(__file__).resolve().parent / "hidden_data"
SPLITS = ("final", "intermediate")

ANCHORS = {

    "final": (4.212162974044415, 5.403678393388951),
    "intermediate": (4.90108511497506, 5.935456731233547),
}


def load_panel(split: str):
    """The sealed panel for `split`: instances plus their pre-drawn cost noise."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    with np.load(HIDDEN / split / "panel.npz") as data:
        coords, rewards, noise, draw = (data["coords"], data["rewards"],
                                        data["noise"], data["draw"])
    return [
        {"draw": int(draw[i]), "index": i,
         "instance": sopcc.instance_from_arrays(coords[i], rewards[i]),
         "noise": noise[i]}
        for i in range(coords.shape[0])
    ]


def graded_reward(metric: float, split: str) -> float:
    """Map the raw metric to a reward in [0, 1): 0 at m0, 0.5 at m_ref, no cutoff above."""
    m0, m_ref = ANCHORS[split]
    x_ref = m_ref - m0
    if not x_ref > 0:
        raise AssertionError("malformed task: the reference does not beat the trivial baseline")
    if not np.isfinite(metric):
        return 0.0
    u = max(0.0, float(metric) - m0) / x_ref
    return u / (1.0 + u)
