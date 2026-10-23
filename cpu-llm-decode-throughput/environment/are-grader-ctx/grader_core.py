from __future__ import annotations

M0 = 1.0
BASELINE_SCORE = 18.0


def graded_reward(speedup, valid):
    """Map the measured speedup to a reward in [0, 1]; 0 for anything not valid."""
    if not valid:
        return 0.0
    x_ref = BASELINE_SCORE - M0
    assert x_ref > 0, "malformed task: the baseline score does not beat the trivial baseline"
    u = max(0.0, float(speedup) - M0) / x_ref
    return u / (1.0 + u)
