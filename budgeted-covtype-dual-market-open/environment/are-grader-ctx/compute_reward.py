from __future__ import annotations

import json
import math
from pathlib import Path

METRIC_PATH = Path("/logs/verifier/metric.json")
REWARD_PATH = Path("/logs/verifier/reward.txt")


try:
    value = float(json.loads(METRIC_PATH.read_text())["reward"])
    if not math.isfinite(value):
        raise ValueError("non-finite reward")
    reward = min(1.0, max(0.0, value))
    REWARD_PATH.write_text(f"{reward:.12g}\n")
except Exception:
    REWARD_PATH.write_text("0\n")
