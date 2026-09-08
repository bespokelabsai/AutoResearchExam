from __future__ import annotations

import json
import math
from pathlib import Path

REWARD_PATH = Path("/logs/verifier/reward.txt")
METRIC_PATH = Path("/logs/verifier/metric.json")

try:
    payload = json.loads(METRIC_PATH.read_text())
    value = float(payload["reward"]) if payload.get("valid") else 0.0
    if not math.isfinite(value):
        raise ValueError("non-finite reward")
    reward = f"{min(1.0, max(0.0, value)):.12g}"
    REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    REWARD_PATH.write_text(reward + "\n")
    print(f"reward={reward}")
except Exception as exc:
    REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    REWARD_PATH.write_text("0\n")
    print(f"reward=0 (fallback: {exc})")
