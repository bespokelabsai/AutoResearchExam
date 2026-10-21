import json
import math
from pathlib import Path

REWARD_PATH = Path("/logs/verifier/reward.txt")

try:
    REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(Path("/logs/verifier/metric.json").read_text())
    if not payload.get("valid", False):
        raise ValueError(f"invalid submission: {payload.get('reason')}")
    value = float(payload["reward"])
    if not math.isfinite(value):
        raise ValueError("non-finite reward")
    reward = f"{min(1.0, max(0.0, value)):.12g}"
    REWARD_PATH.write_text(reward + "\n")
    print(f"reward={reward}")
except Exception as exc:
    REWARD_PATH.write_text("0\n")
    print(f"reward=0 (fallback: {exc})")
