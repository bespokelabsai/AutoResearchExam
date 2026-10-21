import json
import math
from pathlib import Path

REWARD_PATH = Path("/logs/verifier/reward.txt")

try:
    REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    value = float(json.loads(Path("/logs/verifier/metric.json").read_text())["reward"])
    if not math.isfinite(value):
        raise ValueError("non-finite reward")
    reward = f"{min(1.0, max(0.0, value)):.12g}"
    REWARD_PATH.write_text(reward + "\n")
    print(f"reward={reward}")
except Exception as exc:
    try:
        REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    REWARD_PATH.write_text("0\n")
    print(f"reward=0 (fallback: {exc})")
