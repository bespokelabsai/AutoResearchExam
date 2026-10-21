import json
import math
import sys
from pathlib import Path

METRIC_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/logs/verifier/metric.json")
REWARD_PATH = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/logs/verifier/reward.txt")

try:
    record = json.loads(METRIC_PATH.read_text())
    if not record.get("valid", False):
        raise ValueError(f"invalid submission: {record.get('detail', '')[:200]}")
    value = float(record["reward"])
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
