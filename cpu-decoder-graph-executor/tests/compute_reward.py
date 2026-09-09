import json
import math
from pathlib import Path

REWARD_PATH = Path("/logs/verifier/reward.txt")
METRIC_PATH = Path("/logs/verifier/metric.json")

M0 = 1.0
BASELINE_SCORE = 1.304


def graded_reward(metric, valid, m0=M0, baseline_score=BASELINE_SCORE):
    if not valid:
        return 0.0
    x_ref = baseline_score - m0
    if not x_ref > 0:
        raise ValueError("malformed task: the anchor does not beat the trivial baseline")
    u = max(0.0, float(metric) - m0) / x_ref
    return u / (1.0 + u)


def main():
    try:
        record = json.loads(METRIC_PATH.read_text())
        value = graded_reward(record["metric"], bool(record.get("valid")))
        if not math.isfinite(value):
            raise ValueError("non-finite reward")
        reward = f"{min(1.0, max(0.0, value)):.12g}"
        REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
        REWARD_PATH.write_text(reward + "\n")
        print(f"metric={record['metric']:.6g} valid={record.get('valid')} reward={reward}")
    except Exception as exc:
        REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
        REWARD_PATH.write_text("0\n")
        print(f"reward=0 (fallback: {exc})")


if __name__ == "__main__":
    main()
