from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path

REWARD_PATH = Path("/logs/verifier/reward.txt")
METRIC_PATH = Path("/logs/verifier/metric.json")


def clamped(raw: object) -> float:
    """Return ``raw`` as a finite float in [0, 1].  Anything unusable is exactly 0.

    A non-finite metric is a broken measurement, not a perfect one, so NaN and both
    infinities score 0 rather than saturating at 1 -- the same convention the grader
    itself applies upstream in grader_core.graded_reward.
    """
    try:
        value = float(raw)
    except BaseException:
        return 0.0
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, value))


def read_metric() -> float:
    """The clamped reward recorded by the grader, or 0 if the channel is unreadable."""
    try:
        document = json.loads(METRIC_PATH.read_text())
        return clamped(document["reward"])
    except BaseException as exc:
        print(f"falling back to 0: {type(exc).__name__}: {exc}")
        return 0.0


def format_reward(value: float) -> str:
    """Render the reward, then re-parse it so the text on disk is in bounds too."""
    try:
        text = f"{clamped(value):.12g}"
        parsed = float(text)
    except BaseException:
        return "0"
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        return "0"
    if parsed == 0.0:
        return "0"
    return text


def write_reward(text: str) -> bool:
    """Put ``text`` in reward.txt.  Never raises; reports whether the bytes landed."""
    payload = text + "\n"
    try:
        REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    except BaseException as exc:
        print(f"could not create {REWARD_PATH.parent}: {type(exc).__name__}: {exc}")
    try:
        REWARD_PATH.write_text(payload)
        return True
    except BaseException as exc:
        print(f"could not write {REWARD_PATH}: {type(exc).__name__}: {exc}")
    try:
        if REWARD_PATH.is_dir():
            shutil.rmtree(REWARD_PATH, ignore_errors=True)
        else:
            os.remove(REWARD_PATH)
    except BaseException:
        pass
    try:
        fd = os.open(str(REWARD_PATH), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            os.write(fd, payload.encode())
        finally:
            os.close(fd)
        return True
    except BaseException as exc:
        print(f"could not write {REWARD_PATH}: {type(exc).__name__}: {exc}")
        return False


def main() -> None:
    reward = format_reward(read_metric())
    if write_reward(reward):
        print(f"reward={reward}")
    else:
        print(f"reward={reward} (not written)")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        write_reward("0")
        print(f"reward=0 (compute_reward failed: {type(exc).__name__}: {exc})")
