#!/usr/bin/env python3
import json
import math
from pathlib import Path

REWARD_PATH = Path("/logs/verifier/reward.txt")


def clamp01(raw: object) -> float:
    """Total map from any object at all onto a finite reward inside [0, 1].

    Never raises, and never returns NaN, an infinity or a value outside the band. A
    non-numeric, unparseable, overflowing or non-finite input degrades to the floor, 0.0,
    which is exactly what an invalid submission scores.
    """
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, value))


def emit(raw: object) -> None:
    """Clamp, write, then re-read: the file that lands must itself parse back into [0, 1].

    A short or garbled write is the one way a clamped value could still reach the platform
    out of band, so the readback is checked and anything unexpected is replaced by the floor.
    """
    REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    text = f"{clamp01(raw):.12g}"
    REWARD_PATH.write_text(text + "\n")
    if clamp01(REWARD_PATH.read_text().strip()) != float(text):
        REWARD_PATH.write_text("0\n")


try:
    record = json.loads(Path("/logs/verifier/metric.json").read_text())
    emit(record["reward"])
    print(f"reward={REWARD_PATH.read_text().strip()}")
except BaseException as exc:
    try:
        emit(0.0)
    except BaseException:
        try:
            with open(REWARD_PATH, "w") as handle:
                handle.write("0\n")
        except BaseException:
            pass
    print(f"reward=0 (fallback: {exc})")
