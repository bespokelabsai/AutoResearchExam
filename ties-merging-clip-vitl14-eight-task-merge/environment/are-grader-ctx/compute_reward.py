from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

METRIC_PATH = Path("/logs/verifier/metric.json")
REWARD_PATH = Path("/logs/verifier/reward.txt")


def clamp(value: object) -> float:
    """Any object -> a finite float in [0, 1]. Never raises, never returns NaN, inf or -0.0."""
    try:
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            return 0.0
        x = float(value)
    except BaseException:
        return 0.0
    if not math.isfinite(x):
        return 0.0
    if not x > 0.0:
        return 0.0
    return 1.0 if x > 1.0 else x


def render(x: float) -> str:
    """A clamped float -> its decimal text, re-parsed to prove the bound survived formatting."""
    try:
        text = f"{x:.12g}"
        back = float(text)
    except BaseException:
        return "0"
    if not math.isfinite(back) or back < 0.0 or back > 1.0:
        return "0"
    return text


def emit(text: str) -> None:
    """Replace reward.txt atomically, so no reader ever sees a partial or stale token."""
    REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = REWARD_PATH.with_name(REWARD_PATH.name + ".tmp")
    with open(tmp, "w") as fh:
        fh.write(text + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, REWARD_PATH)


def emit_zero_hard() -> None:
    """Last resort when the atomic write failed: clear whatever occupies the path, write 0."""
    try:
        REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    except BaseException:
        pass
    try:
        if REWARD_PATH.is_dir():
            os.rmdir(REWARD_PATH)
        elif REWARD_PATH.exists():
            os.unlink(REWARD_PATH)
    except BaseException:
        pass
    REWARD_PATH.write_text("0\n")


def main() -> None:
    note = ""
    try:
        payload = json.loads(METRIC_PATH.read_text())
        text = render(clamp(payload["reward"] if isinstance(payload, dict) else None))
    except BaseException as exc:
        text, note = "0", f" (fallback: {exc})"
    try:
        emit(text)
        print(f"reward={text}{note}")
    except BaseException as exc:
        try:
            emit_zero_hard()
            print(f"reward=0 (fallback: {exc})")
        except BaseException:
            pass


def selftest() -> None:
    """Every adversarial input the contract names, checked at the clamp and at the text."""
    huge = 1e308
    cases = [
        float("nan"), float("inf"), float("-inf"), -huge * huge, huge / 1e-308,
        -0.0, -1.0, -1e-300, 2.0, 1e309, 10 ** 400, "nan", "-inf", "-1", "1e400", "",
        "0.5", 0.5, 1.0, 0.0, None, True, False, [], {}, object(),
    ]
    for case in cases:
        c = clamp(case)
        assert isinstance(c, float) and math.isfinite(c) and 0.0 <= c <= 1.0, (case, c)
        assert math.copysign(1.0, c) > 0.0, (case, "negative zero")
        t = render(c)
        assert 0.0 <= float(t) <= 1.0 and not t.startswith("-"), (case, t)
    assert render(clamp(0.0)) == "0" and render(clamp(float("nan"))) == "0"
    assert render(clamp(2.0)) == "1" and render(clamp(0.5)) == "0.5"
    print(f"selftest ok: {len(cases)} inputs clamped into [0, 1]")


if __name__ == "__main__":
    if "--selftest" in sys.argv[1:]:
        selftest()
    else:
        main()
