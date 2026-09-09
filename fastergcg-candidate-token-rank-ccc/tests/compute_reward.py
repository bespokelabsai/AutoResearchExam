import json
import math
from pathlib import Path

REWARD_PATH = Path("/logs/verifier/reward.txt")
METRIC_PATH = Path("/logs/verifier/metric.json")
FALLBACK = "0"


def clamp01(value) -> float:
    """Any object -> a finite float in [0, 1]. Never raises.

    Anything that is not a real, finite number -- None, a string that is not a
    number, a container, NaN, +-inf from an overflowed division, an integer too
    large to be a float -- is not a reward and lands on 0.0, the same value every
    other failure path writes. Booleans are not rewards either.
    """
    try:
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            return 0.0
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    if number <= 0.0:
        return 0.0
    return 1.0 if number >= 1.0 else number


def format_reward(value: float) -> str:
    """A clamped float -> the exact text written to reward.txt.

    The formatted text is parsed back and re-checked, so no rounding of the last
    significant digit can turn a bounded value into an out-of-range, negative or
    unreadable one on the way to disk.
    """
    try:
        text = f"{value:.12g}"
        back = float(text)
    except (TypeError, ValueError, OverflowError):
        return FALLBACK
    if not math.isfinite(back) or back < 0.0 or back > 1.0:
        return FALLBACK
    return text


def write_reward(text: str) -> bool:
    """Write the one line. Returns False only if the filesystem refused it."""
    try:
        REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        REWARD_PATH.write_text(text + "\n")
        return True
    except Exception:
        return False


def main() -> int:
    fallback_reason = None
    try:
        raw = json.loads(METRIC_PATH.read_text())["reward"]
    except Exception as exc:
        raw, fallback_reason = None, repr(exc)

    text = format_reward(clamp01(raw))
    if text == FALLBACK and fallback_reason is None and raw != 0:
        fallback_reason = f"unusable reward {raw!r}"
    if not write_reward(text):
        write_reward(FALLBACK)

    if fallback_reason is None:
        print(f"reward={text}")
    else:
        print(f"reward={text} (fallback: {fallback_reason})")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        try:
            already = REWARD_PATH.read_text().strip()
        except Exception:
            already = ""
        if not already:
            write_reward(FALLBACK)
        print(f"reward={already or FALLBACK} (fallback: {exc!r})")
        raise SystemExit(0)
