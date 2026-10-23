import json
import math
import os
import tempfile
from pathlib import Path

REWARD_PATH = Path("/logs/verifier/reward.txt")
METRIC_PATH = Path("/logs/verifier/metric.json")


def clamp_unit(value) -> float:
    """Map ANY object to a finite float in the closed interval [0, 1].

    The comparisons are written so that NaN cannot slip through either end: NaN fails
    `math.isfinite` before it reaches them, and the `<= 0.0` branch also normalises -0.0
    to 0.0 so no rendered reward can ever carry a minus sign.
    """
    try:
        x = float(value)
    except Exception:
        return 0.0
    if not math.isfinite(x):
        return 0.0
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x


def render(x: float) -> str:
    """Render the clamped value, then read it back: only a string that parses to a finite
    number inside [0, 1] is ever handed to the writer.  Anything else degrades to "0"."""
    try:
        text = f"{x:.12g}"
        back = float(text)
    except Exception:
        return "0"
    if not math.isfinite(back) or back < 0.0 or back > 1.0:
        return "0"
    return text


def emit(text: str) -> bool:
    """Write reward.txt.  Never raises; returns whether the bytes landed.

    The write goes through a temp file in the same directory and one os.replace, so a
    reader can only ever see the whole value or no file -- a crash partway through a
    direct write could otherwise leave a truncated, unparseable reward.txt behind.
    """
    payload = text + "\n"
    try:
        REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(dir=str(REWARD_PATH.parent), prefix=".reward-")
        with os.fdopen(fd, "w") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, REWARD_PATH)
        return True
    except Exception:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except Exception:
                pass
        try:
            REWARD_PATH.write_text(payload)
            return True
        except Exception:
            return False


def in_range(value) -> bool:
    """Whether `value` is a finite number inside [0, 1]."""
    try:
        x = float(value)
    except Exception:
        return False
    return math.isfinite(x) and 0.0 <= x <= 1.0


def already_bounded() -> bool:
    """Whether reward.txt on disk already holds a finite value inside [0, 1].

    The last-resort handler consults this so that a failure AFTER a good reward was
    written -- a closed stdout, a signal during the diagnostic print -- cannot replace a
    valid submission's score with 0.
    """
    try:
        return in_range(REWARD_PATH.read_text().strip())
    except Exception:
        return False


def say(message: str) -> None:
    """Diagnostics must never be the reason this script fails."""
    try:
        print(message)
    except Exception:
        pass


def main() -> None:
    error = None
    try:
        value = json.loads(METRIC_PATH.read_text())["reward"]
    except Exception as exc:
        value, error = None, f"{type(exc).__name__}: {exc}"
    reward = render(clamp_unit(value))
    wrote = emit(reward)
    if error is None and not in_range(value):
        error = f"clamped from {value!r}"
    notes = [n for n in (error, None if wrote else "reward.txt is unwritable") if n]
    say(f"reward={reward}" + (f" ({'; '.join(notes)})" if notes else ""))


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        kept = already_bounded()
        if not kept:
            emit("0")
        try:
            detail = f"{type(exc).__name__}: {exc}"
        except Exception:
            detail = "unprintable exception"
        say(("reward.txt kept" if kept else "reward=0") + f" (fallback: {detail})")
