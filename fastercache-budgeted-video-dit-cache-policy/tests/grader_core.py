from __future__ import annotations

import math
import os
import stat
from collections.abc import Iterable, Sequence
from pathlib import Path
from types import ModuleType

ANCHORS = {
    "intermediate": (26.2950, 13.3142),
    "final": (25.3818, 12.4367),
}
M_REFERENCE = 28.93
DEGENERATE_REWARD = 0.10

ENTRY_POINT = "policy.py"
POLICY_CLASS = "CachePolicy"
MAX_DELIVERABLE_BYTES = 256 * 1024 * 1024

WRITABLE_ROOT = Path("/app/output")
AGENT_ROOT = Path("/app")
TRUSTED_ROOT = Path("/tests")


def graded_reward(metric: float | None, valid: bool, split: str = "final") -> float:
    """Map the raw metric to a reward in [0, 1].  Invalid short-circuits to exactly 0."""
    if not valid or metric is None or not math.isfinite(float(metric)):
        return 0.0
    m = float(metric)
    m_strongest_trivial, m_weakest_defensible = ANCHORS[split]
    x_ref = M_REFERENCE - m_strongest_trivial
    if x_ref <= 0:
        raise ValueError("malformed task: the reference does not beat the trivial baseline")
    x = m - m_strongest_trivial
    if x < 0.0:
        band = m_strongest_trivial - m_weakest_defensible
        if band <= 0.0:
            return 0.0
        v = max(0.0, m - m_weakest_defensible) / band
        return DEGENERATE_REWARD * min(1.0, v)
    u = x / x_ref
    return DEGENERATE_REWARD + (1.0 - DEGENERATE_REWARD) * (u / (1.0 + u))


class SubmissionError(RuntimeError):
    """The submission violates a constraint stated in instruction.md."""


class WritableSurfaceError(RuntimeError):
    """Something outside the writable surface was in a position to affect the score."""


def _is_under(path: str | Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(root.resolve())
    except (ValueError, OSError):
        return False
    return True


def check_writable_surface(modules: Sequence[ModuleType],
                           data_paths: Iterable[str | Path] = ()) -> dict:
    """Enforce the writable surface instruction.md states, on the verifier side.

    A solution may change files under ``/app/output`` and nothing else.  The verifier side
    of that same list is: every module that drives the graded run is loaded out of the
    sealed ``/tests`` tree, and every path the grader reads its prompts, weights and
    references from sits outside ``/app`` altogether.  ``/app/output`` itself is never
    imported here -- it is copied to a staging directory and reached only over the worker
    pipe -- so a legitimate submission always passes.  A run that does not pass is one
    where agent-writable content reached the grading path, and it is not gradeable.
    """
    for module in modules:
        name = getattr(module, "__name__", repr(module))
        origin = getattr(module, "__file__", None)
        if origin is None:
            raise WritableSurfaceError(f"grading module {name!r} has no file of origin")
        if not _is_under(origin, TRUSTED_ROOT):
            raise WritableSurfaceError(
                f"grading module {name!r} was loaded from {origin}, outside the sealed "
                f"{TRUSTED_ROOT}")
    checked = []
    for path in data_paths:
        if _is_under(path, AGENT_ROOT):
            raise WritableSurfaceError(
                f"the grader would read {path}, which lies under the agent-writable "
                f"{AGENT_ROOT}")
        checked.append(str(path))
    return {
        "writable_root": str(WRITABLE_ROOT),
        "trusted_root": str(TRUSTED_ROOT),
        "modules_checked": len(modules),
        "data_paths_checked": checked,
    }


def inspect_submission(root: str | Path) -> dict:
    """Walk the submitted tree with ``os.lstat`` and reject links before any copy.

    A submitted symlink would be followed by a privileged copy and could materialise
    verifier-owned files where the score is computed from; a hard link to an existing
    file has the same effect without a link to notice.  Both are rejected outright, at
    every depth, before ``shutil.copytree`` ever runs.
    """
    root = Path(root)
    if not root.is_dir():
        raise SubmissionError(f"no deliverable directory at {root}")
    total = 0
    files = 0
    stack = [root]
    seen = set()
    while stack:
        current = stack.pop()
        key = os.stat(current, follow_symlinks=False).st_ino
        if key in seen:
            raise SubmissionError("the deliverable contains a directory cycle")
        seen.add(key)
        for entry in sorted(os.scandir(current), key=lambda e: e.name):
            st = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(st.st_mode):
                raise SubmissionError(
                    f"the deliverable contains a symbolic link: "
                    f"{Path(entry.path).relative_to(root)}")
            if stat.S_ISDIR(st.st_mode):
                stack.append(Path(entry.path))
                continue
            if not stat.S_ISREG(st.st_mode):
                raise SubmissionError(
                    f"the deliverable contains a non-regular file: "
                    f"{Path(entry.path).relative_to(root)}")
            if st.st_nlink != 1:
                raise SubmissionError(
                    f"the deliverable contains a hard link: "
                    f"{Path(entry.path).relative_to(root)}")
            total += st.st_size
            files += 1
    if total > MAX_DELIVERABLE_BYTES:
        raise SubmissionError(
            f"the deliverable holds {total} bytes, over the "
            f"{MAX_DELIVERABLE_BYTES} byte limit")
    entry_point = root / ENTRY_POINT
    if not entry_point.is_file() or entry_point.stat().st_size == 0:
        raise SubmissionError(f"no non-empty {ENTRY_POINT} in the deliverable directory")
    return {"bytes": total, "files": files}
