from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import grader_core

PHASE = os.environ.get("GRADE_SPLIT", "final")
METRIC_PATH = Path("/logs/verifier/metric.json")

LAUNCH_PREFIX = ["/sbin/runuser", "-u", grader_core.CANDIDATE_USER, "--"]
NEW_SESSION = os.setsid


@pytest.fixture(scope="session")
def result():
    record = grader_core.grade(phase=PHASE, launch_prefix=LAUNCH_PREFIX,
                               new_session=NEW_SESSION,
                               call_timeout=grader_core.CALL_BUDGET_SEC,
                               import_timeout=grader_core.IMPORT_BUDGET_SEC)
    METRIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRIC_PATH.write_text(json.dumps(record, indent=2, sort_keys=True))
    print(f"phase                : {record['phase']}")
    print(f"runs in panel        : {record['n_runs']}")
    print(f"submission valid     : {record['valid']}")
    print(f"detail               : {record['detail']}")
    if record["valid"]:
        print(f"mean final macro-F1  : {record['metric']:.4f}")
        print(f"across-run sd        : {record['run_sd']:.4f}")
        print(f"run min / max        : {record['run_min']:.4f} / {record['run_max']:.4f}")
        print("mean macro-F1 by AL iteration: "
              + "  ".join(f"{v:.3f}" for v in record["per_iteration_mean"]))
    print(f"slowest prune call   : {record['slowest_call_sec']}s of "
          f"{grader_core.CALL_BUDGET_SEC:g}s")
    return record


def test_deliverable_is_stageable(result):
    """The deliverable directory exists, holds a non-empty prune.py, and contains no symlink
    or hard link -- a submitted link would let the staging copy reach outside the tree."""
    assert result["detail"] != "", "grade produced no detail string"
    assert "contains links" not in result["detail"], result["detail"]
    assert "missing or empty entry point" not in result["detail"], result["detail"]
    assert "no deliverable directory" not in result["detail"], result["detail"]


def test_every_seeded_run_completed(result):
    """Every run in the sealed panel imported the module, answered all five prune calls
    inside the per-call budget, and returned a selection the protocol accepts."""
    assert result["valid"], result["detail"]
    assert result["slowest_call_sec"] <= grader_core.CALL_BUDGET_SEC


def test_metric_is_recorded(result):
    """The recorded metric is a finite macro-F1 in percent, and the reward derived from it is
    inside [0, 1]."""
    assert METRIC_PATH.is_file()
    assert result["metric"] is not None and math.isfinite(result["metric"])
    assert 0.0 <= result["metric"] <= 100.0
    assert 0.0 <= result["reward"] <= 1.0
