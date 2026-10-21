from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import grader_core

METRIC_PATH = Path("/logs/verifier/metric.json")


def test_policy() -> None:
    split = os.environ.get("BUDGETED_SPLIT", "final")
    if split not in {"intermediate", "final"}:
        raise ValueError("unknown grading split")
    try:
        policy_path = Path("/app/output/policy.py")
        result = grader_core.run(
            Path("/tests/hidden_data") / split,
            policy_path,
        )
    except Exception as error:
        METRIC_PATH.write_text(json.dumps({
            "valid": False,
            "metric_name": "balanced_accuracy",
            "metric": 0.0,
            "reward": 0.0,
            "error_type": type(error).__name__,
        }))
        raise
    METRIC_PATH.write_text(json.dumps(result, sort_keys=True))
    assert result["valid"] is True
