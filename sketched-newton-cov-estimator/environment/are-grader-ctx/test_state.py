import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grader_core

DELIVERABLE = Path(os.environ.get("GRADER_DELIVERABLE", "/app/output"))
SPLIT = os.environ.get("GRADE_SPLIT", "final")
SPLIT_DIR = Path(os.environ.get(
    "GRADER_SPLIT_DIR", str(Path(__file__).resolve().parent / "hidden_data" / SPLIT)))
METRIC_PATH = Path(os.environ.get("GRADER_METRIC_PATH", "/logs/verifier/metric.json"))


def test_limiting_covariance_estimator():
    """Run the submitted estimator over every sealed instance and score its variance error.

    The agent's code runs only in per-replication subprocesses launched by grader_core, as a
    non-root uid, with no access to the instance parameters or to the true covariance. This
    process recomputes the metric from verifier-owned ground truth; nothing the submission
    reports about itself is read.
    """
    report = {"metric": None, "reward": 0.0, "split": SPLIT}
    try:
        report = grader_core.grade(str(SPLIT_DIR), str(DELIVERABLE))
        report["split"] = SPLIT
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        METRIC_PATH.parent.mkdir(parents=True, exist_ok=True)
        METRIC_PATH.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")

    print("")
    for line in grader_core.summary_lines(report) if "error" not in report else \
            [f"grading failed: {report['error']}", "metric (lower is better)  : n/a",
             "reward                    : 0.000000"]:
        print(line)
    assert "error" not in report, report.get("error")
