from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import grader_core
import run_panel

SOLUTION_SRC = os.environ.get("SOLUTION_DIR", "/app/solution")
GRADE_SPLIT = os.environ.get("GRADE_SPLIT", "final")
PANEL_PATH = f"/tests/hidden_data/{GRADE_SPLIT}/panel.json"
METRIC_PATH = Path("/logs/verifier/metric.json")
PYTHON = os.environ.get("PYTHON_BIN", "/usr/local/bin/python3")
WORKERS = 8




PANEL_TIMEOUT_SEC = 2000.0

_RESULT: dict | None = None


def _privilege_drop() -> tuple[str, ...]:
    """Resolve the tool that runs candidate code as an unprivileged, per-run uid.

    Fails closed: if the root verifier cannot drop privilege it must not run the
    submission at all, because a root candidate could rewrite the reward channel.
    ``{uid}`` is filled in per run by :func:`run_panel._job`; ``runuser`` is not used
    because it needs a passwd entry and the per-run uids have none.
    """
    if os.geteuid() != 0:
        return ()
    for path in ("/usr/bin/setpriv", "/usr/sbin/setpriv"):
        if os.path.exists(path):
            return (path, "--reuid={uid}", "--regid={uid}", "--clear-groups", "--")
    raise RuntimeError("no privilege-dropping tool in the verifier image")


def _grade() -> dict:
    started = time.monotonic()
    panel = grader_core.load_panel(PANEL_PATH)
    launcher = _privilege_drop()
    sealed_paths = run_panel.harden_shared_writable_paths()
    results = run_panel.execute_panel(
        panel,
        SOLUTION_SRC,
        workers=WORKERS,
        python_exe=PYTHON,
        launcher=launcher,
        timeout=PANEL_TIMEOUT_SEC,
    )
    scored = grader_core.score_panel(panel, results)
    scored["valid"] = True
    scored["reason"] = ""
    scored["reward"] = grader_core.graded_reward(scored["metric"], True)
    scored["grade_seconds"] = round(time.monotonic() - started, 1)
    scored["sealed_shared_paths"] = sealed_paths
    return scored


def result() -> dict:
    global _RESULT
    if _RESULT is None:
        try:
            _RESULT = _grade()
        except run_panel.DeliverableError as exc:
            _RESULT = {"valid": False, "reason": str(exc), "metric": 0.0, "reward": 0.0}
        except Exception as exc:
            _RESULT = {
                "valid": False,
                "reason": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-2000:],
                "metric": 0.0,
                "reward": 0.0,
            }
        METRIC_PATH.parent.mkdir(parents=True, exist_ok=True)
        METRIC_PATH.write_text(json.dumps(_RESULT, indent=1, sort_keys=True))
    return _RESULT


def test_deliverable_satisfies_the_stated_contract():
    """/app/solution/trainer.py exists, parses, and exposes train(env, seed, budget, report).

    A missing, unparseable or wrong-signature deliverable scores 0, which is the contract
    instruction.md states.
    """
    r = result()
    assert r["valid"], r["reason"]


def test_every_panel_run_produced_a_checkpoint_series():
    """All 72 (environment, seed) runs completed the harness protocol.

    Runs may legitimately end early (budget exhausted, a crash inside the submitted code);
    what this checks is that the harness itself produced a full record for each one, so the
    metric is computed over the whole panel rather than a truncated slice.
    """
    r = result()
    assert r["valid"], r["reason"]
    assert r["n_runs"] == 72, f"expected 72 runs, got {r['n_runs']}"


def test_metric_and_reward_are_recorded():
    """The raw metric and the mapped reward land in /logs/verifier/metric.json in range."""
    r = result()
    assert METRIC_PATH.exists()
    assert 0.0 <= float(r["reward"]) <= 1.0
    if r["valid"]:
        assert 0.0 <= float(r["metric"]) <= 1.0



        print(
            "reliable_success_fraction=%.4f  reliable=%d/%d"
            % (r["metric"], r["n_reliable"], r["n_runs"])
        )
