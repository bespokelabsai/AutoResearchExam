from __future__ import annotations

import json
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import grader_core
from harness import channel
from harness import hcore
from harness import run_eval

TESTS_DIR = Path(__file__).resolve().parent
SUBMISSION_DIR = Path("/app/output")
METRIC_PATH = Path("/logs/verifier/metric.json")
SPLIT = os.environ.get("HARBOR_GRADED_SPLIT", "final")
SPLIT_IS_KNOWN = SPLIT in grader_core.ANCHORS
HIDDEN_DIR = TESTS_DIR / "hidden_data" / SPLIT
AGENT_UID = 1001
AGENT_GID = 1001
GRADER_MODULES = tuple(
    m for m in (grader_core, hcore, channel, run_eval, sys.modules.get(__name__))
    if m is not None)

random.seed(hcore.SEED_TORCH)
np.random.seed(hcore.SEED_TORCH)
torch.manual_seed(hcore.SEED_TORCH)


def _stage(stage_root: Path) -> tuple[Path, Path, Path]:
    """Copy the harness and the submission out of /tests into a candidate-reachable dir.

    /tests stays 0700 root.  The worker uid 1001 executes a root-owned 0444 copy of the
    harness from a 0755 parent it cannot write.
    """
    os.chmod(stage_root, 0o755)
    harness_dst = stage_root / "harness"
    shutil.copytree(TESTS_DIR / "harness", harness_dst)
    for path in [harness_dst, *harness_dst.rglob("*")]:
        os.chown(path, 0, 0)
        os.chmod(path, 0o755 if path.is_dir() else 0o444)

    solution_dst = stage_root / "solution"
    shutil.copytree(SUBMISSION_DIR, solution_dst, symlinks=True)
    for path in [solution_dst, *solution_dst.rglob("*")]:
        os.lchown(path, AGENT_UID, AGENT_GID)

    work = stage_root / "work"
    work.mkdir()
    os.chmod(work, 0o755)
    return harness_dst / "worker.py", solution_dst, work


def _blank_result(failure_reason: str | None = None) -> dict:
    """The zero-reward shape every downstream reader can rely on."""
    try:
        devices = torch.cuda.device_count()
    except BaseException:
        devices = 0
    return {
        "metric": 0.0,
        "metric_name": "mean_psnr_db_vs_uncached_reference",
        "metric_is_measured": False,
        "reward": 0.0,
        "valid": False,
        "failure_reason": failure_reason,
        "split": SPLIT,
        "per_prompt": [],
        "writable_surface": None,
        "deliverable": None,
        "worker_stderr": "",
        "visible_cuda_devices": devices,
        "grading_seconds": 0.0,
    }


def _write_metric(result: dict) -> None:
    """Populate the reward channel.  Never raises; falls back to primitives only."""
    result["metric"] = float(result.get("metric") or 0.0)
    try:
        METRIC_PATH.parent.mkdir(parents=True, exist_ok=True)
        METRIC_PATH.write_text(json.dumps(result, indent=2))
        return
    except BaseException as exc:
        fallback = {
            "metric": float(result.get("metric") or 0.0),
            "metric_name": "mean_psnr_db_vs_uncached_reference",
            "metric_is_measured": bool(result.get("metric_is_measured")),
            "reward": float(result.get("reward") or 0.0),
            "valid": bool(result.get("valid")),
            "failure_reason": str(result.get("failure_reason")),
            "split": str(SPLIT),
            "report_error": f"{type(exc).__name__}: {exc}",
        }
        try:
            METRIC_PATH.parent.mkdir(parents=True, exist_ok=True)
            METRIC_PATH.write_text(json.dumps(fallback, indent=2))
        except BaseException as exc:
            print(f"could not write {METRIC_PATH}: {type(exc).__name__}: {exc}")


def _grade() -> dict:
    started = time.monotonic()
    result = _blank_result()
    stage_root = None
    try:
        if not SPLIT_IS_KNOWN:
            raise RuntimeError(f"unknown graded split {SPLIT!r}")
        result["writable_surface"] = grader_core.check_writable_surface(
            GRADER_MODULES,
            data_paths=(TESTS_DIR, HIDDEN_DIR, Path(hcore.MODEL_DIR)))
        result["deliverable"] = grader_core.inspect_submission(SUBMISSION_DIR)
        stage_root = Path(tempfile.mkdtemp(prefix="candrun_", dir="/tmp"))
        worker, solution, work = _stage(stage_root)
        report = run_eval.evaluate(
            HIDDEN_DIR, solution, work,
            worker_path=worker, drop_privileges=True, reference_cache=None)
        result["worker_stderr"] = report.get("worker_stderr", "")[-2000:]
        result["per_prompt"] = report.get("prompts", [])
        if not report.get("valid"):
            result["failure_reason"] = report.get("reason") or "the policy run did not complete"
        else:
            metric = float(report["metric"])
            if not np.isfinite(metric):
                result["failure_reason"] = "the computed metric is not finite"
            else:
                result["metric"] = metric
                result["metric_is_measured"] = True
                result["valid"] = True
    except grader_core.SubmissionError as exc:
        result["failure_reason"] = str(exc)
    except BaseException as exc:
        result["failure_reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        if stage_root is not None:
            shutil.rmtree(stage_root, ignore_errors=True)
    try:
        reward = grader_core.graded_reward(
            result["metric"] if result["valid"] else None, result["valid"], SPLIT)
        reward = float(reward)
        if not np.isfinite(reward):
            raise ValueError("the mapped reward is not finite")
    except BaseException as exc:
        result["valid"] = False
        result["failure_reason"] = (
            result["failure_reason"] or f"the reward map failed: {type(exc).__name__}: {exc}")
        reward = 0.0
    result["reward"] = min(1.0, max(0.0, reward)) if result["valid"] else 0.0
    result["grading_seconds"] = round(time.monotonic() - started, 1)
    _write_metric(result)
    return result


_RESULT: dict | None = None


def result() -> dict:
    """Grade once.  Never raises, so a fault cannot re-run the grader per check."""
    global _RESULT
    if _RESULT is None:
        try:
            _RESULT = _grade()
        except BaseException as exc:
            _RESULT = _blank_result(f"the grader failed: {type(exc).__name__}: {exc}")
            _write_metric(_RESULT)
    return _RESULT


def test_metric_json_written():
    """The reward channel is always populated, whatever the submission did."""
    res = result()
    assert METRIC_PATH.is_file()
    print(f"split={res['split']} valid={res['valid']} "
          f"mean_psnr_db={res['metric']:.4f}")
    if not res["valid"]:
        print(f"failure_reason={res['failure_reason']}")


def test_only_the_writable_surface_reaches_the_grader():
    """instruction.md: a solution may change files under /app/output and nowhere else."""
    res = result()
    surface = res["writable_surface"]
    if surface is None:
        raise AssertionError(
            f"the writable-surface check did not pass: {res['failure_reason']}")
    assert surface["writable_root"] == str(SUBMISSION_DIR)
    assert surface["trusted_root"] == str(TESTS_DIR)
    print(f"writable_root={surface['writable_root']} "
          f"trusted_root={surface['trusted_root']} "
          f"modules_checked={surface['modules_checked']}")


def test_deliverable_is_present_and_within_limits():
    """instruction.md: policy.py in /app/output, whole directory at most 256 MiB, no links."""
    res = result()
    if res["deliverable"] is None:
        raise AssertionError(f"deliverable rejected: {res['failure_reason']}")
    assert res["deliverable"]["bytes"] <= grader_core.MAX_DELIVERABLE_BYTES


def test_policy_ran_on_every_sealed_prompt():
    """instruction.md: the policy is run on every prompt of the sealed split."""
    res = result()
    if not res["valid"]:
        raise AssertionError(f"the policy did not complete: {res['failure_reason']}")
    expected = len(json.loads((HIDDEN_DIR / "prompts.json").read_text()))
    assert len(res["per_prompt"]) == expected


def test_unit_budget_respected():
    """instruction.md: at most 62 evaluation units over the 50 steps, on every prompt."""
    res = result()
    if not res["valid"]:
        raise AssertionError(f"the policy did not complete: {res['failure_reason']}")
    for entry in res["per_prompt"]:
        assert entry["units_spent"] <= hcore.BUDGET_UNITS, entry


def test_policy_time_budget_respected():
    """instruction.md: at most 60 s of policy time per prompt."""
    res = result()
    if not res["valid"]:
        raise AssertionError(f"the policy did not complete: {res['failure_reason']}")
    for entry in res["per_prompt"]:
        assert entry["policy_seconds"] <= hcore.POLICY_SECONDS_PER_PROMPT, entry


def test_gpu_is_visible_to_the_grader():
    """An environment fault must be distinguishable from a hard task in a rollout log."""
    res = result()
    print(f"visible_cuda_devices={res['visible_cuda_devices']}")
    assert res["visible_cuda_devices"] >= 1
