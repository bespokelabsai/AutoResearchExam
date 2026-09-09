from __future__ import annotations

import gzip
import json
import math
import os
import pwd
import shutil
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import runner
from worker_client import CALL_BUDGET_SEC, IMPORT_BUDGET_SEC

TESTS_DIR = Path(__file__).resolve().parent
HIDDEN = TESTS_DIR / "hidden_data"
PYTHON = sys.executable
DELIVERABLE_DIR = Path("/app/output")
ENTRY_MODULE = "prune"
STAGE_ROOT = Path("/grade_stage")
CANDIDATE_USER = "agent"

DRAWS = ("s0", "s1")
SEEDS = tuple(range(12))

M0 = 77.9
BASELINE_SCORE = 93.11


class SubmissionInvalid(Exception):
    """The submission cannot be scored: it is missing, unsafe, crashed, or out of budget."""


def scan_for_links(root: Path):
    """Reject every symlink and every hard-linked file anywhere under the submitted tree,
    BEFORE anything is copied.  A submitted directory symlink would otherwise be recursed
    into by the copy and materialise verifier-owned files where the score is computed from."""
    offences = []
    stack = [root]
    while stack:
        current = stack.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                info = os.lstat(entry.path)
                rel = os.path.relpath(entry.path, root)
                if os.path.islink(entry.path):
                    offences.append(f"symlink: {rel}")
                    continue
                if os.path.isdir(entry.path):
                    stack.append(Path(entry.path))
                    continue
                if info.st_nlink != 1:
                    offences.append(f"hard link (st_nlink={info.st_nlink}): {rel}")
    return offences


def stage_deliverable(source: Path, dest: Path):
    if not source.is_dir():
        raise SubmissionInvalid(f"no deliverable directory at {source}")
    entry = source / (ENTRY_MODULE + ".py")
    if not entry.is_file() or entry.stat().st_size == 0:
        raise SubmissionInvalid(f"missing or empty entry point {entry}")
    offences = scan_for_links(source)
    if offences:
        raise SubmissionInvalid("submitted tree contains links: " + "; ".join(offences[:5]))
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(source, dest, symlinks=True)
    os.lchown(dest, 0, 0)
    for base, dirs, files in os.walk(dest):
        for name in dirs + files:
            os.lchown(os.path.join(base, name), 0, 0)
    return runner.make_readonly(dest)


def load_pool_texts(draw):
    with gzip.open(HIDDEN / "pools" / f"pool_{draw}.jsonl.gz", "rt", encoding="utf-8") as fh:
        return [json.loads(line)["text"] for line in fh]


def build_specs(phase, stage_root, seeds=SEEDS):
    eval_y = np.load(HIDDEN / phase / "eval_y.npy")
    specs = []
    for draw in DRAWS:
        pool_texts_path = runner.stage_pool_texts(
            load_pool_texts(draw), Path(stage_root) / f"pool_{draw}.json")
        pool_labels = np.load(HIDDEN / "pools" / f"pool_{draw}_y.npy")
        pool_X = np.load(HIDDEN / "pools" / f"pool_{draw}_X.npy")
        eval_X = np.load(HIDDEN / phase / f"eval_{draw}_X.npy")
        for seed in seeds:
            specs.append({"draw": draw, "seed": seed, "pool_labels": pool_labels,
                          "pool_X": pool_X, "eval_y": eval_y, "eval_X": eval_X,
                          "pool_texts_path": pool_texts_path})
    return specs


def graded_reward(metric, valid):
    """rational_squash: 0 at the strongest trivial baseline, 0.5 at the baseline score,
    strictly increasing above it, asymptotic to 1, never negative."""
    if not valid or metric is None or not math.isfinite(metric):
        return 0.0
    x_ref = BASELINE_SCORE - M0
    assert x_ref > 0, "malformed task: the baseline score does not beat the trivial baseline"
    u = max(0.0, metric - M0) / x_ref
    return u / (1.0 + u)


def grade(phase, launch_prefix, new_session=os.setsid, call_timeout=CALL_BUDGET_SEC,
          import_timeout=IMPORT_BUDGET_SEC, deliverable=DELIVERABLE_DIR,
          stage_root=STAGE_ROOT, seeds=SEEDS):
    """Run the sealed panel and return the record written to /logs/verifier/metric.json."""
    record = {"phase": phase, "valid": False, "metric": None, "reward": 0.0,
              "n_runs": len(DRAWS) * len(seeds), "detail": "", "per_iteration_mean": None,
              "run_sd": None, "run_min": None, "run_max": None, "slowest_call_sec": None}

    stage_root = Path(stage_root)
    stage_root.mkdir(parents=True, exist_ok=True)
    os.chmod(stage_root, 0o755)
    scratch_root = stage_root / "scratch"
    err_root = stage_root / "worker_stderr"
    err_root.mkdir(parents=True, exist_ok=True)
    module_dir = stage_root / "deliverable"

    worker_script = stage_root / "prune_worker.py"
    shutil.copyfile(TESTS_DIR / "prune_worker.py", worker_script)
    os.chown(worker_script, 0, 0)
    os.chmod(worker_script, 0o644)

    try:
        stage_deliverable(Path(deliverable), module_dir)
        specs = build_specs(phase, stage_root, seeds)
        results = runner.execute_runs(
            specs, python_exe=PYTHON,
            worker_script=str(worker_script),
            module_dir=str(module_dir), entry_module=ENTRY_MODULE,
            scratch_root=str(scratch_root), as_user=CANDIDATE_USER,
            launch_prefix=launch_prefix, new_session=new_session, call_budget=call_timeout,
            import_budget=import_timeout, stderr_root=str(err_root))
    except SubmissionInvalid as exc:
        record["detail"] = str(exc)
        return record

    failures = [r for r in results if r["error"]]
    timings = [t for r in results for t in r["timings"]]
    record["slowest_call_sec"] = round(max(timings), 3) if timings else None
    if failures:
        record["detail"] = (f"{len(failures)}/{len(results)} runs failed; first: "
                            f"{failures[0]['error'][:800]}")
        return record

    finals = [r["final"] for r in results]
    per_draw = {}
    for r in results:
        per_draw.setdefault(r["draw"], []).append(r["final"])
    record["per_draw_mean"] = {d: statistics.fmean(v) for d, v in per_draw.items()}
    record["per_run_final"] = [round(v, 6) for v in finals]
    record["valid"] = True
    record["metric"] = statistics.fmean(finals)
    record["run_sd"] = statistics.stdev(finals)
    record["run_min"] = min(finals)
    record["run_max"] = max(finals)
    record["per_iteration_mean"] = [round(float(v), 4) for v in
                                    np.mean(np.array([r["curve"] for r in results]), axis=0)]
    record["reward"] = graded_reward(record["metric"], True)
    record["detail"] = "all runs completed"
    return record
