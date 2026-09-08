#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import grader_core

TESTS_DIR = Path(__file__).resolve().parent
HIDDEN = TESTS_DIR / "hidden_data"
SPLITS = ["intermediate", "final"]
AGENT_SHARDS = [
    "plain_text/train-00000-of-00080.parquet",
    "plain_text/train-00001-of-00080.parquet",
]
SUBMISSION_DIR = Path("/app/output")
METRIC_PATH = Path("/logs/verifier/metric.json")


def _pin_determinism() -> None:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _write_metric(payload: dict) -> None:
    METRIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRIC_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


def test_sealed_splits_are_group_disjoint() -> None:
    """The sealed halves must be disjoint on the canonical group key -- (shard, document
    index) -- not on rows, because every row is a 64-token window OF a document. Asserted
    while both partitions exist, and re-asserted here rather than trusted from build time.
    A duplicate-key fixture is injected so the check cannot pass merely because the real
    ranges happen to be unique."""
    specs = [grader_core.load_split_spec(HIDDEN / s) for s in SPLITS]
    integrity = grader_core.assert_split_disjoint(specs, AGENT_SHARDS)
    assert set(integrity["shards"]).isdisjoint(AGENT_SHARDS)

    duplicated = specs + [dict(specs[0])]
    try:
        grader_core.assert_split_disjoint(duplicated, AGENT_SHARDS)
    except grader_core.SubmissionInvalid as exc:
        assert exc.reason == "sealed_splits_overlap"
    else:
        raise AssertionError("duplicate group keys were not rejected")

    leaked = [dict(specs[0], shard=AGENT_SHARDS[0])]
    try:
        grader_core.assert_split_disjoint(leaked, AGENT_SHARDS)
    except grader_core.SubmissionInvalid as exc:
        assert exc.reason == "sealed_split_uses_agent_shard"
    else:
        raise AssertionError("an agent-visible shard was accepted as a sealed split")


def test_sealed_split_is_derived_at_the_declared_shape() -> None:
    """The graded half derives to exactly the declared shape from the sealed shard, and the
    duplicate scrub that ran at image-build time left a record. A missing or mis-shaped half
    is a task defect and fails closed: grade() raises and the reward is 0."""
    report = json.loads((HIDDEN / "dedup_report.json").read_text())
    assert report["documents_scanned"] > 0
    assert report["cross_shard_duplicates_removed"] >= 0
    split = os.environ.get("GRADED_SPLIT", "final")
    acts = np.load(grader_core.materialize_sealed(HIDDEN / split), mmap_mode="r")
    assert acts.shape == (grader_core.N_SEALED, grader_core.D_MODEL), (split, acts.shape)
    assert acts.dtype == np.float16


def test_reward_map_endpoints() -> None:
    """The reward map honours its contract: invalid is exactly 0, the no-effort floor is
    exactly 0, the strongest trivial baseline is exactly the degenerate constant, the
    baseline score reward is 0.55, and the map is monotone with no cutoff."""
    gr = grader_core.graded_reward
    xs = [1.2, 1.0, 0.8, 0.6, 0.45, 0.30, 0.19, 0.12, 0.05, 0.01]
    rewards = [gr(x, True) for x in xs]


    checks = {
        "invalid_none_is_zero": gr(None, False) == 0.0,
        "invalid_value_is_zero": gr(0.05, False) == 0.0,
        "floor_is_zero": gr(grader_core.M_FLOOR, True) == 0.0,
        "worse_than_floor_is_zero": gr(grader_core.M_FLOOR + 0.5, True) == 0.0,
        "trivial_baseline_is_c":
            abs(gr(grader_core.M0, True) - grader_core.DEGENERATE_REWARD) < 1e-12,
        "baseline_score_reward_is_0.55": abs(gr(grader_core.BASELINE_SCORE, True) - 0.55) < 1e-9,
        "monotone": all(b >= a for a, b in zip(rewards, rewards[1:])),
        "bounded": all(0.0 <= r <= 1.0 for r in rewards),
        "no_saturation_at_one": rewards[-1] < 1.0,
    }
    failed = sorted(k for k, ok in checks.items() if not ok)
    assert not failed, f"reward map violates: {failed}"


def test_grade_submission() -> None:
    """Run the agent's deliverable on the sealed split and write the grade record.

    Every failure path -- missing directory, symlink or hard link in the submitted tree,
    unloadable dictionary, contract violation, crash, timeout, malformed codes -- lands on
    reward exactly 0 through the same record, so no exception can escape and leave
    /logs/verifier/metric.json unwritten.
    """
    _pin_determinism()
    split = os.environ.get("GRADED_SPLIT", "final")
    assert split in SPLITS, f"unknown GRADED_SPLIT={split!r}"
    try:
        result = grader_core.grade(
            split_dir=HIDDEN / split,
            submission_dir=SUBMISSION_DIR,
            all_split_dirs=[HIDDEN / s for s in SPLITS],
            agent_shards=AGENT_SHARDS,
        )
        payload = result.to_json()
    except BaseException as exc:
        payload = {"valid": False, "reason": "grader_internal_error",
                   "detail": f"{type(exc).__name__}: {exc}"[:4000], "reward": 0.0,
                   "metric": None, "split": split}
    payload["graded_split"] = split
    _write_metric(payload)
    print(json.dumps({k: v for k, v in payload.items()
                      if k in ("valid", "reason", "metric", "reward", "mean_l0",
                               "n_atoms", "encode_seconds", "graded_split")},
                     indent=2, default=str))
    assert 0.0 <= float(payload["reward"]) <= 1.0
