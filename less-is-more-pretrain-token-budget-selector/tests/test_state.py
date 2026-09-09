from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import grader_core
import pretrain_harness

METRIC_PATH = Path("/logs/verifier/metric.json")
SPLIT = os.environ.get("HIDDEN_SPLIT", "final")
HIDDEN = Path(__file__).resolve().parent / "hidden_data" / SPLIT
CONFIG = Path(__file__).resolve().parent / "train_config.json"
SESSION = Path("/tmp/grade-session")


def _write(record: dict) -> None:
    METRIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRIC_PATH.write_text(json.dumps(record, indent=2, sort_keys=True, default=str))


def test_split_integrity():
    """The graded pool and the graded evaluation shard share no document.

    Asserted here, at grade time, on the shipped bytes -- not only in the builder that
    produced them.  The duplicate-key fixture rows make the check meaningful: a set-based
    intersection over unique keys cannot detect grouped leakage, so the fixture below
    deliberately repeats a key on each side and the assertion still has to hold.
    """
    guard = json.loads((HIDDEN / "eval" / "guard.json").read_text())
    eval_hashes = set(guard["hashes"])
    eval_urls = set(guard["urls"])
    pool_meta = json.loads((HIDDEN / "pool" / "meta.json").read_text())
    pool_guard = json.loads((HIDDEN / "pool" / "guard.json").read_text())
    pool_hashes = set(pool_guard["hashes"])
    pool_urls = set(pool_guard["urls"])
    assert pool_hashes, "the graded pool shipped no document hashes"
    assert eval_hashes, "the graded evaluation shard shipped no document hashes"
    assert not (pool_hashes & eval_hashes), "pool/eval text overlap"
    assert not (pool_urls & eval_urls), "pool/eval URL overlap"
    assert pool_meta["n_tokens"] > 200_000_000, pool_meta

    fix_pool = ["k1", "k1", "k2", "k3", "k3"]
    fix_eval = ["k4", "k4", "k5"]
    assert not (set(fix_pool) & set(fix_eval))
    assert len(fix_pool) != len(set(fix_pool)) and len(fix_eval) != len(set(fix_eval))
    leaky_eval = ["k2", "k2", "k9"]
    assert set(fix_pool) & set(leaky_eval), "the disjointness assertion cannot detect leakage"


def test_writable_surface():
    """The verifier reads exactly the files instruction.md lets a solution change.

    instruction.md, "Files you may change", names `/app/output/select.py` plus any further file
    under `/app/output/` as the complete writable surface, and states that the harness, the
    config and everything else under `/app` are development material the grader never reads.
    This asserts the verifier honours that same list: the deliverable it stages is `/app/output`
    and its entry point is `select.py`; the harness, the config and the graded data come from
    the sealed tree beside this file; and no agent-editable path is reachable as an import or
    present in this image.  It scores nothing -- no submission can influence it.
    """
    sealed = Path(__file__).resolve().parent
    assert grader_core.DELIVERABLE_DIR == "/app/output"
    assert grader_core.ENTRY_POINT == "select.py"
    assert grader_core.WRITABLE_SURFACE_ROOT == grader_core.DELIVERABLE_DIR
    assert Path(pretrain_harness.__file__).resolve().parent == sealed
    assert CONFIG.is_file() and CONFIG.resolve().parent == sealed
    assert str(HIDDEN.resolve()).startswith(str(sealed) + os.sep)
    for name in ("pool", "eval"):
        assert (HIDDEN / name).is_dir(), f"the sealed {name} is missing"
    assert grader_core.writable_surface_violations(CONFIG) == []


def test_grade():
    """Execute the submission, recompute the metric from the sealed slice, write it out."""
    record = {"reward": 0.0, "valid": False, "invalid_reason": "grader did not complete",
              "split": SPLIT}
    try:
        try:
            import torch
            record["grader_cuda_device_count"] = int(torch.cuda.device_count())
            record["grader_cuda_available"] = bool(torch.cuda.is_available())
        except Exception as exc:
            record["grader_cuda_device_count"] = -1
            record["infra_error"] = f"the grader itself cannot reach CUDA: {exc}"

        raw = grader_core.grade(
            deliverable_src=grader_core.DELIVERABLE_DIR,
            pool_dir=HIDDEN / "pool",
            eval_dir=HIDDEN / "eval",
            config_path=CONFIG,
            session_dir=SESSION,
            log=lambda msg: print(msg, flush=True),
        )
        raw["split"] = SPLIT
        record = grader_core.finalize(raw)
    except BaseException:
        record = {"reward": 0.0, "valid": False, "split": SPLIT,
                  "invalid_reason": "grader raised",
                  "traceback": traceback.format_exc()[-6000:]}
    finally:
        _write(record)

    print(json.dumps({k: v for k, v in record.items()
                      if k not in ("select_stderr_tail", "traceback", "grade_traceback")},
                     indent=2, sort_keys=True, default=str), flush=True)
    assert isinstance(record["reward"], float) and 0.0 <= record["reward"] <= 1.0
