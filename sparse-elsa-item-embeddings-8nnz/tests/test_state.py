import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import grader_core

DELIVERABLE = Path("/app/output")
TRAIN_PATH = Path("/tests/data/train.npz")


SPLIT = os.environ.get("GRADE_SPLIT", "final")
EVAL_PATH = Path("/tests/hidden_data") / SPLIT / "eval.npz"
N_EVAL_ROWS = 6250
HARNESS_DIR = Path("/tmp/graded_harness")
WORK_DIR = Path("/tmp/graded_work")
METRIC_PATH = Path("/logs/verifier/metric.json")
PYTHON = "/usr/local/bin/python3"

_RESULT = None


def graded_run():
    global _RESULT
    if _RESULT is None:
        _RESULT = grader_core.grade(
            code_dir=DELIVERABLE,
            train_path=TRAIN_PATH,
            eval_path=EVAL_PATH,
            harness_dir=HARNESS_DIR,
            work_dir=WORK_DIR,
            metric_path=METRIC_PATH,
            python_exe=PYTHON,
            run_as="agent",
            timeout_sec=grader_core.CANDIDATE_TIMEOUT_SEC,
        )
    return _RESULT


def test_graded_run_is_unprivileged_and_bounded():
    """The submitted code must be executed with privilege dropped to the agent uid, detached
    into its own session, and killed at the declared wall-clock budget -- so it can neither read
    the sealed evaluation slice nor outlive its own deadline."""
    cmd = grader_core.candidate_command(
        PYTHON,
        HARNESS_DIR / "run_candidate.py",
        HARNESS_DIR / "candidate_code",
        HARNESS_DIR / "train.npz",
        WORK_DIR / "embedding.npz",
        run_as="agent",
        timeout=grader_core.CANDIDATE_TIMEOUT_SEC,
    )
    assert any(part.endswith("runuser") for part in cmd), cmd
    assert cmd[cmd.index("-u") + 1] == "agent", cmd
    assert "setsid" in cmd and "timeout" in cmd, cmd
    assert "--signal=KILL" in cmd, cmd
    assert str(int(grader_core.CANDIDATE_TIMEOUT_SEC)) in cmd, cmd
    assert not EVAL_PATH.is_relative_to(HARNESS_DIR), EVAL_PATH


def test_entry_point_executes_within_budget():
    """/app/output/solution.py must import, expose build_item_embeddings, and return an
    embedding within the declared wall-clock budget."""
    res = graded_run()
    assert res["stage"] != "execution", res["detail"]


def test_embedding_satisfies_declared_budget():
    """The returned matrix must be 10000 x d with 1 <= d <= 65536, finite, real, and at most
    8 stored nonzeros in every row."""
    res = graded_run()
    assert res["n_dims"] is not None, res["detail"]
    assert 1 <= res["n_dims"] <= grader_core.MAX_DIMS, res["n_dims"]
    assert res["max_nnz_per_item"] <= grader_core.MAX_NNZ_PER_ITEM, res["max_nnz_per_item"]


def test_metric_and_reward_recorded():
    """The mean nDCG@100 over the sealed evaluation rows is recomputed here and recorded with
    the reward it maps to."""
    res = graded_run()
    assert res["valid"], res["detail"]
    assert res["n_eval_rows"] == N_EVAL_ROWS
    assert res["metric"] is not None and 0.0 <= res["metric"] <= 1.0
    assert 0.0 <= res["reward"] <= 1.0
    print(
        f"mean nDCG@100 = {res['metric']:.6f} (sem {res.get('per_row_sem')}) "
        f"over {res['n_eval_rows']} rows; d={res['n_dims']}; "
        f"max nnz/item={res['max_nnz_per_item']}; candidate ran {res['elapsed_sec']}s"
    )
