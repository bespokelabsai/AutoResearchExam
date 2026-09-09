from __future__ import annotations

import json
import os
import shutil
import stat
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

TESTS_DIR = Path(__file__).resolve().parent
DELIVERABLE_DIR = Path("/app/output")
ENTRY_POINT = DELIVERABLE_DIR / "solution.py"
STAGE_DIR = Path("/grade_stage")
WORK_DIR = Path("/grade_work")
LOGS_DIR = Path("/logs/verifier")
RUNNER = TESTS_DIR / "runner.py"

HALF = os.environ.get("GRADE_SPLIT", "final")
HIDDEN_ROOT = TESTS_DIR / "hidden_data"
HIDDEN_DIR = HIDDEN_ROOT / HALF
SHARED_DIR = HIDDEN_ROOT / "shared"

FEATURES = [f"f{i}" for i in range(9)]
LABEL = "y"

SEEDS = tuple(range(12))
N_TRAIN_EXTRA = 3_500
N_UNLABELED = 10_300
N_EVAL = 5_000
N_TRAIN = 8_500
N_POOL = 15_300

SEED_TIMEOUT_SEC = 100.0
MAX_DELIVERABLE_BYTES = 131_072

TOTAL_BUDGET_SEC = 1_200.0

FAILED_SEED_AUC = 0.5

METRIC_QUANTUM = 1e-4

STDERR_TAIL_BYTES = 2_000

M0 = 0.5800
BASELINE_SCORE = 0.9172


def graded_reward(metric: float, valid: bool) -> float:
    """rational_squash over the panel-mean ROC-AUC.  Higher is better."""
    if not valid:
        return 0.0
    x_ref = BASELINE_SCORE - M0
    assert x_ref > 0, "malformed task: the anchor does not beat the trivial baseline"
    u = max(0.0, metric - M0) / x_ref
    return u / (1.0 + u)


def load_half(half_dir: Path = HIDDEN_DIR) -> dict[str, pd.DataFrame]:
    out = {}
    for name, source_dir in (
        ("dev", SHARED_DIR),
        ("reservoir", SHARED_DIR),
        ("evalpool", half_dir),
    ):
        df = pd.read_csv(source_dir / f"{name}.csv")
        assert list(df.columns) == FEATURES + [LABEL], (name, list(df.columns))
        out[name] = df
    return out


def _stratified_take(y: np.ndarray, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (taken, remaining) row-index arrays, stratified on y."""
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=seed)
    taken, remaining = next(sss.split(np.zeros(len(y)), y))
    return taken, remaining


def build_panel_inputs(tables: dict[str, pd.DataFrame], seed: int) -> dict[str, object]:
    """Deterministically build one seed's (train, pool, eval) triple.

    The sealed reservoir is partitioned into labelled-extra / unlabelled row blocks
    first and rows are drawn from those blocks; the evaluation rows come from a
    disjoint sealed pool that never supplies a training row.  ``eval_y`` is returned
    to the trusted caller only and is never written anywhere the submitted code can
    read.
    """
    dev, res, ep = tables["dev"], tables["reservoir"], tables["evalpool"]

    res_y = res[LABEL].to_numpy()
    extra_idx, rest_idx = _stratified_take(res_y, N_TRAIN_EXTRA, seed)
    unl_rel, _ = _stratified_take(res_y[rest_idx], N_UNLABELED, seed + 5_000)
    unl_idx = rest_idx[unl_rel]

    ev_idx, _ = _stratified_take(ep[LABEL].to_numpy(), N_EVAL, seed + 9_000)

    train_X = pd.concat(
        [dev[FEATURES], res.iloc[extra_idx][FEATURES]], ignore_index=True
    )
    train_y = np.concatenate(
        [
            dev[LABEL].to_numpy(dtype=np.int8),
            res.iloc[extra_idx][LABEL].to_numpy(dtype=np.int8),
        ]
    )
    eval_X = ep.iloc[ev_idx][FEATURES].reset_index(drop=True)
    eval_y = ep.iloc[ev_idx][LABEL].to_numpy(dtype=np.int8)

    pool = pd.concat([res.iloc[unl_idx][FEATURES], eval_X], ignore_index=True)
    rng = np.random.default_rng(1_000_003 + seed)
    pool = pool.iloc[rng.permutation(len(pool))].reset_index(drop=True)

    assert train_X.shape == (N_TRAIN, 9), train_X.shape
    assert pool.shape == (N_POOL, 9), pool.shape
    assert eval_X.shape == (N_EVAL, 9), eval_X.shape
    assert len(train_y) == N_TRAIN and set(np.unique(train_y)) <= {0, 1}
    return {
        "train_X": train_X,
        "train_y": train_y,
        "pool_X": pool,
        "eval_X": eval_X,
        "eval_y": eval_y,
    }


def write_inputs(path: Path, inputs: dict[str, object], seed: int) -> None:
    """Serialise the feature-only view of one seed for the runner subprocess."""
    np.savez(
        path,
        train_X=inputs["train_X"].to_numpy(dtype=np.int64),
        train_y=inputs["train_y"].astype(np.int8),
        pool_X=inputs["pool_X"].to_numpy(dtype=np.int64),
        eval_X=inputs["eval_X"].to_numpy(dtype=np.int64),
        columns=np.array(FEATURES),
        seed=np.int64(seed),
    )


class DeliverableError(RuntimeError):
    """The submission is structurally unusable: the run scores exactly 0."""


def stage_deliverable() -> tuple[Path, dict[str, object]]:
    """Copy /app/output into a root-owned staging tree, hostile-input first.

    Every path is checked with ``lstat`` at every depth BEFORE anything is copied, and
    the copy itself never follows a link (``symlinks=True``).  A symlink, a hard link,
    a device/fifo/socket, an over-size tree or a missing entry point is a hard failure,
    not a warning: without the pre-scan a submitted
    ``solution.py -> /tests/hidden_data/final/evalpool.csv`` symlink would be
    materialised into a location the score is computed from.
    """
    if not DELIVERABLE_DIR.is_dir():
        raise DeliverableError(f"no deliverable directory at {DELIVERABLE_DIR}")

    total = 0
    n_files = 0
    for root, dirs, files in os.walk(DELIVERABLE_DIR, followlinks=False):
        for name in dirs + files:
            p = Path(root) / name
            st = os.lstat(p)
            if stat.S_ISLNK(st.st_mode):
                raise DeliverableError(f"symlink in deliverable: {p}")
            if not (stat.S_ISDIR(st.st_mode) or stat.S_ISREG(st.st_mode)):
                raise DeliverableError(f"not a regular file or directory: {p}")
            if stat.S_ISREG(st.st_mode):
                if st.st_nlink > 1:
                    raise DeliverableError(f"hard link in deliverable: {p}")
                total += st.st_size
                n_files += 1
    if total > MAX_DELIVERABLE_BYTES:
        raise DeliverableError(
            f"deliverable is {total} bytes, over the {MAX_DELIVERABLE_BYTES} byte limit"
        )
    if not ENTRY_POINT.is_file() or ENTRY_POINT.stat().st_size == 0:
        raise DeliverableError(f"missing or empty entry point {ENTRY_POINT}")

    if STAGE_DIR.exists():
        shutil.rmtree(STAGE_DIR)
    shutil.copytree(DELIVERABLE_DIR, STAGE_DIR, symlinks=True)
    for p in [STAGE_DIR, *STAGE_DIR.rglob("*")]:
        os.chown(p, 0, 0)
        os.chmod(p, 0o755 if p.is_dir() else 0o644)
    return STAGE_DIR, {"deliverable_bytes": total, "deliverable_files": n_files}


def score_output(out_path: Path, eval_y: np.ndarray) -> tuple[str, float]:
    """Validate and score one seed's output array.  Never executes candidate code."""
    if not out_path.is_file():
        return "no_output", FAILED_SEED_AUC
    try:
        scores = np.load(out_path, allow_pickle=False)
    except Exception as exc:
        return f"unloadable_output:{type(exc).__name__}", FAILED_SEED_AUC
    if scores.dtype.kind not in "fiub" or scores.ndim != 1 or scores.shape != (N_EVAL,):
        return f"bad_shape_or_dtype:{scores.shape}:{scores.dtype}", FAILED_SEED_AUC
    scores = scores.astype(np.float64)
    if not np.all(np.isfinite(scores)):
        return "non_finite_output", FAILED_SEED_AUC
    return "ok", float(roc_auc_score(eval_y, scores))


def aggregate(records: list[dict], extra: dict) -> dict:
    """Assemble the metric record, including the reward, from per-seed results."""
    aucs = [float(r["auc"]) for r in records]
    n_ok = sum(1 for r in records if r.get("status") == "ok")
    raw = float(np.mean(aucs))
    metric = round(raw / METRIC_QUANTUM) * METRIC_QUANTUM
    valid = n_ok > 0
    result = {
        "half": HALF,
        "seeds": list(SEEDS),
        "metric_name": "mean_roc_auc_12_seed_panel",
        "seed_timeout_sec": SEED_TIMEOUT_SEC,
        "total_budget_sec": TOTAL_BUDGET_SEC,
        "valid": valid,
        "seeds_ok": n_ok,
        "seeds_failed": len(records) - n_ok,
        "per_seed": [
            {k: v for k, v in r.items() if k in ("seed", "status", "auc", "elapsed_sec")}
            for r in records
        ],
        "failure_detail": [
            {
                "seed": r["seed"],
                "status": r.get("status"),
                "stderr_tail": r.get("stderr_tail", ""),
            }
            for r in records
            if r.get("status") != "ok"
        ],
        "metric_raw": raw,
        "metric": metric if valid else None,
        "reward": graded_reward(metric, valid),
        "wall_clock_sec": round(sum(float(r.get("elapsed_sec", 0.0)) for r in records), 2),
    }
    result.update(extra)
    if not valid:
        result["invalid_reason"] = "every seed failed to produce a usable score array"
    return result


def invalid_result(reason: str) -> dict:
    """The structural-failure record: reward is exactly 0, no arithmetic involved."""
    return {
        "half": HALF,
        "seeds": list(SEEDS),
        "metric_name": "mean_roc_auc_12_seed_panel",
        "valid": False,
        "invalid_reason": reason,
        "seeds_ok": 0,
        "seeds_failed": len(SEEDS),
        "metric": None,
        "reward": 0.0,
    }


def write_metric(result: dict) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    (LOGS_DIR / "metric.json").write_text(json.dumps(result, indent=2, default=str))
