from __future__ import annotations

import math
import os
import shutil
import stat
from pathlib import Path

import numpy as np

N_TRAIN = 672
N_TEST = 75
N_COVARIATES = 25

GRADED_RUN_BUDGET_SEC = 1200.0

M0 = 5.40
BASELINE_SCORE = 0.58

STATUS_OK = 0
STATUS_RAISED = 1
STATUS_BAD_SHAPE = 2
STATUS_NON_FINITE = 3


def load_sealed_panel(panel_path: str | os.PathLike) -> tuple[list[dict], np.ndarray]:
    """Return (inputs, taus). `inputs` carries only what a candidate may see."""
    with np.load(panel_path, allow_pickle=False) as f:
        n = int(f["n_realizations"])
        inputs, taus = [], []
        for k in range(n):
            inputs.append({
                "X_train": f[f"X_train_{k:03d}"].astype(np.float64),
                "t_train": f[f"t_train_{k:03d}"].astype(np.float64),
                "y_train": f[f"y_train_{k:03d}"].astype(np.float64),
                "X_test": f[f"X_test_{k:03d}"].astype(np.float64),
            })
            taus.append(f[f"tau_test_{k:03d}"].astype(np.float64))
    for d, tau in zip(inputs, taus):
        assert d["X_train"].shape == (N_TRAIN, N_COVARIATES), "sealed panel geometry changed"
        assert d["X_test"].shape == (N_TEST, N_COVARIATES), "sealed panel geometry changed"
        assert tau.shape == (N_TEST,), "sealed panel geometry changed"
    return inputs, np.asarray(taus, dtype=np.float64)


def pehe(tau_true: np.ndarray, tau_hat: np.ndarray) -> float:
    """Root-mean-squared error of the estimated effect over one realization's test units."""
    return float(np.sqrt(np.mean((np.asarray(tau_true, dtype=np.float64)
                                  - np.asarray(tau_hat, dtype=np.float64)) ** 2)))


def constant_baseline_tau(one_input: dict) -> float:
    """The no-covariate constant effect: mean(y | t=1) - mean(y | t=0) on the training rows.

    Recomputed here by the verifier from trusted bytes. A call that fails is scored as if it
    had returned this constant, which is exactly what a submission gets by *returning* the
    constant, so a failed call can never beat an honest one.
    """
    t = one_input["t_train"]
    y = one_input["y_train"]
    treated, control = y[t == 1.0], y[t == 0.0]
    if treated.size == 0 or control.size == 0:
        return 0.0
    return float(treated.mean() - control.mean())


def score_panel(inputs: list[dict], taus: np.ndarray, preds: np.ndarray,
                status: np.ndarray) -> dict:
    """Per-realization PEHE with the constant-baseline fallback, then the panel mean.

    `preds` and `status` come from the candidate's process and are treated as untrusted
    data: shape, dtype and finiteness are re-checked here, and any realization that fails
    that re-check is scored by the fallback regardless of what status it claims.
    """
    n = len(inputs)
    per, n_ok, n_fallback = [], 0, 0
    for k in range(n):
        row = preds[k] if (preds.ndim == 2 and preds.shape[0] > k) else None
        ok = (row is not None and row.shape == (N_TEST,) and np.all(np.isfinite(row))
              and int(status[k]) == STATUS_OK)
        if ok:
            per.append(pehe(taus[k], row))
            n_ok += 1
        else:
            fallback = np.full(N_TEST, constant_baseline_tau(inputs[k]))
            per.append(pehe(taus[k], fallback))
            n_fallback += 1
    per_arr = np.asarray(per, dtype=np.float64)
    return {
        "metric": float(per_arr.mean()),
        "n_realizations": n,
        "n_ok": n_ok,
        "n_fallback": n_fallback,
        "pehe_median": float(np.median(per_arr)),
        "pehe_p90": float(np.quantile(per_arr, 0.9)),
        "pehe_max": float(per_arr.max()),
    }


def graded_reward(metric: float | None, valid: bool) -> float:
    """Continuous reward in [0, 1]; lower mean PEHE is strictly better, with no cutoff.

    Log-ratio improvement in baseline-score units, squashed by u/(1+u):

        u = max(0, log(m0 / m) / log(m0 / baseline_score)),   reward = u / (1 + u)

    m >= m0 (the strongest trivial predictor) -> 0.  m == baseline_score -> 0.5.
    m -> 0 -> 1. PEHE is ratio-scale with a true zero and improves multiplicatively, so the
    improvement is measured as a log ratio.
    """
    if not valid or metric is None:
        return 0.0
    m = float(metric)
    if not math.isfinite(m) or m < 0.0:
        return 0.0
    scale = math.log(M0 / BASELINE_SCORE)
    assert scale > 0.0, "malformed task: the baseline anchor does not beat the trivial baseline"
    if m <= 0.0:
        return 1.0
    u = max(0.0, math.log(M0 / m) / scale)
    return u / (1.0 + u)


def copy_solution_tree(src: str | os.PathLike, dst: str | os.PathLike) -> dict:
    """Copy the submitted tree into a root-owned, agent-read-only sandbox.

    Every submitted path is treated as hostile. Symlinks, hard links and non-regular files
    are pruned at every depth BEFORE anything is copied, and the copy itself never follows
    a link (`shutil.copyfile(..., follow_symlinks=False)`, the equivalent of
    `cp --no-dereference` / `copytree(symlinks=True)`), so a submitted
    `estimator.py -> /tests/hidden_data/final/panel.npz` cannot materialise verifier bytes
    inside the sandbox the run reads from.
    """
    src, dst = Path(src), Path(dst)
    pruned: list[str] = []
    dst.mkdir(parents=True, exist_ok=True)
    for root, dirs, files in os.walk(src, followlinks=False):
        rel = Path(root).relative_to(src)
        kept_dirs = []
        for d in sorted(dirs):
            p = Path(root) / d
            if p.is_symlink():
                pruned.append(str(p))
                continue
            kept_dirs.append(d)
            (dst / rel / d).mkdir(parents=True, exist_ok=True)
        dirs[:] = kept_dirs
        for f in sorted(files):
            p = Path(root) / f
            st = p.lstat()
            if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode) or st.st_nlink > 1:
                pruned.append(str(p))
                continue
            shutil.copyfile(p, dst / rel / f, follow_symlinks=False)
    for p in [dst, *sorted(dst.rglob("*"))]:
        os.chown(p, 0, 0)
        os.chmod(p, 0o755 if p.is_dir() else 0o644)
    return {"pruned": pruned}


def write_runner_inputs(inputs: list[dict], path: str | os.PathLike) -> None:
    """Serialize the grade-time inputs for the candidate process. Carries NO targets."""
    arrays: dict[str, np.ndarray] = {"n_realizations": np.array(len(inputs))}
    for k, d in enumerate(inputs):
        arrays[f"X_train_{k:03d}"] = d["X_train"]
        arrays[f"t_train_{k:03d}"] = d["t_train"]
        arrays[f"y_train_{k:03d}"] = d["y_train"]
        arrays[f"X_test_{k:03d}"] = d["X_test"]
    np.savez(path, **arrays)
    os.chmod(path, 0o644)
