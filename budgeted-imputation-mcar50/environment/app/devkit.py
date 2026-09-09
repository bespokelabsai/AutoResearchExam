from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cellspec

DEV_ROOT = Path("/app/data/dev")
DEFAULT_MODULE_DIR = Path("/app/output")
CELLRUNNER = Path(__file__).resolve().parent / "cellrunner.py"

CELL_BUDGET_SEC = 30.0


def manifest() -> dict:
    with open(DEV_ROOT / "manifest.json") as fh:
        return json.load(fh)


def _entry(dataset_id: str) -> dict:
    for entry in manifest()["datasets"]:
        if entry["id"] == dataset_id:
            return entry
    raise KeyError(f"unknown dev dataset {dataset_id!r}")


def load_dev_cell(dataset_id: str, seed_index: int = 0):
    """Return (X_train, X_eval, X_eval_true, mask_eval) for one development cell."""
    entry = _entry(dataset_id)
    with np.load(DEV_ROOT / entry["file"], allow_pickle=False) as data:
        X = np.ascontiguousarray(data["X"], dtype=np.float64)
    return cellspec.build_cell(
        X, int(entry["seeds"][seed_index]), entry["n_train"], entry["n_eval"]
    )


def _score_cell(module_dir: Path, X_train, X_eval, X_eval_true, mask_eval, seed):
    """Run the submitted module on one cell. Returns (score_or_None, reason, seconds)."""
    env = dict(os.environ)
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONSAFEPATH": "1",
        }
    )
    with tempfile.TemporaryDirectory() as tmp:
        env["TMPDIR"] = tmp
        env["HOME"] = tmp
        in_path = os.path.join(tmp, "in.npz")
        out_path = os.path.join(tmp, "out.npy")
        np.savez(in_path, X_train=X_train, X_eval=X_eval, seed=np.array([seed], dtype=np.int64))
        argv = [sys.executable, str(CELLRUNNER), in_path, out_path, str(module_dir)]
        started = time.monotonic()
        try:
            proc = subprocess.run(
                argv,
                cwd=tmp,
                env=env,
                capture_output=True,
                text=True,
                timeout=CELL_BUDGET_SEC,
                start_new_session=True,
            )
        except subprocess.TimeoutExpired:
            return None, "timeout", time.monotonic() - started
        elapsed = time.monotonic() - started
        if proc.returncode != 0:
            tail = (proc.stderr or "").strip().splitlines()[-3:]
            return None, "error: " + " | ".join(tail), elapsed
        if not os.path.isfile(out_path):
            return None, "no output", elapsed
        try:
            arr = np.load(out_path, allow_pickle=False)
        except Exception as exc:
            return None, f"unreadable output: {exc}", elapsed

    if not isinstance(arr, np.ndarray) or arr.dtype.kind != "f":
        return None, f"bad dtype {getattr(arr, 'dtype', type(arr))}", elapsed
    if arr.shape != X_eval_true.shape:
        return None, f"bad shape {arr.shape} != {X_eval_true.shape}", elapsed
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        return None, "non-finite value returned", elapsed
    return cellspec.imputation_r2(arr, X_eval_true, mask_eval), "ok", elapsed


def main() -> int:
    ap = argparse.ArgumentParser(description="run the development panel")
    ap.add_argument("--list", action="store_true", help="show the development datasets")
    ap.add_argument("--datasets", default="", help="comma-separated dataset ids")
    ap.add_argument("--seeds", default="", help="comma-separated seed indices")
    ap.add_argument("--module", default=str(DEFAULT_MODULE_DIR), help="directory holding impute.py")
    args = ap.parse_args()

    entries = manifest()["datasets"]
    if args.list:
        print(f"{'id':6s} {'dataset':22s} {'rows':>6s} {'d':>4s} {'n_train':>8s} {'n_eval':>7s} seeds")
        for e in entries:
            print(
                f"{e['id']:6s} {e['source']:22s} {e['n_rows']:6d} {e['n_features']:4d} "
                f"{e['n_train']:8d} {e['n_eval']:7d} {len(e['seeds'])}"
            )
        return 0

    wanted = [s for s in args.datasets.split(",") if s] or [e["id"] for e in entries]
    module_dir = Path(args.module).resolve()
    if not (module_dir / "impute.py").is_file():
        print(f"no impute.py under {module_dir}", file=sys.stderr)
        return 2

    panel = []
    print(f"{'dataset':10s} {'seed':>5s} {'seconds':>8s}  R2")
    for dataset_id in wanted:
        entry = _entry(dataset_id)
        indices = [int(s) for s in args.seeds.split(",") if s] or list(range(len(entry["seeds"])))
        per_dataset = []
        with np.load(DEV_ROOT / entry["file"], allow_pickle=False) as data:
            X = np.ascontiguousarray(data["X"], dtype=np.float64)
        for k in indices:
            seed = int(entry["seeds"][k])
            X_train, X_eval, X_eval_true, mask_eval = cellspec.build_cell(
                X, seed, entry["n_train"], entry["n_eval"]
            )
            score, reason, elapsed = _score_cell(
                module_dir, X_train, X_eval, X_eval_true, mask_eval, seed
            )
            shown = "0.0000  <- scored as zero: " + reason if score is None else f"{score:.4f}"
            print(f"{dataset_id:10s} {k:5d} {elapsed:8.2f}  {shown}")
            per_dataset.append(0.0 if score is None else score)
        if per_dataset:
            spread = max(per_dataset) - min(per_dataset)
            print(
                f"{dataset_id:10s} {'mean':>5s} {'':>8s}  {np.mean(per_dataset):.4f} "
                f"(spread over seeds {spread:.4f})"
            )
            panel.extend(per_dataset)
    if panel:
        print(f"\npanel mean over {len(panel)} cells: {np.mean(panel):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
