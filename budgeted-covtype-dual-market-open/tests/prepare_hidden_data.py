from __future__ import annotations

import gzip
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

URL = "https://archive.ics.uci.edu/static/public/31/covertype.zip"
SHA256 = "89a975c2457cd48e824238ae43c5a3cb762e42c4b4078d9b44a4514055105f6d"
ROOT = Path("/tests/hidden_data")


def balance(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = []
    for class_id in range(7):
        candidates = np.flatnonzero(y == class_id)
        indices.extend(rng.choice(candidates, size=min(300, len(candidates)), replace=False).tolist())
    indices = np.asarray(sorted(indices), dtype=np.int64)
    return X[indices], y[indices]


def write_split(path: Path, X_pool: np.ndarray, y_pool: np.ndarray,
                X_test: np.ndarray, y_test: np.ndarray, costs: np.ndarray) -> None:
    np.save(path / "train_features.npy", X_pool.astype(np.float32))
    np.save(path / "train_labels.npy", y_pool.astype(np.int64))
    np.save(path / "test_features.npy", X_test.astype(np.float32))
    np.save(path / "test_labels.npy", y_test.astype(np.int64))
    meta = {
        "n_features": 54,
        "n_classes": 7,
        "n_train": int(len(y_pool)),
        "n_test": int(len(y_test)),
        "feature_cost": 6.0,
        "label_cost": 2000.0,
        "costs": costs.tolist(),
        "dynamic_price": "both",
    }
    (path / "meta.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    raw = urllib.request.urlopen(URL, timeout=180).read()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError("dataset archive digest mismatch")
    compressed = zipfile.ZipFile(io.BytesIO(raw)).read("covtype.data.gz")
    values = np.loadtxt(gzip.GzipFile(fileobj=io.BytesIO(compressed)), delimiter=",", dtype=np.float32)
    X = values[:, :54]
    y = values[:, 54].astype(np.int64) - 1
    X, _, y, _ = train_test_split(X, y, train_size=250_000, stratify=y, random_state=0)
    rng = np.random.default_rng(0)
    permutation = rng.permutation(54)
    X = X[:, permutation]
    y = rng.permutation(7)[y]
    costs = np.ones(54, dtype=np.float32)
    costs[0] = 3.0
    costs[14:54] = 3.0
    costs = costs[permutation]
    X_train, X_hold, y_train, y_hold = train_test_split(
        X, y, train_size=0.55, stratify=y, random_state=0
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_hold, y_hold, train_size=0.5, stratify=y_hold, random_state=0
    )
    intermediate_pool, final_pool = train_test_split(
        np.arange(len(y_train)), train_size=60_000, test_size=60_000,
        stratify=y_train, random_state=73
    )
    X_inter, y_inter = balance(X_val, y_val, seed=20260811)
    X_final, y_final = balance(X_test, y_test, seed=20260812)
    write_split(ROOT / "intermediate", X_train[intermediate_pool], y_train[intermediate_pool],
                X_inter, y_inter, costs)
    write_split(ROOT / "final", X_train[final_pool], y_train[final_pool],
                X_final, y_final, costs)


if __name__ == "__main__":
    main()
