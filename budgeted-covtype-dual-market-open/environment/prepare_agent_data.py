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
OUT = Path("/app/data")


def load_source() -> tuple[np.ndarray, np.ndarray]:
    raw = urllib.request.urlopen(URL, timeout=180).read()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError("dataset archive digest mismatch")
    compressed = zipfile.ZipFile(io.BytesIO(raw)).read("covtype.data.gz")
    values = np.loadtxt(gzip.GzipFile(fileobj=io.BytesIO(compressed)), delimiter=",", dtype=np.float32)
    return values[:, :54], values[:, 54].astype(np.int64) - 1


def main() -> None:
    X, y = load_source()
    X, _, y, _ = train_test_split(X, y, train_size=250_000, stratify=y, random_state=0)
    rng = np.random.default_rng(0)
    permutation = rng.permutation(54)
    X = X[:, permutation]
    y = rng.permutation(7)[y]
    X_train, _, y_train, _ = train_test_split(X, y, train_size=0.55, stratify=y, random_state=0)
    intermediate_idx, final_idx = train_test_split(
        np.arange(len(y_train)), train_size=60_000, test_size=60_000,
        stratify=y_train, random_state=73
    )
    remaining = np.setdiff1d(
        np.arange(len(y_train)), np.concatenate((intermediate_idx, final_idx)),
        assume_unique=False,
    )
    public_idx, _ = train_test_split(
        remaining, train_size=15_000, stratify=y_train[remaining], random_state=31
    )
    costs = np.ones(54, dtype=np.float32)
    costs[0] = 3.0
    costs[14:54] = 3.0
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "train_features.npy", X_train[public_idx].astype(np.float32))
    meta = {
        "n_features": 54,
        "n_classes": 7,
        "n_train": 15_000,
        "feature_cost": 6.0,
        "label_cost": 2000.0,
        "costs": costs[permutation].tolist(),
        "dynamic_price": "both",
        "note": "This development pool is unlabeled and is not used for grading.",
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
