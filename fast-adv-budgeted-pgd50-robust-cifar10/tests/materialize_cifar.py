#!/usr/bin/env python3
import argparse
import hashlib
import io
import sys

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

PARQUET_SHA256 = {
    "train": "8428b53a88a11ac374111006708df51469e315a22ac6d66470afd9c78d2ae883",
    "test": "841389e6f2d64f28bf17310e430aebac20ec3ba611a3c5e231dc93c645ce84de",
}
CONTENT_SHA256 = {
    "train": {
        "images": "c5b8e1d062f7373a5bc758689b1b9c2a7caf093109b95ceb98cecaf2bd266807",
        "labels": "d6d2fe5c7528f766adb2e13bf983d1e4707936c06b4ccf240ef3d2dcd72c986d",
    },
    "test": {
        "images": "f9b1e2130bab680db56eb06d886f7f1ad0f1183907ceff4659d1735c890f5b04",
        "labels": "cbb7365de8ed11f05cc4c3a1e7f78144127c5e851efd83762fb18202461230bb",
    },
}
ROWS = {"train": 50000, "test": 10000}
PER_CLASS = {"train": 5000, "test": 1000}


def decode(parquet_path, split):
    digest = hashlib.sha256(open(parquet_path, "rb").read()).hexdigest()
    if digest != PARQUET_SHA256[split]:
        raise SystemExit(
            f"{parquet_path}: sha256 {digest} != pinned {PARQUET_SHA256[split]}"
        )
    table = pq.read_table(parquet_path)
    if table.schema.names != ["img", "label"]:
        raise SystemExit(f"unexpected parquet schema: {table.schema.names}")
    if table.num_rows != ROWS[split]:
        raise SystemExit(f"{split}: {table.num_rows} rows, expected {ROWS[split]}")
    cells = table.column("img").to_pylist()
    labels = np.asarray(table.column("label").to_pylist(), dtype=np.int64)
    images = np.zeros((len(cells), 32, 32, 3), dtype=np.uint8)
    for i, cell in enumerate(cells):
        blob = cell["bytes"] if isinstance(cell, dict) else cell
        im = Image.open(io.BytesIO(blob))
        if im.mode != "RGB" or im.size != (32, 32):
            raise SystemExit(f"row {i}: mode={im.mode} size={im.size}")
        images[i] = np.asarray(im, dtype=np.uint8)

    counts = np.bincount(labels, minlength=10)
    if counts.tolist() != [PER_CLASS[split]] * 10:
        raise SystemExit(f"{split}: unbalanced classes {counts.tolist()}")
    for name, arr in (("images", images), ("labels", labels)):
        digest = hashlib.sha256(arr.tobytes()).hexdigest()
        if digest != CONTENT_SHA256[split][name]:
            raise SystemExit(
                f"{split}/{name}: content sha256 {digest} != pinned "
                f"{CONTENT_SHA256[split][name]}"
            )
    return images, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--split", required=True, choices=sorted(ROWS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    images, labels = decode(args.parquet, args.split)
    np.savez(args.out, images=images, labels=labels)
    print(
        f"wrote {args.out}: images {images.shape} {images.dtype}, "
        f"labels {labels.shape} {labels.dtype}, content hashes verified",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
