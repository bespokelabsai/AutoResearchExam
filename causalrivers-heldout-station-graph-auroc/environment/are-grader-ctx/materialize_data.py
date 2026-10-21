import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def main(shard_dir, target_dir):
    shard_dir, target_dir = Path(shard_dir), Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((shard_dir / "manifest.json").read_text())
    for name, entry in sorted(manifest.items()):
        blocks = []
        for shard in entry["shards"]:
            with np.load(shard_dir / shard["file"]) as handle:
                blocks.append(handle["block"])
        matrix = np.ascontiguousarray(np.concatenate(blocks, axis=int(entry["axis"])))
        del blocks
        if list(matrix.shape) != entry["shape"] or matrix.dtype != np.dtype(entry["dtype"]):
            raise SystemExit(f"{name}: reassembled {matrix.shape}/{matrix.dtype}, "
                             f"expected {entry['shape']}/{entry['dtype']}")
        digest = hashlib.sha256(matrix.tobytes()).hexdigest()
        if digest != entry["sha256_bytes"]:
            raise SystemExit(f"{name}: sha256 {digest} != manifest {entry['sha256_bytes']}")
        np.save(target_dir / f"{name}.npy", matrix)
        print(f"{name}: {matrix.shape} verified and written to {target_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
