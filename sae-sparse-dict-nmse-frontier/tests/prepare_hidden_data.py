#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_acts import normalize_doc

OWT_REPO = "Skylion007/openwebtext"
OWT_REVISION = "79d93d786212f7344586290adb811d4ae6a1762c"
SEALED_SHARD = "plain_text/train-00079-of-00080.parquet"
AGENT_SHARDS = [
    "plain_text/train-00000-of-00080.parquet",
    "plain_text/train-00001-of-00080.parquet",
]
N_SEALED = 1_048_576
SPLITS = {
    "intermediate": (0, 20000),
    "final": (20000, 40000),
}
HIDDEN = Path(__file__).resolve().parent / "hidden_data"
SCRATCH = Path("/tmp/hidden_prepare")


def main() -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    HIDDEN.mkdir(parents=True, exist_ok=True)

    sealed_src = hf_hub_download(repo_id=OWT_REPO, repo_type="dataset",
                                 revision=OWT_REVISION, filename=SEALED_SHARD,
                                 local_dir=str(SCRATCH / "owt"))
    agent_paths = [hf_hub_download(repo_id=OWT_REPO, repo_type="dataset",
                                   revision=OWT_REVISION, filename=f,
                                   local_dir=str(SCRATCH / "owt")) for f in AGENT_SHARDS]

    agent_hashes: set[str] = set()
    for path in agent_paths:
        texts = pq.read_table(path, columns=["text"]).column("text").to_pylist()
        agent_hashes.update(normalize_doc(t) for t in texts)
        del texts
    print(f"agent-visible document fingerprints: {len(agent_hashes)}")

    sealed_texts = pq.read_table(sealed_src, columns=["text"]).column("text").to_pylist()
    lo = min(a for a, _ in SPLITS.values())
    hi = max(b for _, b in SPLITS.values())
    excluded: list[int] = []
    seen: set[str] = set()
    cross_shard_dups = 0
    within_sealed_dups = 0
    for i in range(lo, hi):
        h = normalize_doc(sealed_texts[i])
        if h in agent_hashes:
            excluded.append(i)
            cross_shard_dups += 1
        elif h in seen:
            excluded.append(i)
            within_sealed_dups += 1
        else:
            seen.add(h)
    del sealed_texts, agent_hashes

    sealed_dst = HIDDEN / "sealed_shard.parquet"
    shutil.move(sealed_src, sealed_dst)
    (HIDDEN / "excluded_doc_indices.json").write_text(json.dumps(excluded))

    for name, (doc_start, doc_end) in SPLITS.items():
        out_dir = HIDDEN / name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "spec.json").write_text(json.dumps({
            "shard": SEALED_SHARD,
            "revision": OWT_REVISION,
            "doc_start": doc_start,
            "doc_end": doc_end,
            "max_tokens": N_SEALED,
            "ctx": 64,
            "excluded_doc_indices": [i for i in excluded if doc_start <= i < doc_end],
        }, indent=2, sort_keys=True))

    (HIDDEN / "dedup_report.json").write_text(json.dumps({
        "sealed_shard": SEALED_SHARD,
        "documents_scanned": hi - lo,
        "cross_shard_duplicates_removed": cross_shard_dups,
        "within_sealed_duplicates_removed": within_sealed_dups,
    }, indent=2, sort_keys=True))

    shutil.rmtree(SCRATCH, ignore_errors=True)
    print(json.dumps({"cross_shard_duplicates_removed": cross_shard_dups,
                      "within_sealed_duplicates_removed": within_sealed_dups,
                      "sealed_shard_bytes": sealed_dst.stat().st_size}))


if __name__ == "__main__":
    main()
