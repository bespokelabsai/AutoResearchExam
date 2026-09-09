#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

EXIT_CANDIDATE_FAILURE = 3


def pin_determinism() -> None:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


    torch.use_deterministic_algorithms(True, warn_only=True)


def fail(out_dir: Path, reason: str, detail: str) -> None:
    (out_dir / "failure.json").write_text(json.dumps(
        {"reason": reason, "detail": detail[:4000]}))
    sys.stderr.write(f"{reason}: {detail[:4000]}\n")
    sys.exit(EXIT_CANDIDATE_FAILURE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission-dir", required=True)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-rows", type=int, required=True)
    ap.add_argument("--n-rows", type=int, required=True)
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--max-k", type=int, required=True)
    ap.add_argument("--max-total-nonzero", type=int, required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    pin_determinism()

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    sys.path.insert(0, args.submission_dir)
    spent = 0.0
    try:
        t0 = time.monotonic()
        import encoder
        if not hasattr(encoder, "load"):
            spent += time.monotonic() - t0
            fail(out_dir, "encoder_missing_load", "module defines no load(device)")
        model = encoder.load(device_str)
        spent += time.monotonic() - t0
    except SystemExit:
        raise
    except BaseException:
        fail(out_dir, "encoder_load_failed", traceback.format_exc())

    if not hasattr(model, "encode"):
        fail(out_dir, "encoder_missing_encode", "load() returned an object with no encode()")

    acts = np.load(args.acts, mmap_mode="r")
    if acts.shape != (args.n_rows, 768):
        fail(out_dir, "runner_bad_inputs", str(acts.shape))

    n_batches = args.n_rows // args.batch_rows
    total_nonzero = 0
    for b in range(n_batches):
        lo = b * args.batch_rows
        x = torch.from_numpy(np.array(
            acts[lo:lo + args.batch_rows])).to(device).float()
        t0 = time.monotonic()
        try:
            ret = model.encode(x)
            if device_str == "cuda":
                torch.cuda.synchronize()
        except BaseException:
            spent += time.monotonic() - t0
            fail(out_dir, "encode_raised", f"batch {b}\n{traceback.format_exc()}")
        spent += time.monotonic() - t0
        if spent > args.budget:
            fail(out_dir, "encode_budget_exceeded",
                 f"{spent:.2f}s of the {args.budget:.0f}s budget after batch {b}")

        if not (isinstance(ret, tuple) and len(ret) == 2):
            fail(out_dir, "encode_bad_return", f"batch {b}: expected a 2-tuple, got {type(ret)}")
        idx, val = ret
        if not isinstance(idx, torch.Tensor) or not isinstance(val, torch.Tensor):
            fail(out_dir, "encode_bad_return", f"batch {b}: entries must be torch tensors")
        if idx.dim() != 2 or val.dim() != 2 or idx.shape != val.shape:
            fail(out_dir, "encode_bad_shape",
                 f"batch {b}: idx {tuple(idx.shape)} val {tuple(val.shape)}")
        if idx.shape[0] != args.batch_rows:
            fail(out_dir, "encode_bad_shape",
                 f"batch {b}: first dimension {idx.shape[0]} != {args.batch_rows}")
        k = int(idx.shape[1])
        if k > args.max_k:
            fail(out_dir, "encode_k_too_large", f"batch {b}: K={k}")
        if idx.dtype not in (torch.int16, torch.int32, torch.int64):
            fail(out_dir, "encode_bad_dtype", f"batch {b}: idx dtype {idx.dtype}")
        if val.dtype not in (torch.float16, torch.float32, torch.float64):
            fail(out_dir, "encode_bad_dtype", f"batch {b}: val dtype {val.dtype}")

        idx_np = idx.detach().to("cpu").to(torch.int64).numpy()
        val_np = val.detach().to("cpu").to(torch.float32).numpy()
        keep = val_np != 0.0
        rows = np.nonzero(keep)[0].astype(np.int32)
        cols = idx_np[keep].astype(np.int64)
        vals = val_np[keep].astype(np.float32)
        if cols.size and (cols.min() < -(1 << 31) or cols.max() > (1 << 31) - 1):
            fail(out_dir, "encode_index_out_of_range", f"batch {b}")
        total_nonzero += int(rows.size)
        if total_nonzero > args.max_total_nonzero:
            fail(out_dir, "nonzero_budget_exceeded", f"batch {b}: {total_nonzero}")
        np.save(out_dir / f"b{b:05d}_row.npy", rows)
        np.save(out_dir / f"b{b:05d}_col.npy", cols.astype(np.int32))
        np.save(out_dir / f"b{b:05d}_val.npy", vals)
        del x, idx, val, ret

    (out_dir / "run_meta.json").write_text(json.dumps({
        "encode_seconds": spent,
        "total_nonzero": total_nonzero,
        "n_batches": n_batches,
        "device": device_str,
        "cuda_device_count": torch.cuda.device_count(),
        "uid": os.getuid(),
    }))
    print(f"encode_seconds={spent:.3f} total_nonzero={total_nonzero} uid={os.getuid()} "
          f"device={device_str} cuda_devices={torch.cuda.device_count()}")


if __name__ == "__main__":
    main()
