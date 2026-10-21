from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np

TASKS = ("sun397", "cars", "resisc45", "eurosat", "svhn", "gtsrb", "mnist", "dtd")


class MergeTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise MergeTimeout("merge() exceeded its wall-clock budget")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--unlabeled", required=True)
    ap.add_argument("--assets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"status": "error", "reason": "unset", "merge_seconds": None}

    def flush(code: int) -> int:
        (out / "report.json").write_text(json.dumps(report, indent=1))
        return code

    import torch

    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    try:
        report["cuda_device_count"] = int(torch.cuda.device_count())
        probe = (torch.ones(64, 64, device=args.device) @ torch.eye(64, device=args.device)).sum()
        report["cuda_probe_sum"] = float(probe.item())
    except Exception as exc:
        report["reason"] = f"cuda_unavailable_to_uid_{os.getuid()}: {exc!r}"
        print(f"[runner] CUDA unavailable to uid {os.getuid()}: {exc!r}", file=sys.stderr)
        return flush(3)

    from safetensors.torch import load_file, save_file

    entry = Path(args.solution) / "merge.py"
    if not entry.is_file() or entry.stat().st_size == 0:
        report["reason"] = "missing_or_empty_entry_point"
        return flush(5)

    sys.path.insert(0, str(Path(args.solution).resolve()))
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("candidate_merge", str(entry))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except BaseException as exc:
        report["reason"] = f"import_failed: {exc!r}"
        traceback.print_exc()
        return flush(6)

    fn = getattr(module, "merge", None)
    if not callable(fn):
        report["reason"] = "no_callable_merge"
        return flush(7)

    assets = Path(args.assets)
    try:
        from transformers import CLIPVisionConfig, CLIPVisionModel

        cfg = CLIPVisionConfig.from_pretrained(str(assets / "pretrained"))
        with torch.device("meta"):
            canon = set(CLIPVisionModel(cfg).state_dict())
        pretrained = {k: v.float() for k, v in
                      load_file(str(assets / "pretrained" / "model.safetensors")).items()
                      if k in canon}
        if set(pretrained) != canon:
            report["reason"] = "asset_keyset_mismatch:pretrained"
            return flush(4)
        experts = {}
        for task in TASKS:
            experts[task] = {k: v.float() for k, v in
                             load_file(str(assets / task / "model.safetensors")).items()}
            missing = set(pretrained) ^ set(experts[task])
            if missing:
                report["reason"] = f"asset_keyset_mismatch:{task}"
                return flush(4)
    except Exception as exc:
        report["reason"] = f"asset_load_failed: {exc!r}"
        traceback.print_exc()
        return flush(4)

    signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, args.budget)
    t0 = time.monotonic()
    try:
        merged = fn(pretrained, experts, str(args.unlabeled), args.device)
    except MergeTimeout:
        report["reason"] = "merge_timeout"
        report["merge_seconds"] = time.monotonic() - t0
        return flush(8)
    except BaseException as exc:
        report["reason"] = f"merge_raised: {exc!r}"
        report["merge_seconds"] = time.monotonic() - t0
        traceback.print_exc()
        return flush(9)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
    report["merge_seconds"] = time.monotonic() - t0

    if not isinstance(merged, dict):
        report["reason"] = "return_not_a_dict"
        return flush(10)
    absent = sorted(set(pretrained) - set(merged))
    if absent:
        report["reason"] = f"missing_key:{absent[0]}"
        return flush(10)
    extra = sorted(set(merged) - set(pretrained))
    if extra:
        report["reason"] = f"unexpected_key:{extra[0]}"
        return flush(10)
    try:
        tensors = {}
        for k in sorted(pretrained):
            v = merged[k]
            t = v if torch.is_tensor(v) else torch.as_tensor(v)
            tensors[k] = t.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
        save_file(tensors, str(out / "merged.safetensors"))
    except Exception as exc:
        report["reason"] = f"return_unserializable: {exc!r}"
        traceback.print_exc()
        return flush(11)

    report["status"] = "ok"
    report["reason"] = "ok"
    report["returned_keys"] = len(tensors)
    return flush(0)


if __name__ == "__main__":
    sys.exit(main())
