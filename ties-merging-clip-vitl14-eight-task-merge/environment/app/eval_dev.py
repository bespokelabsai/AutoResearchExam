from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch

TASKS = ("sun397", "cars", "resisc45", "eurosat", "svhn", "gtsrb", "mnist", "dtd")
ASSETS = Path("/opt/assets")
DEV = Path("/app/data/dev")
DEV_UNLABELED = Path("/app/data/dev_unlabeled")
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def pin_determinism() -> None:
    import random

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def load_inputs():
    from safetensors.torch import load_file
    from transformers import CLIPVisionConfig, CLIPVisionModel

    with torch.device("meta"):
        canon = set(CLIPVisionModel(
            CLIPVisionConfig.from_pretrained(str(ASSETS / "pretrained"))).state_dict())
    pretrained = {k: v.float() for k, v in
                  load_file(str(ASSETS / "pretrained" / "model.safetensors")).items()
                  if k in canon}
    assert set(pretrained) == canon
    experts = {t: {k: v.float() for k, v in
                   load_file(str(ASSETS / t / "model.safetensors")).items()} for t in TASKS}
    return pretrained, experts


def score(state_dict, tasks, limit, batch, device):
    from transformers import CLIPVisionConfig, CLIPVisionModel

    blob = torch.load(str(ASSETS / "heads.pt"), map_location="cpu", weights_only=True)
    proj = blob["visual_projection"].to(device=device, dtype=torch.float32)
    model = CLIPVisionModel(CLIPVisionConfig.from_pretrained(str(ASSETS / "pretrained")))
    model.load_state_dict({k: v.detach().to(dtype=torch.float32)
                           for k, v in state_dict.items()}, strict=True)
    model = model.to(device=device, dtype=torch.float32).eval()
    mean = torch.tensor(CLIP_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    acc = {}
    with torch.inference_mode():
        for task in tasks:
            imgs = np.load(DEV / task / "images.npy", mmap_mode="r")
            labels = np.load(DEV / task / "labels.npy")
            if limit is None or limit >= len(labels):
                sel = np.arange(len(labels))
            else:
                sel = np.arange(0, len(labels), max(1, len(labels) // limit))[:limit]
            n = len(sel)
            head = blob["heads"][task].to(device=device, dtype=torch.float32)
            correct = 0
            for s in range(0, n, batch):
                rows = sel[s:s + batch]
                chunk = np.ascontiguousarray(imgs[rows])
                x = torch.from_numpy(chunk).to(device).to(torch.float32).div_(255.0)
                x = x.sub_(mean).div_(std)
                emb = model(pixel_values=x).pooler_output @ proj.t()
                emb = emb / emb.norm(dim=-1, keepdim=True)
                pred = (emb @ head.t()).argmax(dim=-1).cpu().numpy()
                correct += int((pred == labels[rows]).sum())
            acc[task] = 100.0 * correct / n
            print(f"  {task:9s} n={n:6d}  top1={acc[task]:6.2f}", flush=True)
    return acc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", default="/app/output")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--unlabeled", default=str(DEV_UNLABELED))
    args = ap.parse_args()
    pin_determinism()

    entry = Path(args.solution) / "merge.py"
    if not entry.is_file():
        print(f"no entry point at {entry}", file=sys.stderr)
        return 2
    sys.path.insert(0, str(Path(args.solution).resolve()))
    spec = importlib.util.spec_from_file_location("candidate_merge", str(entry))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    pretrained, experts = load_inputs()
    t0 = time.monotonic()
    merged = module.merge(pretrained, experts, args.unlabeled, args.device)
    dt = time.monotonic() - t0
    print(f"merge() returned in {dt:.1f}s", flush=True)

    missing = set(pretrained) - set(merged)
    extra = set(merged) - set(pretrained)
    if missing or extra:
        print(f"key set mismatch: {len(missing)} missing, {len(extra)} unexpected",
              file=sys.stderr)
        return 3
    for k, v in merged.items():
        t = v if torch.is_tensor(v) else torch.as_tensor(v)
        if tuple(t.shape) != tuple(pretrained[k].shape):
            print(f"shape mismatch at {k}", file=sys.stderr)
            return 3
        if not bool(torch.isfinite(t.float()).all()):
            print(f"non-finite values at {k}", file=sys.stderr)
            return 3

    acc = score(merged, args.tasks, args.limit, args.batch, args.device)
    mean = sum(acc.values()) / len(acc)
    print(json.dumps({"per_task_accuracy": acc, "unweighted_mean_top1": mean,
                      "merge_seconds": dt}, indent=1))
    if set(args.tasks) != set(TASKS):
        print("NOTE: the reported mean covers only the tasks you selected.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
