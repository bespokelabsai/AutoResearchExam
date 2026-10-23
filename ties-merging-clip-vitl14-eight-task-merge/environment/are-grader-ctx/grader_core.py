from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

TASKS = ("sun397", "cars", "resisc45", "eurosat", "svhn", "gtsrb", "mnist", "dtd")

ASSETS = Path("/opt/assets")
EVAL_ROOT = Path("/opt/eval")
HIDDEN_ROOT = Path("/tests/hidden_data")

IMAGE_SIDE = 224
EVAL_BATCH = 256
SEED = 0

MERGE_BUDGET_SEC = 1200.0
OVERHEAD_ALLOWANCE_SEC = 600.0

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

DEGENERATE_REWARD = 0.10
M_FLOOR = 65.23
M0 = 80.05
BASELINE_SCORE = 86.00
HIGHER_IS_BETTER = True


def graded_reward(metric: float | None, valid: bool) -> float:
    """metric -> reward in [0, 1]. The invalid path short-circuits before any arithmetic."""
    if not valid or metric is None or not math.isfinite(float(metric)):
        return 0.0
    m = float(metric)
    sigma = 1.0 if HIGHER_IS_BETTER else -1.0
    x = sigma * (m - M0)
    x_ref = sigma * (BASELINE_SCORE - M0)
    assert x_ref > 0, "malformed task: the baseline score does not beat the trivial baseline"
    c = DEGENERATE_REWARD
    g = sigma * (M0 - M_FLOOR)
    if c <= 0.0 or g <= 0:
        u = max(0.0, x) / x_ref
        return u / (1.0 + u)
    if x < 0:
        v = max(0.0, sigma * (m - M_FLOOR)) / g
        return c * v
    u = x / x_ref
    return c + (1.0 - c) * (u / (1.0 + u))




def determinism_env() -> dict[str, str]:
    """The pins handed to every process that runs torch, root grader and candidate alike."""
    return {
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "TQDM_DISABLE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }


def pin_determinism() -> None:
    import random

    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)




def vision_config():
    from transformers import CLIPVisionConfig

    return CLIPVisionConfig.from_pretrained(str(ASSETS / "pretrained"))


def canonical_keys() -> set[str]:
    """The 391 keys CLIPVisionModel itself owns. The pretrained checkpoint carries one extra
    buffer (`vision_model.embeddings.position_ids`) that this transformers version dropped, so
    the key set comes from the module and never from the file."""
    import torch
    from transformers import CLIPVisionModel

    with torch.device("meta"):
        return set(CLIPVisionModel(vision_config()).state_dict())


def reference_state_dict() -> dict:
    """The pretrained CLIPVisionModel state dict -- the key set and shapes a merge must return."""
    from safetensors.torch import load_file

    raw = load_file(str(ASSETS / "pretrained" / "model.safetensors"))
    keys = canonical_keys()
    out = {k: v.float() for k, v in raw.items() if k in keys}
    assert set(out) == keys, f"pretrained checkpoint is missing {sorted(keys - set(out))[:3]}"
    return out


def eval_assets(device: str):
    """The frozen heads and visual projection. Public, and identical in both images."""
    import torch

    blob = torch.load(str(ASSETS / "heads.pt"), map_location="cpu", weights_only=True)
    heads = {t: blob["heads"][t].to(device=device, dtype=torch.float32) for t in TASKS}
    proj = blob["visual_projection"].to(device=device, dtype=torch.float32)
    assert proj.shape == (768, 1024), proj.shape
    for t, h in heads.items():
        assert h.ndim == 2 and h.shape[1] == 768, (t, tuple(h.shape))
        norms = h.norm(dim=-1)
        assert bool(((norms - 1.0).abs() < 1e-3).all()), f"{t}: head rows are not unit norm"
    return heads, proj



VALIDATION_FAILURES = (
    "missing_key", "unexpected_key", "shape_mismatch", "non_finite", "not_castable",
)


def validate_state_dict(candidate: dict, reference: dict) -> str | None:
    """None when the returned state dict is gradeable, else a short machine-readable reason."""
    import torch

    if not isinstance(candidate, dict):
        return "not_a_dict"
    cand_keys, ref_keys = set(candidate), set(reference)
    if ref_keys - cand_keys:
        return f"missing_key:{sorted(ref_keys - cand_keys)[0]}"
    if cand_keys - ref_keys:
        return f"unexpected_key:{sorted(cand_keys - ref_keys)[0]}"
    for k in sorted(ref_keys):
        v = candidate[k]
        if not torch.is_tensor(v):
            return f"not_castable:{k}"
        if tuple(v.shape) != tuple(reference[k].shape):
            return f"shape_mismatch:{k}"
        try:
            f = v.detach().to(dtype=torch.float32)
        except Exception:
            return f"not_castable:{k}"
        if not bool(torch.isfinite(f).all()):
            return f"non_finite:{k}"
    return None




def panel_dirs(panel: str) -> tuple[Path, Path]:
    return EVAL_ROOT / panel, HIDDEN_ROOT / panel


def audit_panel(panel: str) -> dict:
    """Fail-closed split audit: every task present, and no sealed row duplicates a dev row."""
    _, labels_root = panel_dirs(panel)
    out = {}
    for task in TASKS:
        info = json.loads((labels_root / task / "split_audit.json").read_text())
        assert info["task"] == task
        assert info["panel_rows"] > 0, f"{panel}/{task}: empty panel"
        assert info["exact_duplicates_remaining"] == 0, (
            f"{panel}/{task}: a sealed row is a pixel-exact duplicate of a dev row")
        assert info["panel_rows"] + info["dropped_exact_duplicates_of_dev"] == \
            info["panel_rows_before_dedup"], f"{panel}/{task}: audit row counts disagree"
        out[task] = info
    return out


def evaluate(state_dict: dict, panel: str, device: str = "cuda",
             batch: int = EVAL_BATCH) -> dict:
    """Per-task top-1 accuracy (percent) and their unweighted mean, on one sealed panel."""
    import torch

    images_root, labels_root = panel_dirs(panel)
    heads, proj = eval_assets(device)

    from transformers import CLIPVisionModel

    model = CLIPVisionModel(vision_config())
    model.load_state_dict({k: v.detach().to(dtype=torch.float32) for k, v in state_dict.items()},
                          strict=True)
    model = model.to(device=device, dtype=torch.float32).eval()

    mean = torch.tensor(CLIP_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    per_task, counts = {}, {}
    with torch.inference_mode():
        for task in TASKS:
            imgs = np.load(images_root / task / "images.npy", mmap_mode="r")
            labels = np.load(labels_root / task / "labels.npy")
            assert imgs.shape[0] == labels.shape[0], task
            assert imgs.shape[1:] == (3, IMAGE_SIDE, IMAGE_SIDE), (task, imgs.shape)
            head = heads[task]
            assert int(labels.max()) < head.shape[0], (task, int(labels.max()), head.shape)
            correct = 0
            for start in range(0, imgs.shape[0], batch):
                chunk = np.ascontiguousarray(imgs[start:start + batch])
                x = torch.from_numpy(chunk).to(device=device, non_blocking=False)
                x = x.to(dtype=torch.float32).div_(255.0).sub_(mean).div_(std)
                pooled = model(pixel_values=x).pooler_output
                emb = pooled @ proj.t()
                emb = emb / emb.norm(dim=-1, keepdim=True)
                pred = (emb @ head.t()).argmax(dim=-1).cpu().numpy()
                correct += int((pred == labels[start:start + batch]).sum())
            per_task[task] = 100.0 * correct / int(imgs.shape[0])
            counts[task] = int(imgs.shape[0])
    metric = sum(per_task[t] for t in TASKS) / len(TASKS)
    return {"per_task_accuracy": per_task, "panel_rows": counts, "metric": metric}
