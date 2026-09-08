import hashlib
import math

import numpy as np
import torch
import torch.nn.functional as F

TEST_NPZ = "/tests/hidden_data/cifar10_test.npz"
TEST_ROWS = 10000
TEST_IMAGES_SHA256 = "f9b1e2130bab680db56eb06d886f7f1ad0f1183907ceff4659d1735c890f5b04"
TEST_LABELS_SHA256 = "cbb7365de8ed11f05cc4c3a1e7f78144127c5e851efd83762fb18202461230bb"

SPLITS = ("intermediate", "final")
SPLIT_SEED = 20260818
SPLIT_ROWS = TEST_ROWS // 2

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)
EPSILON_PIXEL = 8.0 / 255.0
ALPHA_PIXEL = 2.0 / 255.0
ATTACK_ITERS = 50
RESTARTS = 10
EVAL_BATCH = 256
SEED = 0

LOGIT_SCALE_REF = 20.0

M0 = 10.0
BASELINE_SCORE = 46.06


def reward_from_metric(metric_percent):
    """Map robust accuracy in percent to a reward in [0, 1]; 0 at the degenerate baseline."""
    x_ref = BASELINE_SCORE - M0
    if not x_ref > 0:
        raise ValueError("malformed task: the anchor does not beat the trivial baseline")
    u = max(0.0, float(metric_percent) - M0) / x_ref
    return u / (1.0 + u)


def split_indices(split, labels):
    """Row indices of one sealed half, as a sorted int array.

    Both halves are always rebuilt and checked here, on every grading pass, rather than in a
    test that could be deselected: an edit that made them overlap, drop a row or unbalance a
    class raises instead of silently scoring the wrong images.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}, expected one of {SPLITS}")
    rng = np.random.default_rng(SPLIT_SEED)
    halves = {name: [] for name in SPLITS}
    for cls in range(10):
        rows = np.flatnonzero(labels == cls)
        shuffled = rows[rng.permutation(rows.shape[0])]
        cut = shuffled.shape[0] // 2
        halves["intermediate"].append(shuffled[:cut])
        halves["final"].append(shuffled[cut:])
    halves = {name: np.sort(np.concatenate(rows)) for name, rows in halves.items()}

    first, second = (set(halves[name].tolist()) for name in SPLITS)
    if first & second:
        raise ValueError("sealed halves overlap")
    if first | second != set(range(TEST_ROWS)):
        raise ValueError("sealed halves do not cover the test split")
    for name, idx in halves.items():
        if idx.shape[0] != SPLIT_ROWS:
            raise ValueError(f"sealed half {name!r} has {idx.shape[0]} rows, expected {SPLIT_ROWS}")
        counts = np.bincount(labels[idx], minlength=10).tolist()
        if counts != [SPLIT_ROWS // 10] * 10:
            raise ValueError(f"sealed half {name!r} is class-unbalanced: {counts}")
    return halves[split]


def load_test_tensors(device, split):
    """Load the sealed test split, assert it is the data whose content was pinned, and return
    the half named by `split`.  The hashes are checked on the whole 10,000 rows before any
    subsetting, so the integrity check is the same one on either half."""
    with np.load(TEST_NPZ) as blob:
        images = np.ascontiguousarray(blob["images"])
        labels = np.ascontiguousarray(blob["labels"])
    if images.shape != (TEST_ROWS, 32, 32, 3) or images.dtype != np.uint8:
        raise ValueError(f"sealed images malformed: {images.shape} {images.dtype}")
    if labels.shape != (TEST_ROWS,) or labels.dtype != np.int64:
        raise ValueError(f"sealed labels malformed: {labels.shape} {labels.dtype}")
    if hashlib.sha256(images.tobytes()).hexdigest() != TEST_IMAGES_SHA256:
        raise ValueError("sealed images do not match the pinned content hash")
    if hashlib.sha256(labels.tobytes()).hexdigest() != TEST_LABELS_SHA256:
        raise ValueError("sealed labels do not match the pinned content hash")

    idx = split_indices(split, labels)
    images = np.ascontiguousarray(images[idx])
    labels = np.ascontiguousarray(labels[idx])

    mean = torch.tensor(CIFAR10_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, dtype=torch.float32).view(1, 3, 1, 1)
    x = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous().float().div_(255.0)
    x = (x - mean) / std
    return x.to(device), torch.from_numpy(labels).to(device)


def _clamp(t, low, high):
    return torch.max(torch.min(t, high), low)


def _logits(model, x, temperature):
    out = model(x)
    if temperature != 1.0:
        out = out / temperature
    return out


def _logit_scale(model, x_all, device):
    """s = mean over the clean test set of the per-example std of the 10 logits."""
    total = 0.0
    with torch.no_grad():
        for start in range(0, x_all.shape[0], EVAL_BATCH):
            out = model(x_all[start:start + EVAL_BATCH])
            total += out.float().std(dim=1).sum().item()
    return total / x_all.shape[0]


def _clean_correct(model, x_all, y_all, temperature):
    correct = 0
    with torch.no_grad():
        for start in range(0, x_all.shape[0], EVAL_BATCH):
            out = _logits(model, x_all[start:start + EVAL_BATCH], temperature)
            correct += (out.argmax(1) == y_all[start:start + EVAL_BATCH]).sum().item()
    return correct


def _attack_batch(model, x, y, temperature, eps, alpha, low, high):
    """The paper's attack_pgd: `RESTARTS` random restarts of `ATTACK_ITERS` sign steps,
    keeping per example the delta of the restart with the largest cross-entropy."""
    max_loss = torch.zeros(y.shape[0], device=x.device)
    max_delta = torch.zeros_like(x)
    box_low, box_high = low - x, high - x
    for _ in range(RESTARTS):
        delta = torch.zeros_like(x)
        for c in range(3):
            delta[:, c, :, :].uniform_(-eps[0, c, 0, 0].item(), eps[0, c, 0, 0].item())
        delta = _clamp(delta, box_low, box_high)
        delta.requires_grad_(True)
        for _ in range(ATTACK_ITERS):
            out = _logits(model, x + delta, temperature)
            idx = torch.where(out.argmax(1) == y)[0]
            if idx.numel() == 0:
                break
            loss = F.cross_entropy(out, y)
            grad = torch.autograd.grad(loss, delta, allow_unused=True)[0]
            if grad is None:
                break
            grad = grad.detach()
            with torch.no_grad():
                step = _clamp(delta[idx] + alpha * torch.sign(grad[idx]), -eps, eps)
                delta[idx] = _clamp(step, box_low[idx], box_high[idx])
        with torch.no_grad():
            out = _logits(model, x + delta, temperature)
            all_loss = F.cross_entropy(out, y, reduction="none").detach()
            improved = all_loss >= max_loss
            max_delta[improved] = delta.detach()[improved]
            max_loss = torch.max(max_loss, all_loss)
    return max_delta


def evaluate(model, device, split):
    """Return the graded metric and its diagnostics for an already-loaded, eval-mode model."""
    for setter in (
        lambda: torch.set_num_threads(4),
        lambda: setattr(torch.backends.cudnn, "deterministic", True),
        lambda: setattr(torch.backends.cudnn, "benchmark", False),
        lambda: setattr(torch.backends.cuda.matmul, "allow_tf32", False),
        lambda: torch.use_deterministic_algorithms(True),
    ):
        try:
            setter()
        except Exception:
            pass
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    x_all, y_all = load_test_tensors(device, split)
    for p in model.parameters():
        p.requires_grad_(False)

    scale = _logit_scale(model, x_all, device)
    temperature = max(1.0, scale / LOGIT_SCALE_REF)
    if not math.isfinite(temperature):
        raise ValueError("model produced non-finite logits on the clean test set")
    clean_correct = _clean_correct(model, x_all, y_all, temperature)

    std = torch.tensor(CIFAR10_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    mean = torch.tensor(CIFAR10_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    eps = EPSILON_PIXEL / std
    alpha = ALPHA_PIXEL / std
    low = (0.0 - mean) / std
    high = (1.0 - mean) / std

    robust_correct = 0
    for start in range(0, x_all.shape[0], EVAL_BATCH):
        x = x_all[start:start + EVAL_BATCH]
        y = y_all[start:start + EVAL_BATCH]
        delta = _attack_batch(model, x, y, temperature, eps, alpha, low, high)
        with torch.no_grad():
            out = _logits(model, x + delta, temperature)
            robust_correct += (out.argmax(1) == y).sum().item()

    n = x_all.shape[0]
    return {
        "robust_accuracy_percent": round(100.0 * robust_correct / n, 2),
        "clean_accuracy_percent": round(100.0 * clean_correct / n, 2),
        "logit_scale_s": scale,
        "temperature": temperature,
        "n_eval_images": n,
        "split": split,
    }
