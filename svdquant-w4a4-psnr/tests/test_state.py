import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grader_core

METRIC_PATH = Path("/logs/verifier/metric.json")


def invalid(reason):
    return {
        "valid": False,
        "metric_name": "mean_psnr_db",
        "metric": None,
        "reward": 0.0,
        "reason": reason,
    }


def test_submission():
    split = os.environ.get("GRADER_SPLIT", "final")
    if split not in {"intermediate", "final"}:
        result = invalid("invalid grader split")
    else:
        entry = Path("/app/output/quantize.py")
        prompts = Path("/tests/hidden_data") / split / "prompts.jsonl"
        try:
            print(f"trusted CUDA device_count={__import__('torch').cuda.device_count()}")
            if set(entry.parent.iterdir()) != {entry}:
                raise ValueError("only /app/output/quantize.py may be created or modified")
            if not entry.is_file() or entry.is_symlink() or entry.stat().st_size == 0:
                raise FileNotFoundError("missing or empty /app/output/quantize.py")
            result = grader_core.score(entry, prompts)
        except Exception as exc:
            print(f"grading failure: {type(exc).__name__}: {exc}")
            result = invalid(f"{type(exc).__name__}: {exc}")
    METRIC_PATH.write_text(json.dumps(result, sort_keys=True) + "\n")
    assert result.get("valid") is True, result.get("reason", "invalid submission")
