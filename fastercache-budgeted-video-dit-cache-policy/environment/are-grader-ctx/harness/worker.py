from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from harness import hcore

WORKER_SEED = 987654321


def _seed_everything() -> None:
    random.seed(WORKER_SEED)
    np.random.seed(WORKER_SEED)
    torch.manual_seed(WORKER_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(WORKER_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _write(out, header: dict, arrays: dict) -> None:
    out.write(hcore.encode_frame(header, arrays))
    out.flush()


def _read(stream) -> tuple[dict, dict]:
    return hcore.read_frame(stream, deadline=time.monotonic() + 86400.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", required=True)
    args = parser.parse_args()

    proto_out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    proto_in = sys.stdin.buffer

    _seed_everything()
    torch.set_num_threads(1)

    policy_cls = None
    policy = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_shape = tuple(hcore.LATENT_SHAPE)
    num_steps = hcore.NUM_INFERENCE_STEPS
    budget = hcore.BUDGET_UNITS
    last_latents = None

    while True:
        try:
            header, arrays = _read(proto_in)
        except (EOFError, struct.error):
            return 0
        op = header.get("op")
        try:
            if op == "init":
                num_steps = int(header["num_steps"])
                budget = int(header["budget"])
                latent_shape = tuple(int(v) for v in header["latent_shape"])
                if torch.cuda.is_available():
                    torch.fft.fftn(torch.zeros(4, 4, 4, device=device,
                                               dtype=torch.float32)).abs().sum().item()
                sys.path.insert(0, str(Path(args.solution).resolve()))
                _seed_everything()
                module = importlib.import_module("policy")
                policy_cls = getattr(module, "CachePolicy")
                _write(proto_out, {"ok": True}, {})
            elif op == "begin":
                t0 = time.monotonic()
                policy = policy_cls(num_steps=num_steps, budget=budget, device=device,
                                    dtype=hcore.DTYPE)
                _write(proto_out, {"ok": True, "seconds": time.monotonic() - t0}, {})
            elif op == "decide":
                state = dict(header["state"])
                last_latents = arrays["latents"].to(device=device, dtype=hcore.DTYPE)
                state["latents"] = last_latents
                t0 = time.monotonic()
                mode = policy.decide(int(state["step"]), state)
                _write(proto_out,
                       {"ok": True, "mode": mode, "seconds": time.monotonic() - t0}, {})
            elif op == "reconstruct":
                state = dict(header["state"])
                state["latents"] = last_latents
                mode = str(header["mode"])
                computed = {k: v.to(device=device, dtype=hcore.DTYPE)
                            for k, v in arrays.items()}
                t0 = time.monotonic()
                supplied = policy.reconstruct(int(state["step"]), mode, computed, state)
                elapsed = time.monotonic() - t0
                if not isinstance(supplied, (tuple, list)) or len(supplied) != 2:
                    raise TypeError("reconstruct must return a 2-tuple "
                                    "(eps_cond, eps_uncond)")
                out_arrays = {}
                for name, tensor in zip(("cond", "uncond"), supplied):
                    if name in computed:
                        continue
                    if not torch.is_tensor(tensor):
                        raise TypeError(f"reconstruct returned {type(tensor).__name__} "
                                        f"for the {name} prediction, expected a Tensor")
                    if tuple(tensor.shape) != latent_shape:
                        raise ValueError(f"reconstruct returned shape "
                                         f"{tuple(tensor.shape)} for the {name} "
                                         f"prediction, expected {latent_shape}")
                    if tensor.dtype != hcore.DTYPE:
                        raise TypeError(f"reconstruct returned dtype {tensor.dtype} for "
                                        f"the {name} prediction, expected "
                                        f"{hcore.DTYPE_NAME}")
                    out_arrays[name] = tensor
                _write(proto_out, {"ok": True, "seconds": elapsed}, out_arrays)
            elif op == "shutdown":
                _write(proto_out, {"ok": True}, {})
                return 0
            else:
                raise ValueError(f"unknown op {op!r}")
        except BaseException as exc:
            detail = f"{type(exc).__name__}: {exc}"
            try:
                _write(proto_out, {"ok": False, "error": detail}, {})
            except Exception:
                pass
            sys.stderr.write(json.dumps({"worker_error": detail}) + "\n")
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
