import argparse
import importlib.util
import inspect
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import PixArtSigmaPipeline


def load_submission(path):
    spec = importlib.util.spec_from_file_location("candidate_quantize", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load entry point")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "quantize_model", None)
    if not callable(fn):
        raise TypeError("quantize_model is not callable")
    return fn


def check_tensor(value, name, shape, dtype, positive=False, code=False):
    if not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != dtype:
        raise ValueError(f"invalid {name}")
    if code:
        if value.numel() and (int(value.min()) < -7 or int(value.max()) > 7):
            raise ValueError("qweight outside INT4 range")
    else:
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"non-finite {name}")
        if positive and not bool((value > 0).all()):
            raise ValueError(f"non-positive {name}")


def validate_replacement(module, in_features, out_features, has_bias, rank):
    expected = ["qweight", "scales", "l1", "l2", "smooth"]
    tensors = {name: value for name, value in module.named_parameters(recurse=True)}
    tensors.update({name: value for name, value in module.named_buffers(recurse=True)})
    if has_bias:
        expected.append("bias")
    if list(tensors) != expected or hasattr(module, "weight"):
        raise ValueError("replacement tensor state does not match the required schema")
    groups = math.ceil(in_features / 64)
    check_tensor(tensors["qweight"], "qweight", (out_features, in_features), torch.int8, code=True)
    check_tensor(tensors["scales"], "scales", (out_features, groups), torch.float16, positive=True)
    check_tensor(tensors["smooth"], "smooth", (in_features,), torch.float16, positive=True)
    r = tensors["l1"].shape[1] if tensors["l1"].ndim == 2 else -1
    if not 0 <= r <= rank:
        raise ValueError("rank exceeds budget")
    check_tensor(tensors["l1"], "l1", (in_features, r), torch.float16)
    check_tensor(tensors["l2"], "l2", (r, out_features), torch.float16)
    if has_bias:
        check_tensor(tensors["bias"], "bias", (out_features,), torch.float16)
    elif "bias" in tensors:
        raise ValueError("unexpected bias")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rank", type=int, default=32)
    args = parser.parse_args()
    if args.rank < 0 or args.rank > 32 or not torch.cuda.is_available():
        raise RuntimeError("invalid rank or CUDA unavailable")

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    quantize_model = load_submission(args.entry)
    signature = inspect.signature(quantize_model)
    parameters = list(signature.parameters.values())
    if (
        len(parameters) != 2
        or [parameter.name for parameter in parameters] != ["pipe", "rank"]
        or parameters[0].default is not inspect.Parameter.empty
        or parameters[1].default != 32
    ):
        raise TypeError("quantize_model must have signature quantize_model(pipe, rank=32)")
    pipe = PixArtSigmaPipeline.from_pretrained("/opt/pixart-sigma", torch_dtype=torch.float16, local_files_only=True)
    pipe.to("cuda")
    originals = {
        name: (module, module.in_features, module.out_features, module.bias is not None)
        for name, module in pipe.transformer.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    result = quantize_model(pipe, rank=args.rank)
    if result is not None:
        raise ValueError("quantize_model must return None")
    current = dict(pipe.transformer.named_modules())
    if not originals:
        raise RuntimeError("no eligible linear layers found")
    for name, (old, in_features, out_features, has_bias) in originals.items():
        module = current.get(name)
        if module is None or module is old or isinstance(module, torch.nn.Linear):
            raise ValueError(f"linear layer not replaced: {name}")
        validate_replacement(module, in_features, out_features, has_bias, args.rank)
    if any(isinstance(module, torch.nn.Linear) for module in pipe.transformer.modules()):
        raise ValueError("linear layer remains")

    from safetensors.torch import save_file

    tensors = {}
    manifest = {"rank": args.rank, "layers": []}
    for name, (_, in_features, out_features, has_bias) in originals.items():
        module = current[name]
        state = {key: value for key, value in module.named_parameters(recurse=True)}
        state.update({key: value for key, value in module.named_buffers(recurse=True)})
        manifest["layers"].append({
            "name": name, "in_features": in_features, "out_features": out_features,
            "has_bias": has_bias,
        })
        for field in ("qweight", "scales", "l1", "l2", "smooth", "bias"):
            if field in state:
                tensors[f"{name}.{field}"] = state[field].detach().cpu().contiguous()
    outdir = Path(args.output)
    if not outdir.is_dir() or any(outdir.iterdir()):
        raise ValueError("output directory is not fresh")
    save_file(tensors, outdir / "quantized.safetensors")
    (outdir / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
