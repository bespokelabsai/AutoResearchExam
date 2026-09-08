from __future__ import annotations

import json
import math
import os
import random
import struct
import time
from pathlib import Path

import numpy as np
import torch

MODEL_DIR = "/opt/assets/cogvideox"

NUM_INFERENCE_STEPS = 50
BUDGET_UNITS = 62
GUIDANCE_SCALE = 6.0
HEIGHT = 480
WIDTH = 720
NUM_FRAMES = 49
LATENT_SHAPE = (1, 13, 16, 60, 90)
DTYPE = torch.bfloat16
DTYPE_NAME = "bfloat16"
UNIT_COST = {"full": 2, "cond_only": 1, "skip": 0}
MODES = ("full", "cond_only", "skip")

POLICY_SECONDS_PER_PROMPT = 60.0
PROMPT_SECONDS = 900.0
WORKER_STARTUP_SECONDS = 180.0
MAX_DELIVERABLE_BYTES = 256 * 1024 * 1024

PSNR_MAX_DB = 100.0
SEED_TORCH = 20241023


def set_determinism() -> None:
    """Pin every source of run-to-run variation this harness can reach."""
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    torch.set_num_threads(1)
    random.seed(SEED_TORCH)
    np.random.seed(SEED_TORCH)
    torch.manual_seed(SEED_TORCH)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED_TORCH)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


class Harness:
    """Owns the transformer, the VAE and the scheduler."""

    def __init__(self, model_dir: str = MODEL_DIR, device: str = "cuda",
                 num_frames: int = NUM_FRAMES, height: int = HEIGHT, width: int = WIDTH,
                 num_steps: int = NUM_INFERENCE_STEPS, budget: int | None = None):
        from diffusers import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel
        from diffusers.schedulers import CogVideoXDDIMScheduler

        self.device = torch.device(device)
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.num_steps = num_steps
        self.budget = BUDGET_UNITS if budget is None else int(budget)
        self.transformer = CogVideoXTransformer3DModel.from_pretrained(
            model_dir, subfolder="transformer", torch_dtype=DTYPE).to(self.device).eval()
        self.vae = AutoencoderKLCogVideoX.from_pretrained(
            model_dir, subfolder="vae", torch_dtype=DTYPE).to(self.device).eval()
        self.vae.enable_tiling()
        self.scheduler = CogVideoXDDIMScheduler.from_pretrained(model_dir, subfolder="scheduler")
        self.scaling_factor = float(self.vae.config.scaling_factor)
        self.latent_shape = (
            1,
            (num_frames - 1) // 4 + 1,
            int(self.transformer.config.in_channels),
            height // 8,
            width // 8,
        )
        self.forward_calls = 0

    @torch.no_grad()
    def _forward(self, latents: torch.Tensor, embeds: torch.Tensor,
                 t: torch.Tensor) -> torch.Tensor:
        timestep = t.expand(latents.shape[0])
        out = self.transformer(
            hidden_states=latents,
            encoder_hidden_states=embeds,
            timestep=timestep,
            image_rotary_emb=None,
            return_dict=False,
        )[0]
        self.forward_calls += 1
        return out.to(DTYPE)

    @torch.no_grad()
    def sample(self, prompt_embeds: torch.Tensor, negative_embeds: torch.Tensor, seed: int,
               policy=None, deadline: float | None = None) -> dict:
        """Run the fixed schedule.

        ``policy`` is ``None`` for the uncached reference (every step decided ``full``),
        or a :class:`PolicyChannel`-like object exposing ``begin(...)``, ``decide(...)``
        and ``reconstruct(...)``.

        Returns a dict with the final latents and the accounting for the run.
        """
        from diffusers.utils.torch_utils import randn_tensor

        scheduler = self.scheduler
        scheduler.set_timesteps(self.num_steps, device=self.device)
        timesteps = scheduler.timesteps

        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        latents = randn_tensor(self.latent_shape, generator=generator, device=self.device,
                               dtype=DTYPE)
        latents = latents * scheduler.init_noise_sigma

        step_stride = scheduler.config.num_train_timesteps // self.num_steps
        alphas = scheduler.alphas_cumprod
        units_spent = 0
        modes: list[str] = []
        policy_seconds = 0.0

        if policy is not None:
            policy_seconds += policy.begin(self.num_steps, self.budget, self.latent_shape)

        for i, t in enumerate(timesteps):
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError(f"prompt wall-clock budget exhausted at step {i}")
            t_int = int(t.item())
            prev_t = t_int - step_stride
            state = {
                "step": i,
                "num_steps": self.num_steps,
                "timestep": t_int,
                "prev_timestep": prev_t,
                "alpha_cumprod": float(alphas[t_int]),
                "alpha_cumprod_prev": float(alphas[prev_t]) if prev_t >= 0
                else float(scheduler.final_alpha_cumprod),
                "units_spent": units_spent,
                "units_remaining": self.budget - units_spent,
                "steps_remaining": self.num_steps - i,
            }

            if policy is None:
                mode = "full"
            else:
                mode, spent = policy.decide(state, latents)
                policy_seconds += spent
                if mode not in MODES:
                    raise ValueError(f"step {i}: decide returned {mode!r}, "
                                     f"expected one of {MODES}")
                if UNIT_COST[mode] > self.budget - units_spent:
                    raise ValueError(
                        f"step {i}: decide returned {mode!r} costing {UNIT_COST[mode]} "
                        f"units with only {self.budget - units_spent} remaining")
            modes.append(mode)
            units_spent += UNIT_COST[mode]

            computed: dict[str, torch.Tensor] = {}
            if mode in ("full", "cond_only"):
                computed["cond"] = self._forward(latents, prompt_embeds, t)
            if mode == "full":
                computed["uncond"] = self._forward(latents, negative_embeds, t)

            if policy is None:
                eps_cond, eps_uncond = computed["cond"], computed["uncond"]
            else:
                supplied, spent = policy.reconstruct(state, mode, computed)
                policy_seconds += spent
                eps_cond = computed.get("cond", supplied.get("cond"))
                eps_uncond = computed.get("uncond", supplied.get("uncond"))
                if eps_cond is None or eps_uncond is None:
                    raise ValueError(f"step {i}: reconstruct did not supply the "
                                     f"missing prediction(s) for mode {mode!r}")
                if policy_seconds > POLICY_SECONDS_PER_PROMPT:
                    raise TimeoutError(
                        f"step {i}: policy used {policy_seconds:.1f}s of its "
                        f"{POLICY_SECONDS_PER_PROMPT:.0f}s per-prompt budget")

            noise_pred = eps_uncond.float() + GUIDANCE_SCALE * (
                eps_cond.float() - eps_uncond.float())
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0].to(DTYPE)

        if policy is not None and units_spent > self.budget:
            raise ValueError(f"policy spent {units_spent} units, budget is {self.budget}")
        return {
            "latents": latents,
            "units_spent": units_spent,
            "modes": modes,
            "policy_seconds": policy_seconds,
        }

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> np.ndarray:
        """Latents -> uint8 RGB frames, shape (num_frames, height, width, 3)."""
        x = latents.permute(0, 2, 1, 3, 4) / self.scaling_factor
        frames = self.vae.decode(x.to(DTYPE)).sample.float()
        frames = (frames / 2.0 + 0.5).clamp(0.0, 1.0)
        frames = frames[0].permute(1, 2, 3, 0).cpu().numpy()
        return (frames * 255.0).round().clip(0, 255).astype(np.uint8)


def frame_psnr(ref: np.ndarray, cand: np.ndarray) -> list[float]:
    """Per-frame PSNR in dB between two uint8 videos of identical shape."""
    if ref.shape != cand.shape:
        raise ValueError(f"shape mismatch {ref.shape} vs {cand.shape}")
    out = []
    for j in range(ref.shape[0]):
        mse = float(np.mean((ref[j].astype(np.float64) - cand[j].astype(np.float64)) ** 2))
        out.append(PSNR_MAX_DB if mse == 0.0 else 10.0 * math.log10(255.0 ** 2 / mse))
    return out


def video_psnr(ref: np.ndarray, cand: np.ndarray) -> float:
    per_frame = frame_psnr(ref, cand)
    return float(sum(per_frame) / len(per_frame))


def load_embedding(path: str | Path, device: torch.device) -> torch.Tensor:
    from safetensors.torch import load_file

    tensors = load_file(str(path))
    return tensors["prompt_embeds"].to(device=device, dtype=DTYPE)


def load_prompt_set(directory: str | Path) -> list[dict]:
    """Read ``prompts.json`` and pair each entry with its embedding file."""
    directory = Path(directory)
    entries = json.loads((directory / "prompts.json").read_text())
    for entry in entries:
        entry["embedding_path"] = str(directory / "embeds" / f"{entry['id']}.safetensors")
    return entries


def encode_frame(header: dict, arrays: dict[str, torch.Tensor]) -> bytes:
    """Serialise a header plus named bf16 tensors into one length-prefixed frame."""
    meta = []
    payload = []
    for name, tensor in arrays.items():
        t = tensor.detach().to(device="cpu", dtype=DTYPE).contiguous()
        raw = t.view(torch.uint8).numpy().tobytes()
        meta.append({"name": name, "shape": list(t.shape), "nbytes": len(raw)})
        payload.append(raw)
    header = dict(header)
    header["arrays"] = meta
    head = json.dumps(header).encode("utf-8")
    body = b"".join(payload)
    return struct.pack(">II", len(head), len(body)) + head + body


def _read_exact(stream, nbytes: int, deadline: float) -> bytes:
    chunks = []
    remaining = nbytes
    while remaining > 0:
        if time.monotonic() > deadline:
            raise TimeoutError("timed out reading from the policy worker")
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("policy worker closed its output stream")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame(stream, deadline: float) -> tuple[dict, dict[str, torch.Tensor]]:
    """Read one frame.  Every array is validated against its declared shape."""
    head_len, body_len = struct.unpack(">II", _read_exact(stream, 8, deadline))
    if head_len > 1 << 20 or body_len > 1 << 30:
        raise ValueError("policy worker sent an oversized frame")
    header = json.loads(_read_exact(stream, head_len, deadline).decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("policy worker sent a malformed header")
    body = _read_exact(stream, body_len, deadline)
    arrays: dict[str, torch.Tensor] = {}
    offset = 0
    for meta in header.get("arrays", []):
        name = str(meta["name"])
        shape = tuple(int(v) for v in meta["shape"])
        nbytes = int(meta["nbytes"])
        expected = 2
        for dim in shape:
            expected *= dim
        if nbytes != expected or offset + nbytes > len(body):
            raise ValueError(f"policy worker sent a malformed array {name!r}")
        buf = bytearray(body[offset:offset + nbytes])
        offset += nbytes
        arrays[name] = torch.frombuffer(buf, dtype=DTYPE).reshape(shape)
    return header, arrays
