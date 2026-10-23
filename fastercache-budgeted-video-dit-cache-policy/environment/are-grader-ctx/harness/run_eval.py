from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from harness import hcore
from harness.channel import PolicyChannel, PolicyError

HARNESS_DIR = Path(__file__).resolve().parent

CANDIDATE_ENV = {
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "HOME": "/home/agent",
    "LANG": "C.UTF-8",
    "PYTHONHASHSEED": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONSAFEPATH": "1",
    "PYTHONPATH": "",
    "PYTHONUNBUFFERED": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "NVIDIA_VISIBLE_DEVICES": "all",
    "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}


def evaluate(prompts_dir: str | Path, solution_dir: str | Path, work_dir: str | Path,
             worker_path: str | Path = HARNESS_DIR / "worker.py",
             drop_privileges: bool = False, limit: int | None = None,
             offset: int = 0,
             reference_cache: str | Path | None = None,
             model_dir: str = hcore.MODEL_DIR, progress: bool = False,
             shape: dict | None = None, harness: "hcore.Harness | None" = None) -> dict:
    """Score one submission.  Never raises for a submission-side fault."""
    hcore.set_determinism()
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"valid": False, "reason": None, "prompts": [], "metric": None,
                    "worker_stderr": ""}

    prompts = hcore.load_prompt_set(prompts_dir)[offset:]
    if limit is not None:
        prompts = prompts[:limit]
    if not prompts:
        report["reason"] = "no prompts to evaluate"
        return report

    if harness is None:
        harness = hcore.Harness(model_dir=model_dir, **(shape or {}))
    negative = hcore.load_embedding(Path(prompts_dir) / "embeds" / "negative.safetensors",
                                    harness.device)

    channel = None
    try:
        channel = PolicyChannel(worker_path, solution_dir, dict(CANDIDATE_ENV),
                                work_dir / "worker_stderr.log",
                                drop_privileges=drop_privileges, cwd=work_dir)
        channel.init(harness.num_steps, harness.budget, harness.latent_shape)

        scores = []
        for entry in prompts:
            t_prompt = time.monotonic()
            embeds = hcore.load_embedding(entry["embedding_path"], harness.device)
            run = harness.sample(embeds, negative, int(entry["seed"]), policy=channel,
                                 deadline=t_prompt + hcore.PROMPT_SECONDS)
            reference = _reference_video(harness, embeds, negative, entry, reference_cache)
            candidate = harness.decode(run["latents"])
            per_frame = hcore.frame_psnr(reference, candidate)
            psnr = float(np.mean(per_frame))
            scores.append(psnr)
            report["prompts"].append({
                "id": entry["id"],
                "psnr_db": psnr,
                "min_frame_psnr_db": float(np.min(per_frame)),
                "units_spent": run["units_spent"],
                "policy_seconds": round(run["policy_seconds"], 3),
                "wall_seconds": round(time.monotonic() - t_prompt, 1),
                "mode_counts": {m: run["modes"].count(m) for m in hcore.MODES},
            })
            if progress:
                print(f"[{entry['id']}] psnr={psnr:.3f} dB  units={run['units_spent']}  "
                      f"policy={run['policy_seconds']:.2f}s  "
                      f"wall={time.monotonic() - t_prompt:.1f}s", flush=True)
        report["metric"] = float(np.mean(scores))
        report["valid"] = True
    except (PolicyError, TimeoutError, ValueError, TypeError, OSError,
            RuntimeError, MemoryError) as exc:
        report["reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        if channel is not None:
            report["worker_stderr"] = channel.stderr_tail()
            channel.close()
    return report


def _reference_video(harness: hcore.Harness, embeds: torch.Tensor,
                     negative: torch.Tensor, entry: dict,
                     reference_cache: str | Path | None) -> np.ndarray:
    cache_path = None
    if reference_cache is not None:
        cache_path = Path(reference_cache) / f"{entry['id']}.npy"
        if cache_path.is_file():
            return np.load(cache_path)
    run = harness.sample(embeds, negative, int(entry["seed"]), policy=None)
    video = harness.decode(run["latents"])
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, video)
    return video


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", default="/app/prompts")
    parser.add_argument("--solution", default="/app/output")
    parser.add_argument("--work-dir", default="/app/dev_work")
    parser.add_argument("--out", default="/app/dev_work/report.json")
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate only the first N prompts")
    parser.add_argument("--offset", type=int, default=0,
                        help="skip the first N prompts")
    parser.add_argument("--reference-cache", default="/app/dev_work/reference_cache",
                        help="reuse decoded reference videos across runs")
    args = parser.parse_args()

    report = evaluate(args.prompts, args.solution, args.work_dir, limit=args.limit,
                      offset=args.offset,
                      reference_cache=args.reference_cache, progress=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    if report["valid"]:
        print(f"\nmean PSNR = {report['metric']:.4f} dB over "
              f"{len(report['prompts'])} prompt(s)")
        return 0
    print(f"\nINVALID: {report['reason']}", file=sys.stderr)
    if report["worker_stderr"]:
        print(report["worker_stderr"], file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
