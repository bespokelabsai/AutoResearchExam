#!/usr/bin/env python3
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import grader_core

LOGS = Path("/logs/verifier")
METRIC_PATH = LOGS / "metric.json"
CTRF_PATH = LOGS / "ctrf.json"

DELIVERABLE = Path("/app/output/train.py")
MAX_SOURCE_BYTES = 102400
MAX_CHECKPOINT_BYTES = 209715200
TRAIN_DATA = Path("/app/data/cifar10_train.npz")
ARCH_SOURCE = Path("/tests/preact_resnet.py")
RUNNER_SOURCE = Path("/tests/candidate_runner.py")
BUDGET_SEC = 180.0
GRACE_SEC = 5.0

RUN_DIR = Path("/tmp/graded_run")
SEALED_DIR = Path("/tmp/verifier_ckpt")
CKPT_NAME = "model.pt"
AGENT_USER = "agent"
AGENT_UID = 1001
STREAM_TAIL_CHARS = 4000

GRADED_SPLIT = os.environ.get("GRADED_SPLIT", "final")

METRIC_NAME = (
    f"PGD-50 robust accuracy (percent) on the sealed {grader_core.SPLIT_ROWS}-image "
    f"'{GRADED_SPLIT}' half of the CIFAR-10 test split, L-inf eps=8/255, alpha=2/255, "
    "50 iterations, 10 random restarts, worst restart by cross-entropy per example"
)


def _tail(path, limit=STREAM_TAIL_CHARS):
    try:
        path = Path(path)
        if path.is_symlink():
            return "<symlink>"
        text = path.read_text(errors="replace")
    except Exception:
        return ""
    return text[-limit:]


def write_metric(payload):
    LOGS.mkdir(parents=True, exist_ok=True)
    METRIC_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    passed = bool(payload.get("valid")) and payload.get("reward", 0.0) > 0.0
    CTRF_PATH.write_text(
        json.dumps(
            {
                "results": {
                    "tool": {"name": "robust-cifar10-verifier"},
                    "summary": {
                        "tests": 1,
                        "passed": 1 if passed else 0,
                        "failed": 0 if passed else 1,
                        "pending": 0,
                        "skipped": 0,
                        "other": 0,
                        "start": 0,
                        "stop": 0,
                    },
                    "tests": [
                        {
                            "name": "graded_robust_accuracy",
                            "status": "passed" if passed else "failed",
                            "duration": 0,
                            "message": payload.get("failure_reason") or METRIC_NAME,
                        }
                    ],
                }
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def invalid(reason, **extra):
    payload = {
        "metric_name": METRIC_NAME,
        "metric": None,
        "reward": 0.0,
        "valid": False,
        "failure_reason": reason,
    }
    payload.update(extra)
    write_metric(payload)
    return 0


def gpu_diagnostics():
    """What the ROOT process can see, so a rollout of zeros can be read correctly."""
    info = {}
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["device_count"] = int(torch.cuda.device_count())
        if info["cuda_available"] and info["device_count"]:
            info["device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    for var in ("NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES", "CUDA_VISIBLE_DEVICES"):
        info[var] = os.environ.get(var)
    info["dev_nvidia_nodes"] = sorted(
        p.name for p in Path("/dev").glob("nvidia*") if p.exists()
    )
    return info


def prepare_run_dir():
    if RUN_DIR.exists():
        shutil.rmtree(RUN_DIR)
    RUN_DIR.mkdir(parents=True)
    (RUN_DIR / ".cache").mkdir()
    for path in (RUN_DIR, RUN_DIR / ".cache"):
        os.chown(path, AGENT_UID, AGENT_UID)
        os.chmod(path, 0o700)


def child_environment():
    """The environment the graded run gets.  Deliberately minimal, seeded, and thread-pinned;
    the NVIDIA variables are passed through untouched rather than narrowed, so the GPU is as
    visible to uid 1001 as it is to root.  This dict is built from scratch rather than inherited
    from the verifier image's own ENV, so every seed and thread-count pin set there (see
    tests/Dockerfile) is repeated here explicitly; the two lists drifting apart is the failure
    mode, not either one being wrong in isolation."""
    env = {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME": str(RUN_DIR),
        "TMPDIR": str(RUN_DIR),
        "PWD": str(RUN_DIR),
        "LANG": "C.UTF-8",
        "PYTHONPATH": "/app",
        "PYTHONHASHSEED": "0",
        "GRADER_SEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "XDG_CACHE_HOME": str(RUN_DIR / ".cache"),
        "TRITON_CACHE_DIR": str(RUN_DIR / ".cache" / "triton"),
        "MPLCONFIGDIR": str(RUN_DIR / ".cache" / "mpl"),
    }
    for var in (
        "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
        "NVIDIA_REQUIRE_CUDA",
        "LD_LIBRARY_PATH",
        "CUDA_HOME",
    ):
        if var in os.environ:
            env[var] = os.environ[var]
    return env


def kill_process_group(proc):
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, OSError):
        pgid = None
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    try:
        proc.wait(timeout=GRACE_SEC)
    except Exception:
        pass
    try:
        subprocess.run(["pkill", "-9", "-u", AGENT_USER], check=False, timeout=30)
    except Exception:
        pass


def run_deliverable():
    """Run the submitted training program as uid 1001 under the wall-clock budget.

    Returns (info, checkpoint_path_or_None).
    """
    shutil.copyfile(DELIVERABLE, RUN_DIR / "train.py")
    os.chmod(RUN_DIR / "train.py", 0o444)
    shutil.copyfile(RUNNER_SOURCE, RUN_DIR / "candidate_runner.py")
    os.chmod(RUN_DIR / "candidate_runner.py", 0o444)
    shutil.copyfile(ARCH_SOURCE, RUN_DIR / "preact_resnet.py")
    os.chmod(RUN_DIR / "preact_resnet.py", 0o444)
    out_path = RUN_DIR / CKPT_NAME
    stdout_path, stderr_path = RUN_DIR / "stdout.txt", RUN_DIR / "stderr.txt"

    cmd = [
        "runuser", "-u", AGENT_USER, "--",
        "/usr/bin/python3", "-u", str(RUN_DIR / "candidate_runner.py"),
        str(RUN_DIR / "train.py"),
        "--data", str(TRAIN_DATA),
        "--out", str(out_path),
    ]
    started = time.monotonic()
    with open(stdout_path, "wb") as so, open(stderr_path, "wb") as se:
        proc = subprocess.Popen(
            cmd,
            cwd=str(RUN_DIR),
            env=child_environment(),
            stdout=so,
            stderr=se,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        timed_out = False
        try:
            returncode = proc.wait(timeout=BUDGET_SEC)
        except subprocess.TimeoutExpired:
            returncode, timed_out = None, True
    elapsed = time.monotonic() - started
    kill_process_group(proc)

    info = {
        "returncode": returncode,
        "timed_out": timed_out,
        "wall_clock_sec": round(elapsed, 2),
        "budget_sec": BUDGET_SEC,
        "stdout_tail": _tail(stdout_path),
        "stderr_tail": _tail(stderr_path),
    }
    if not out_path.exists() or not out_path.is_file() or out_path.is_symlink():
        return info, None
    size = out_path.stat().st_size
    info["checkpoint_bytes"] = size
    if size == 0 or size > MAX_CHECKPOINT_BYTES:
        return info, None

    if SEALED_DIR.exists():
        shutil.rmtree(SEALED_DIR)
    SEALED_DIR.mkdir(mode=0o700)
    sealed = SEALED_DIR / CKPT_NAME
    shutil.copyfile(out_path, sealed)
    os.chmod(sealed, 0o400)
    return info, sealed


def load_model(checkpoint, device):
    import torch

    import preact_resnet

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state and not any(
        isinstance(k, str) and k.startswith("conv1.") for k in state
    ):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint is {type(state).__name__}, expected a state_dict")
    cleaned = {}
    for key, value in state.items():
        if not isinstance(key, str):
            raise ValueError("state_dict keys must be strings")
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"state_dict['{key}'] is {type(value).__name__}, expected a tensor")
        tensor = value.detach().to("cpu")
        if tensor.is_floating_point():
            tensor = tensor.float()
            if not torch.isfinite(tensor).all():
                raise ValueError(f"state_dict['{key}'] contains non-finite values")
        cleaned[key.removeprefix("module.")] = tensor

    model = preact_resnet.PreActResNet18()
    model.load_state_dict(cleaned, strict=True)
    model = model.float().eval().to(device)
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"loaded parameter '{name}' is non-finite")
    return model


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    gpu = gpu_diagnostics()

    if not gpu.get("cuda_available") or not gpu.get("device_count"):
        return invalid("gpu_unavailable_in_verifier", gpu=gpu,
                       detail="torch.cuda reports no usable device to the ROOT grader; this is "
                              "an environment defect, not a property of the submission")

    if GRADED_SPLIT not in grader_core.SPLITS:
        return invalid("unknown_graded_split", gpu=gpu, split=GRADED_SPLIT,
                       detail="GRADED_SPLIT names no sealed half; this is a harness defect, "
                              "not a property of the submission")

    if DELIVERABLE.is_symlink() or not DELIVERABLE.is_file():
        return invalid("missing_deliverable", gpu=gpu,
                       detail=f"{DELIVERABLE} does not exist or is not a regular file")
    source_bytes = DELIVERABLE.stat().st_size
    if source_bytes == 0:
        return invalid("empty_deliverable", gpu=gpu, deliverable_bytes=0)
    if source_bytes > MAX_SOURCE_BYTES:
        return invalid("deliverable_too_large", gpu=gpu, deliverable_bytes=source_bytes,
                       limit_bytes=MAX_SOURCE_BYTES)

    try:
        prepare_run_dir()
        run_info, checkpoint = run_deliverable()
    except Exception:
        return invalid("harness_error_launching_deliverable", gpu=gpu,
                       detail=traceback.format_exc(limit=6))

    if checkpoint is None:
        return invalid("no_valid_checkpoint", gpu=gpu, deliverable_bytes=source_bytes,
                       subprocess=run_info)

    try:
        import torch

        device = torch.device("cuda")
        model = load_model(checkpoint, device)
    except Exception:
        return invalid("checkpoint_rejected", gpu=gpu, deliverable_bytes=source_bytes,
                       subprocess=run_info, detail=traceback.format_exc(limit=6))

    try:
        result = grader_core.evaluate(model, device, GRADED_SPLIT)
    except Exception:
        return invalid("evaluation_failed", gpu=gpu, deliverable_bytes=source_bytes,
                       subprocess=run_info, detail=traceback.format_exc(limit=6))

    metric = float(result["robust_accuracy_percent"])
    reward = grader_core.reward_from_metric(metric)
    write_metric(
        {
            "metric_name": METRIC_NAME,
            "metric": metric,
            "reward": reward,
            "valid": True,
            "failure_reason": None,
            "split": GRADED_SPLIT,
            "deliverable_bytes": source_bytes,
            "evaluation": result,
            "subprocess": run_info,
            "gpu": gpu,
            "device": str(device),
        }
    )
    return 0


def test_graded_reward_recorded():
    """pytest-compatible entry point: grade the submission and assert a reward was recorded."""
    main()
    payload = json.loads(METRIC_PATH.read_text())
    assert 0.0 <= float(payload["reward"]) <= 1.0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BaseException:
        try:
            invalid("unhandled_grader_exception", detail=traceback.format_exc(limit=8))
        finally:
            sys.exit(0)
