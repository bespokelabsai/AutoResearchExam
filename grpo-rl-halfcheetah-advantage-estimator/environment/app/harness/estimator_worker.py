import os
import sys

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "2"

import importlib.util
import json
import struct
import traceback

import numpy as np

WORKER_THREADS = 2

BLOCKED_ROOTS = frozenset(
    {
        "gymnasium",
        "gym",
        "mujoco",
        "mujoco_py",
        "dm_control",
        "dm_env",
        "envpool",
        "brax",
        "pybullet",
        "gymnasium_robotics",
    }
)


class _BlockedImportFinder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in BLOCKED_ROOTS:
            raise ImportError(f"import of {fullname!r} is not permitted inside the estimator")
        return None


sys.meta_path.insert(0, _BlockedImportFinder())


def _read_exactly(stream, n):
    chunks = []
    got = 0
    while got < n:
        chunk = stream.read(n - got)
        if not chunk:
            raise EOFError("harness closed the pipe")
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def _send_ok(out, adv):
    out.write(b"O")
    out.write(adv.tobytes(order="C"))
    out.flush()


def _send_err(out, text):
    payload = text.encode("utf-8", "replace")[:4000]
    out.write(b"X")
    out.write(struct.pack("<I", len(payload)))
    out.write(payload)
    out.flush()


def _load_estimator_class(submission_dir):
    entry = os.path.join(submission_dir, "advantage.py")
    if not os.path.isfile(entry):
        raise FileNotFoundError(f"{entry} does not exist")
    if submission_dir not in sys.path:
        sys.path.insert(0, submission_dir)
    spec = importlib.util.spec_from_file_location("submitted_advantage", entry)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {entry}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["submitted_advantage"] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "AdvantageEstimator"):
        raise AttributeError("advantage.py defines no class named AdvantageEstimator")
    return module.AdvantageEstimator


def main():
    submission_dir = sys.argv[1]
    inp, out = sys.stdin.buffer, sys.stdout.buffer

    try:
        import torch

        torch.set_num_threads(WORKER_THREADS)
    except Exception:
        pass

    estimator = None
    shapes = None
    while True:
        cmd = inp.read(1)
        if not cmd or cmd == b"Q":
            return 0

        if cmd == b"I":
            (length,) = struct.unpack("<I", _read_exactly(inp, 4))
            cfg = json.loads(_read_exactly(inp, length).decode("utf-8"))
            T, N = cfg["num_steps"], cfg["num_envs"]
            O, A = cfg["obs_dim"], cfg["act_dim"]
            shapes = (T, N, O, A)
            try:
                cls = _load_estimator_class(submission_dir)
                estimator = cls(
                    obs_dim=O,
                    act_dim=A,
                    num_envs=N,
                    num_steps=T,
                    total_iterations=cfg["total_iterations"],
                )
            except BaseException:
                _send_err(out, "constructing AdvantageEstimator failed:\n" + traceback.format_exc())
                return 1
            out.write(b"O")
            out.flush()
            continue

        if cmd != b"E" or estimator is None or shapes is None:
            _send_err(out, f"protocol error: unexpected command {cmd!r}")
            return 1

        T, N, O, A = shapes
        (iteration,) = struct.unpack("<i", _read_exactly(inp, 4))
        frame_bytes = (T * N * O) * 4 * 2 + (T * N * A) * 4 + (T * N) * 4 * 2 + (T * N) * 2
        buf = _read_exactly(inp, frame_bytes)
        off = 0

        def take(count, dtype, shape):
            nonlocal off
            width = np.dtype(dtype).itemsize * count
            arr = np.frombuffer(buf, dtype=dtype, count=count, offset=off).reshape(shape)
            off += width
            return arr.copy()

        rollout = {
            "obs": take(T * N * O, np.float32, (T, N, O)),
            "next_obs": take(T * N * O, np.float32, (T, N, O)),
            "actions": take(T * N * A, np.float32, (T, N, A)),
            "logprobs": take(T * N, np.float32, (T, N)),
            "rewards": take(T * N, np.float32, (T, N)),
            "terminations": take(T * N, np.uint8, (T, N)).astype(bool),
            "truncations": take(T * N, np.uint8, (T, N)).astype(bool),
            "iteration": int(iteration),
        }

        try:
            adv = estimator.estimate(rollout)
        except BaseException:
            _send_err(out, "estimate() raised:\n" + traceback.format_exc())
            return 1

        if not isinstance(adv, np.ndarray):
            _send_err(out, f"estimate() returned {type(adv).__name__}, expected numpy.ndarray")
            return 1
        if adv.shape != (T, N):
            _send_err(out, f"estimate() returned shape {adv.shape}, expected {(T, N)}")
            return 1
        if adv.dtype != np.float32:
            _send_err(out, f"estimate() returned dtype {adv.dtype}, expected float32")
            return 1
        if not adv.flags["C_CONTIGUOUS"]:
            _send_err(out, "estimate() returned a non C-contiguous array")
            return 1
        if not np.isfinite(adv).all():
            _send_err(out, "estimate() returned a non-finite entry")
            return 1
        _send_ok(out, adv)


if __name__ == "__main__":
    sys.exit(main())
