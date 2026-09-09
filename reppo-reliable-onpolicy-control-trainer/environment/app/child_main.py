from __future__ import annotations

import importlib.util
import json
import os
import struct
import sys
from dataclasses import dataclass

import numpy as np

_HDR = struct.Struct("<cI")


class Shutdown(BaseException):
    """Raised inside the child when the harness asks it to stop."""


@dataclass(frozen=True)
class Budget:
    """The per-run resource budget."""

    total_env_steps: int
    cpu_seconds: float
    n_checkpoints: int


class _Link:
    def __init__(self, rfd: int, wfd: int):
        self._r = rfd
        self._w = wfd
        self._policy = None

    def _send(self, kind: bytes, payload: bytes = b"") -> None:
        buf = _HDR.pack(kind, len(payload)) + payload
        while buf:
            buf = buf[os.write(self._w, buf) :]

    def _read_exact(self, n: int) -> bytes:
        out = bytearray()
        while len(out) < n:
            chunk = os.read(self._r, n - len(out))
            if not chunk:
                raise Shutdown("harness closed the pipe")
            out += chunk
        return bytes(out)

    def _recv(self) -> tuple[bytes, bytes]:
        kind, length = _HDR.unpack(self._read_exact(_HDR.size))
        return kind, self._read_exact(length)

    def _answer_query(self, payload: bytes) -> None:
        obs = np.frombuffer(payload, dtype="<f4").reshape(-1, ACT_STATE[0])
        act = np.asarray(self._policy(obs.copy()), dtype=np.float32)
        act = act.reshape(obs.shape[0], ACT_STATE[1])
        self._send(b"A", np.ascontiguousarray(act, dtype="<f4").tobytes())

    def request(self, kind: bytes, payload: bytes = b"") -> tuple[bytes, bytes]:
        self._send(kind, payload)
        while True:
            k, p = self._recv()
            if k == b"Q":
                self._answer_query(p)
            elif k == b"X":
                raise Shutdown("harness asked the run to stop")
            else:
                return k, p

    def serve_until_shutdown(self) -> None:
        while True:
            k, p = self._recv()
            if k == b"Q":
                self._answer_query(p)
            elif k == b"X":
                return


ACT_STATE = [8, 2]


class RemoteEnv:
    """The vectorised environment handle a training procedure is given."""

    def __init__(self, link: _Link, cfg: dict):
        self._link = link
        self.n_envs = int(cfg["n_envs"])
        self.obs_dim = int(cfg["obs_dim"])
        self.act_dim = int(cfg["act_dim"])
        self.max_episode_steps = int(cfg["max_episode_steps"])

    def reset(self) -> np.ndarray:
        _, payload = self._link.request(b"R")
        return np.frombuffer(payload, dtype="<f4").reshape(self.n_envs, self.obs_dim).copy()

    def step(self, actions):
        a = np.ascontiguousarray(np.asarray(actions, dtype=np.float32).reshape(
            self.n_envs, self.act_dim), dtype="<f4")
        kind, payload = self._link.request(b"S", a.tobytes())
        if kind == b"E":
            raise RuntimeError("environment-step budget exhausted")
        n, o = self.n_envs, self.obs_dim
        obs = np.frombuffer(payload, dtype="<f4", count=n * o).reshape(n, o).copy()
        reward = np.frombuffer(payload, dtype="<f4", count=n, offset=n * o * 4).copy()
        done = np.frombuffer(payload, dtype=np.uint8, count=n, offset=(n * o + n) * 4).astype(bool)
        return obs, reward, done


def _load_train(solution_dir: str):
    path = os.path.join(solution_dir, "trainer.py")
    if solution_dir not in sys.path:
        sys.path.insert(0, solution_dir)
    spec = importlib.util.spec_from_file_location("trainer", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["trainer"] = module
    spec.loader.exec_module(module)
    fn = getattr(module, "train", None)
    if not callable(fn):
        raise AttributeError("trainer.py defines no callable named 'train'")
    return fn


def main() -> int:
    solution_dir, rfd, wfd = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    link = _Link(rfd, wfd)
    kind, payload = link._recv()
    if kind != b"I":
        return 2
    cfg = json.loads(payload.decode())
    ACT_STATE[0] = int(cfg["obs_dim"])
    ACT_STATE[1] = int(cfg["act_dim"])

    env = RemoteEnv(link, cfg)
    budget = Budget(
        total_env_steps=int(cfg["total_env_steps"]),
        cpu_seconds=float(cfg["cpu_seconds"]),
        n_checkpoints=int(cfg["n_checkpoints"]),
    )

    def report(policy_fn) -> None:
        if not callable(policy_fn):
            raise TypeError("report() expects a callable policy")
        link._policy = policy_fn
        link.request(b"P")

    train = _load_train(solution_dir)
    try:
        link.request(b"G")
        train(env, int(cfg["seed"]), budget, report)
        link.request(b"D")
        link.serve_until_shutdown()
    except Shutdown:
        return 0
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Shutdown:
        sys.exit(0)
