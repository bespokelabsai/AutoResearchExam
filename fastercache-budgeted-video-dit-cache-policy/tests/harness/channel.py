from __future__ import annotations

import os
import select
import signal
import subprocess
import time
from pathlib import Path

import torch

from harness import hcore


class PolicyError(RuntimeError):
    """Raised for anything the submitted policy did wrong."""


class _DeadlineReader:
    """File-like ``read`` over a raw fd that never blocks past ``self.deadline``."""

    def __init__(self, fd: int):
        self.fd = fd
        self.deadline = time.monotonic()

    def read(self, nbytes: int) -> bytes:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("policy worker exceeded its response deadline")
        ready, _, _ = select.select([self.fd], [], [], remaining)
        if not ready:
            raise TimeoutError("policy worker exceeded its response deadline")
        return os.read(self.fd, nbytes)


class PolicyChannel:
    """Runs one worker process and speaks the frame protocol to it."""

    def __init__(self, worker_path: str | Path, solution_dir: str | Path,
                 env: dict[str, str], stderr_path: str | Path,
                 drop_privileges: bool = False, cwd: str | Path = "/"):
        self.latent_shape = tuple(hcore.LATENT_SHAPE)
        self.stderr_path = Path(stderr_path)
        self._stderr_file = open(self.stderr_path, "wb")
        argv = ["setsid"]
        if drop_privileges:
            argv += ["runuser", "-u", "agent", "--"]
        argv += ["/usr/bin/python3", str(worker_path), "--solution", str(solution_dir)]
        self.proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._stderr_file, env=env, cwd=str(cwd), close_fds=True)
        self.reader = _DeadlineReader(self.proc.stdout.fileno())

    def _exchange(self, header: dict, arrays: dict, timeout: float):
        try:
            self.proc.stdin.write(hcore.encode_frame(header, arrays))
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError) as exc:
            raise PolicyError(f"policy worker is not accepting input: {exc}") from exc
        self.reader.deadline = time.monotonic() + timeout
        try:
            reply, out_arrays = hcore.read_frame(self.reader, self.reader.deadline)
        except (TimeoutError, EOFError, ValueError, OSError) as exc:
            raise PolicyError(f"{type(exc).__name__}: {exc}") from exc
        if not reply.get("ok"):
            raise PolicyError(str(reply.get("error", "policy worker reported a failure")))
        return reply, out_arrays

    @staticmethod
    def _scalar_state(state: dict) -> dict:
        return {k: v for k, v in state.items() if k != "latents"}

    def init(self, num_steps: int, budget: int, latent_shape) -> None:
        self.latent_shape = tuple(int(v) for v in latent_shape)
        self._exchange({"op": "init", "num_steps": num_steps, "budget": budget,
                        "latent_shape": list(latent_shape)}, {},
                       hcore.WORKER_STARTUP_SECONDS)

    def begin(self, num_steps: int, budget: int, latent_shape) -> float:
        t0 = time.monotonic()
        self._exchange({"op": "begin"}, {}, hcore.POLICY_SECONDS_PER_PROMPT)
        return time.monotonic() - t0

    def decide(self, state: dict, latents: torch.Tensor) -> tuple[str, float]:
        t0 = time.monotonic()
        reply, _ = self._exchange({"op": "decide", "state": self._scalar_state(state)},
                                  {"latents": latents}, hcore.POLICY_SECONDS_PER_PROMPT)
        return reply.get("mode"), time.monotonic() - t0

    def reconstruct(self, state: dict, mode: str,
                    computed: dict[str, torch.Tensor]) -> tuple[dict, float]:
        t0 = time.monotonic()
        _, arrays = self._exchange(
            {"op": "reconstruct", "mode": mode, "state": self._scalar_state(state)},
            computed, hcore.POLICY_SECONDS_PER_PROMPT)
        elapsed = time.monotonic() - t0
        out = {}
        for name in ("cond", "uncond"):
            if name in computed or name not in arrays:
                continue
            tensor = arrays[name]
            if tuple(tensor.shape) != self.latent_shape:
                raise PolicyError(f"reconstruct returned shape {tuple(tensor.shape)} for "
                                  f"the {name} prediction, expected {self.latent_shape}")
            if not bool(torch.isfinite(tensor.float()).all()):
                raise PolicyError(f"reconstruct returned non-finite values for the "
                                  f"{name} prediction")
            out[name] = tensor.to(device="cuda" if torch.cuda.is_available() else "cpu",
                                  dtype=hcore.DTYPE)
        return out, elapsed

    def stderr_tail(self, limit: int = 4000) -> str:
        try:
            data = self.stderr_path.read_bytes()
        except OSError:
            return ""
        return data[-limit:].decode("utf-8", "replace")

    def close(self) -> None:
        try:
            self._exchange({"op": "shutdown"}, {}, 10.0)
        except Exception:
            pass
        self.kill()

    def kill(self) -> None:
        for stream in (self.proc.stdin, self.proc.stdout):
            try:
                stream.close()
            except Exception:
                pass
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        try:
            self.proc.wait(timeout=30)
        except Exception:
            pass
        try:
            self._stderr_file.close()
        except Exception:
            pass
