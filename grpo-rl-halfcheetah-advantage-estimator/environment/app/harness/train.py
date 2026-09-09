#!/usr/bin/env python3
import os
import sys

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse
import json
import struct
import subprocess
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppo_harness as H

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "estimator_worker.py")

FORBIDDEN_MAPPINGS = ("mujoco", "libglfw", "dm_control", "bullet", "envpool")


class EstimatorFailure(RuntimeError):
    pass


class SandboxViolation(RuntimeError):
    pass


class EstimatorClient:
    """Runs the submitted estimator in its own interpreter and exchanges raw arrays with it."""

    def __init__(self, submission_dir, obs_dim, act_dim):
        self.T, self.N = H.NUM_STEPS, H.NUM_ENVS
        self.O, self.A = obs_dim, act_dim
        env = dict(os.environ)
        env.pop("GRADE_SPLIT", None)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        self.proc = subprocess.Popen(
            [sys.executable, "-u", WORKER, submission_dir],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            close_fds=True,
            env=env,
        )
        cfg = json.dumps(
            {
                "obs_dim": obs_dim,
                "act_dim": act_dim,
                "num_envs": self.N,
                "num_steps": self.T,
                "total_iterations": H.NUM_ITERATIONS,
            }
        ).encode("utf-8")
        self._write(b"I" + struct.pack("<I", len(cfg)) + cfg)
        self._expect_ok()

    def _write(self, payload):
        try:
            self.proc.stdin.write(payload)
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            raise EstimatorFailure(self._death_note())

    def _read(self, n):
        chunks, got = [], 0
        while got < n:
            chunk = self.proc.stdout.read(n - got)
            if not chunk:
                raise EstimatorFailure(self._death_note())
            chunks.append(chunk)
            got += len(chunk)
        return b"".join(chunks)

    def _death_note(self):
        code = self.proc.poll()
        return f"the estimator process exited (status {code})"

    def _expect_ok(self):
        tag = self._read(1)
        if tag == b"O":
            return None
        if tag == b"X":
            (length,) = struct.unpack("<I", self._read(4))
            raise EstimatorFailure(self._read(length).decode("utf-8", "replace"))
        raise EstimatorFailure(f"protocol error: the estimator sent {tag!r}")

    def estimate(self, rollout):
        payload = [b"E", struct.pack("<i", rollout["iteration"])]
        for key, dtype in (
            ("obs", np.float32),
            ("next_obs", np.float32),
            ("actions", np.float32),
            ("logprobs", np.float32),
            ("rewards", np.float32),
            ("terminations", np.uint8),
            ("truncations", np.uint8),
        ):
            payload.append(np.ascontiguousarray(rollout[key], dtype=dtype).tobytes(order="C"))
        self._write(b"".join(payload))
        tag = self._read(1)
        if tag == b"X":
            (length,) = struct.unpack("<I", self._read(4))
            raise EstimatorFailure(self._read(length).decode("utf-8", "replace"))
        if tag != b"O":
            raise EstimatorFailure(f"protocol error: the estimator sent {tag!r}")
        adv = np.frombuffer(self._read(self.T * self.N * 4), dtype=np.float32).reshape(
            self.T, self.N
        ).copy()
        if not np.isfinite(adv).all():
            raise EstimatorFailure("the returned advantages contain a non-finite entry")
        return adv

    def audit(self):
        """Confirm the estimator process has not reached past the arrays it was handed."""
        pid = self.proc.pid
        try:
            with open(f"/proc/{pid}/maps", "r", encoding="utf-8", errors="replace") as fh:
                maps = fh.read()
        except OSError:
            return
        lowered = maps.lower()
        for needle in FORBIDDEN_MAPPINGS:
            if needle in lowered:
                raise SandboxViolation(f"the estimator process loaded {needle!r}")
        for entry in os.listdir("/proc"):
            if not entry.isdigit() or int(entry) == pid:
                continue
            try:
                with open(f"/proc/{entry}/stat", "r", encoding="utf-8") as fh:
                    fields = fh.read().rsplit(")", 1)[-1].split()
                if int(fields[1]) == pid:
                    raise SandboxViolation("the estimator process spawned a child process")
            except (OSError, IndexError, ValueError):
                continue

    def close(self):
        try:
            self.proc.stdin.write(b"Q")
            self.proc.stdin.flush()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--submission", default="/app/output", help="directory holding advantage.py")
    p.add_argument("--ckpt", default=None, help="path for the trained policy checkpoint")
    p.add_argument("--ckpt-fd", type=int, default=None, help="write the checkpoint to this fd")
    p.add_argument("--log", default=None, help="optional path for a JSON training log")
    return p.parse_args()


def main():
    args = parse_args()
    import gymnasium as gym

    torch.set_num_threads(H.HARNESS_THREADS)
    torch.manual_seed(args.seed)
    shuffle_rng = np.random.default_rng(args.seed + 977)

    envs = gym.vector.SyncVectorEnv(
        H.make_env_fns(H.ENV_ID, H.NUM_ENVS),
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))

    policy = H.Policy(obs_dim, act_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=H.LEARNING_RATE, eps=H.ADAM_EPS)

    obs_rms = H.RunningMeanStd(shape=(obs_dim,))
    ret_rms = H.RunningMeanStd(shape=())
    ret_acc = np.zeros(H.NUM_ENVS, dtype=np.float64)

    T, N = H.NUM_STEPS, H.NUM_ENVS
    b_obs = np.zeros((T, N, obs_dim), dtype=np.float32)
    b_next_obs = np.zeros((T, N, obs_dim), dtype=np.float32)
    b_actions = np.zeros((T, N, act_dim), dtype=np.float32)
    b_logprobs = np.zeros((T, N), dtype=np.float32)
    b_rewards = np.zeros((T, N), dtype=np.float32)
    b_term = np.zeros((T, N), dtype=bool)
    b_trunc = np.zeros((T, N), dtype=bool)

    minibatch_size = H.BATCH_SIZE // H.NUM_MINIBATCHES
    episode_log = []
    raw_return = np.zeros(H.NUM_ENVS, dtype=np.float64)

    started = time.monotonic()
    status = 0
    try:
        client = EstimatorClient(args.submission, obs_dim, act_dim)
    except EstimatorFailure as exc:
        print(f"ESTIMATOR FAILURE: {exc}", file=sys.stderr)
        envs.close()
        return 2
    try:
        raw_obs, _ = envs.reset(seed=[args.seed * 1000 + i for i in range(H.NUM_ENVS)])
        for iteration in range(H.NUM_ITERATIONS):
            if H.ANNEAL_LR:
                frac = 1.0 - iteration / H.NUM_ITERATIONS
                optimizer.param_groups[0]["lr"] = frac * H.LEARNING_RATE
            for t in range(T):
                obs_rms.update(raw_obs)
                obs_n = H.normalize_obs(raw_obs, obs_rms)
                with torch.no_grad():
                    action, logprob = policy.act(torch.from_numpy(obs_n))
                action_np = action.numpy()
                raw_next, reward, term, trunc, infos = envs.step(action_np)
                done = np.logical_or(term, trunc)

                true_next = np.array(raw_next, dtype=np.float64, copy=True)
                if done.any() and "final_obs" in infos:
                    final = infos["final_obs"]
                    for i in np.nonzero(done)[0]:
                        if final[i] is not None:
                            true_next[i] = final[i]

                raw_return += reward
                for i in np.nonzero(done)[0]:
                    episode_log.append(
                        [(iteration * T + t + 1) * N, float(raw_return[i])]
                    )
                    raw_return[i] = 0.0

                ret_acc = ret_acc * H.RETURN_SCALE_DECAY + reward
                ret_rms.update(ret_acc)
                scaled = reward / np.sqrt(ret_rms.var + 1e-8)
                ret_acc[done] = 0.0

                b_obs[t] = obs_n
                b_next_obs[t] = H.normalize_obs(true_next, obs_rms)
                b_actions[t] = action_np
                b_logprobs[t] = logprob.numpy()
                b_rewards[t] = np.clip(scaled, -H.REWARD_CLIP, H.REWARD_CLIP)
                b_term[t] = term
                b_trunc[t] = trunc
                raw_obs = raw_next

            advantages = client.estimate(
                {
                    "obs": b_obs,
                    "next_obs": b_next_obs,
                    "actions": b_actions,
                    "logprobs": b_logprobs,
                    "rewards": b_rewards,
                    "terminations": b_term,
                    "truncations": b_trunc,
                    "iteration": iteration,
                }
            )

            f_obs = torch.from_numpy(b_obs.reshape(H.BATCH_SIZE, obs_dim))
            f_actions = torch.from_numpy(b_actions.reshape(H.BATCH_SIZE, act_dim))
            f_logprobs = torch.from_numpy(b_logprobs.reshape(H.BATCH_SIZE))
            f_adv = torch.from_numpy(advantages.reshape(H.BATCH_SIZE))

            indices = np.arange(H.BATCH_SIZE)
            for _ in range(H.UPDATE_EPOCHS):
                shuffle_rng.shuffle(indices)
                for start in range(0, H.BATCH_SIZE, minibatch_size):
                    mb = torch.from_numpy(indices[start : start + minibatch_size].copy())
                    new_logprob = policy.log_prob(f_obs[mb], f_actions[mb])
                    ratio = torch.exp(new_logprob - f_logprobs[mb])
                    mb_adv = f_adv[mb]
                    unclipped = -mb_adv * ratio
                    clipped = -mb_adv * torch.clamp(ratio, 1.0 - H.CLIP_COEF, 1.0 + H.CLIP_COEF)
                    loss = torch.max(unclipped, clipped).mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), H.MAX_GRAD_NORM)
                    optimizer.step()

            if iteration % 16 == 0 or iteration == H.NUM_ITERATIONS - 1:
                client.audit()
                if time.monotonic() - started > H.RUN_WALL_CLOCK_CAP:
                    raise TimeoutError(
                        f"run exceeded its {H.RUN_WALL_CLOCK_CAP:.0f}s wall-clock cap "
                        f"at iteration {iteration}"
                    )
    except EstimatorFailure as exc:
        print(f"ESTIMATOR FAILURE: {exc}", file=sys.stderr)
        status = 2
    except SandboxViolation as exc:
        print(f"SANDBOX VIOLATION: {exc}", file=sys.stderr)
        status = 4
    except TimeoutError as exc:
        print(f"TIMEOUT: {exc}", file=sys.stderr)
        status = 3
    finally:
        client.close()
        envs.close()

    if status == 0:
        payload = {
            "state_dict": {k: v.detach().clone() for k, v in policy.state_dict().items()},
            "obs_mean": torch.tensor(obs_rms.mean, dtype=torch.float64),
            "obs_var": torch.tensor(obs_rms.var, dtype=torch.float64),
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "seed": args.seed,
        }
        if args.ckpt_fd is not None:
            with os.fdopen(os.dup(args.ckpt_fd), "wb") as fh:
                torch.save(payload, fh)
        if args.ckpt:
            os.makedirs(os.path.dirname(os.path.abspath(args.ckpt)), exist_ok=True)
            torch.save(payload, args.ckpt)
        if args.log:
            os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
            tail = episode_log[-20:]
            with open(args.log, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "seed": args.seed,
                        "episodes": episode_log,
                        "final_20_mean": float(np.mean([r for _, r in tail])) if tail else None,
                        "wall_clock_s": time.monotonic() - started,
                    },
                    fh,
                )
        tail = episode_log[-20:]
        if tail:
            print(
                f"seed {args.seed}: {len(episode_log)} training episodes, "
                f"last-20 raw return mean {np.mean([r for _, r in tail]):.1f}, "
                f"{time.monotonic() - started:.0f}s"
            )
    return status


if __name__ == "__main__":
    sys.exit(main())
