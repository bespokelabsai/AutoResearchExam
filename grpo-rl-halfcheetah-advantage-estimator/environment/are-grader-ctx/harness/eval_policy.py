#!/usr/bin/env python3
import argparse
import json
import os
import sys

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppo_harness as H

EPISODE_STEPS = 1000
SAMPLING_SEED = 20260101


def evaluate(ckpt_path, seeds):
    import gymnasium as gym

    torch.set_num_threads(1)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    obs_dim, act_dim = int(payload["obs_dim"]), int(payload["act_dim"])
    policy = H.Policy(obs_dim, act_dim)
    policy.load_state_dict(payload["state_dict"])
    policy.eval()
    for param in policy.parameters():
        if not torch.isfinite(param).all():
            raise ValueError("the checkpoint contains a non-finite parameter")

    stats = H.RunningMeanStd(shape=(obs_dim,))
    stats.mean = np.asarray(payload["obs_mean"], dtype=np.float64)
    stats.var = np.asarray(payload["obs_var"], dtype=np.float64)
    if not (np.isfinite(stats.mean).all() and np.isfinite(stats.var).all() and (stats.var >= 0).all()):
        raise ValueError("the checkpoint contains non-finite observation statistics")

    envs = gym.vector.SyncVectorEnv(
        H.make_env_fns(H.ENV_ID, len(seeds)),
        autoreset_mode=gym.vector.AutoresetMode.DISABLED,
    )
    torch.manual_seed(SAMPLING_SEED)
    raw_obs, _ = envs.reset(seed=[int(s) for s in seeds])
    totals = np.zeros(len(seeds), dtype=np.float64)
    with torch.no_grad():
        for _ in range(EPISODE_STEPS):
            obs_n = H.normalize_obs(raw_obs, stats)
            action, _ = policy.act(torch.from_numpy(obs_n))
            raw_obs, reward, _, _, _ = envs.step(action.numpy())
            totals += reward
    envs.close()
    return [float(x) for x in totals]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--seeds", required=True, help="comma-separated integers, one per episode")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    returns = evaluate(args.ckpt, seeds)
    result = {"returns": returns, "mean_return": float(np.mean(returns))}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
