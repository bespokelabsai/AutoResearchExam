import numpy as np


def train(env, seed, budget, report):
    def policy_fn(obs):
        return np.zeros((obs.shape[0], env.act_dim), dtype=np.float32)

    report(policy_fn)
