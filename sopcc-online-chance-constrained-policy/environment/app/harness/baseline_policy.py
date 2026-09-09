from __future__ import annotations

import numpy as np

ALPHA = 0.50


class Policy:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def prepare(self, instance: dict) -> None:
        self.dist = instance["dist"]
        self.goal = int(instance["goal"])
        self.rewards = instance["rewards"]

    def act(self, obs: dict) -> int:
        current = int(obs["current"])
        visited = obs["visited"]
        budget = ALPHA * float(obs["remaining_budget"])
        out = self.dist[current]
        back = self.dist[:, self.goal]
        feasible = (~visited) & (out + back <= budget)
        feasible[self.goal] = False
        if not feasible.any():
            return self.goal
        candidates = np.flatnonzero(feasible)
        return int(candidates[np.argmin(out[candidates])])
