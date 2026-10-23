from __future__ import annotations

import numpy as np


N_VERTICES = 40
START = 0
GOAL = 1
BUDGET = 2.0
P_FAIL = 0.1
COST_MODEL = "exponential_mean_equals_euclidean"


EPISODES_PER_INSTANCE = 32
MAX_TRAVERSALS = 48


FAILURE_ALLOWANCE = 0.10
PENALTY_SLOPE = 10.0


DEV_ENTROPY = 20240917
DEV_DRAWS = 4
DEV_INSTANCES_PER_DRAW = 50


def instance_seed(entropy: int, index: int) -> np.random.SeedSequence:
    """Seed sequence for instance `index` of the family rooted at `entropy`."""
    return np.random.SeedSequence(entropy=int(entropy), spawn_key=(int(index), 0))


def noise_seed(entropy: int, index: int, variant: int = 0) -> np.random.SeedSequence:
    """Seed sequence for the pre-drawn cost noise of instance `index`.

    `variant` draws an independent noise realisation for the same instance, which is how the
    build measures the metric's own sampling noise.
    """
    key = (int(index), 1) if int(variant) == 0 else (int(index), 1, int(variant))
    return np.random.SeedSequence(entropy=int(entropy), spawn_key=key)


def euclidean_matrix(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=-1))
    np.fill_diagonal(dist, 0.0)
    return dist


def make_instance(seed_sequence: np.random.SeedSequence) -> dict:
    """One SOPCC instance: complete graph, U[0,1]^2 coordinates, U[0,1] vertex rewards.

    The start and goal vertex rewards are forced to 0.0, so no reward is collected for
    free by starting or finishing.
    """
    rng = np.random.Generator(np.random.PCG64(seed_sequence))
    coords = rng.random((N_VERTICES, 2))
    rewards = rng.random(N_VERTICES)
    rewards[START] = 0.0
    rewards[GOAL] = 0.0
    return instance_from_arrays(coords, rewards)


def instance_from_arrays(coords: np.ndarray, rewards: np.ndarray) -> dict:
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    rewards = np.ascontiguousarray(rewards, dtype=np.float64)
    return {
        "coords": coords,
        "rewards": rewards,
        "dist": euclidean_matrix(coords),
        "start": START,
        "goal": GOAL,
        "budget": BUDGET,
        "p_fail": P_FAIL,
        "cost_model": COST_MODEL,
    }


def make_noise(
    seed_sequence: np.random.SeedSequence,
    episodes: int = EPISODES_PER_INSTANCE,
    traversals: int = MAX_TRAVERSALS,
) -> np.ndarray:
    """Pre-drawn unit-Exponential cost multipliers, one row of `traversals` per episode.

    The realised cost of the k-th traversal of an episode is `dist[u, v] * noise[k]`, so
    every traversal costs an Exponential variate with mean equal to the Euclidean edge
    length, and two submissions see the identical random stream.
    """
    rng = np.random.Generator(np.random.PCG64(seed_sequence))
    return rng.exponential(1.0, size=(int(episodes), int(traversals)))


def make_dev_panel(draws: int = DEV_DRAWS, instances_per_draw: int = DEV_INSTANCES_PER_DRAW):
    """The public dev panel: `draws` independent draws of `instances_per_draw` instances."""
    panel = []
    for draw in range(int(draws)):
        for k in range(int(instances_per_draw)):
            index = draw * int(instances_per_draw) + k
            panel.append(
                {
                    "draw": draw,
                    "index": index,
                    "instance": make_instance(instance_seed(DEV_ENTROPY, index)),
                    "noise": make_noise(noise_seed(DEV_ENTROPY, index)),
                }
            )
    return panel


class InvalidAction(Exception):
    """The policy returned something that is not a legal next vertex."""


def check_action(action, current: int, visited: np.ndarray) -> int:
    """Validate one action, returning it as an int or raising InvalidAction."""
    if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
        raise InvalidAction(f"action is {type(action).__name__}, not an int")
    a = int(action)
    if not 0 <= a < N_VERTICES:
        raise InvalidAction(f"action {a} out of range")
    if a == current:
        raise InvalidAction("action is the current vertex")
    if a != GOAL and visited[a]:
        raise InvalidAction(f"action {a} is an already visited vertex")
    return a


def run_episode(instance: dict, noise_row: np.ndarray, choose) -> dict:
    """Drive one episode. `choose(obs)` returns the next vertex, or raises.

    Returns a dict with `delivered` (0.0 unless the goal was reached inside budget),
    `failed`, `reason`, `visits` (count of distinct non-start vertices collected) and
    `traversals`.
    """
    dist = instance["dist"]
    rewards = instance["rewards"]
    visited = np.zeros(N_VERTICES, dtype=bool)
    visited[START] = True
    current = START
    remaining = float(instance["budget"])
    collected = 0.0
    step = 0

    while True:
        if step >= MAX_TRAVERSALS:
            return _failed("traversal_cap", collected, step)
        obs = {
            "current": int(current),
            "visited": visited.copy(),
            "remaining_budget": float(remaining),
            "collected_reward": float(collected),
            "step": int(step),
        }
        try:
            action = check_action(choose(obs), current, visited)
        except InvalidAction as exc:
            return _failed(f"invalid_action: {exc}", collected, step)
        cost = float(dist[current, action]) * float(noise_row[step])
        remaining -= cost
        step += 1
        if remaining < 0.0:
            return _failed("budget_exhausted", collected, step)
        if action == GOAL:
            return {
                "delivered": float(collected),
                "failed": False,
                "reason": "goal",
                "visits": int(visited.sum()) - 1,
                "traversals": step,
                "remaining_budget": float(remaining),
            }
        visited[action] = True
        collected += float(rewards[action])
        current = action


def _failed(reason: str, collected: float, step: int) -> dict:
    return {
        "delivered": 0.0,
        "failed": True,
        "reason": reason,
        "visits": 0,
        "traversals": int(step),
        "collected_when_failed": float(collected),
    }


def penalty(failure_rate: float) -> float:
    """1.0 while the failure rate respects the allowance, ramping linearly to 0.0."""
    f = float(failure_rate)
    if f <= FAILURE_ALLOWANCE:
        return 1.0
    return max(0.0, 1.0 - PENALTY_SLOPE * (f - FAILURE_ALLOWANCE))


def panel_metric(delivered, failed) -> dict:
    """Penalised mean delivered reward over a panel of episodes."""
    delivered = np.asarray(delivered, dtype=np.float64)
    failed = np.asarray(failed, dtype=bool)
    if delivered.size == 0:
        return {"metric": 0.0, "R": 0.0, "F": 1.0, "penalty": 0.0, "episodes": 0}
    R = float(delivered.mean())
    F = float(failed.mean())
    p = penalty(F)
    return {
        "metric": R * p,
        "R": R,
        "F": F,
        "penalty": p,
        "episodes": int(delivered.size),
        "delivered_std": float(delivered.std(ddof=1)) if delivered.size > 1 else 0.0,
    }
