from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

OBS_DIM = 8
ACT_DIM = 2
MAX_EPISODE_STEPS = 100

FAMILIES = (
    "double_integrator",
    "pendulum_balance",
    "pendulum_swingup",
    "cartpole_swingup",
    "reacher2",
    "delayed_pendulum",
)




PARAM_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "double_integrator": {
        "mass": (0.6, 1.6),
        "drag": (0.0, 0.4),
        "ctrl_cost": (0.02, 0.12),
        "vel_cost": (0.05, 0.30),
        "force": (1.0, 2.0),
    },
    "pendulum_balance": {
        "mass": (0.7, 1.4),
        "length": (0.7, 1.3),
        "gravity": (8.0, 12.0),
        "damping": (0.02, 0.20),
        "ctrl_cost": (0.005, 0.05),
        "torque_frac": (0.50, 0.90),
    },
    "pendulum_swingup": {
        "mass": (0.7, 1.4),
        "length": (0.7, 1.3),
        "gravity": (8.0, 12.0),
        "damping": (0.02, 0.20),
        "ctrl_cost": (0.005, 0.05),
        "torque_frac": (0.26, 0.40),
    },
    "cartpole_swingup": {
        "cart_mass": (0.8, 1.4),
        "pole_mass": (0.08, 0.20),
        "length": (0.5, 0.9),
        "gravity": (9.0, 11.0),
        "ctrl_cost": (0.005, 0.05),
        "force_frac": (0.75, 1.25),
    },
    "reacher2": {
        "l1": (0.8, 1.2),
        "l2": (0.7, 1.1),
        "damping": (0.5, 1.5),
        "ctrl_cost": (0.02, 0.10),
        "torque": (1.5, 3.0),
    },
    "delayed_pendulum": {
        "mass": (0.7, 1.4),
        "length": (0.7, 1.3),
        "gravity": (8.0, 12.0),
        "damping": (0.02, 0.20),
        "ctrl_cost": (0.005, 0.05),
        "torque_frac": (0.26, 0.40),
        "delay": (1.0, 2.0),
    },
}

DT = 0.05


@dataclass(frozen=True)
class EnvSpec:
    """A single environment: a family, one parameter draw, and its two transforms."""

    family: str
    params: dict[str, float]
    q_rot: np.ndarray
    a_perm: np.ndarray

    def to_json(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "params": {k: float(v) for k, v in self.params.items()},
            "q_rot": [[float(x) for x in row] for row in self.q_rot],
            "a_perm": [[float(x) for x in row] for row in self.a_perm],
        }

    @staticmethod
    def from_json(d: dict[str, Any]) -> "EnvSpec":
        return EnvSpec(
            family=d["family"],
            params={k: float(v) for k, v in d["params"].items()},
            q_rot=np.asarray(d["q_rot"], dtype=np.float64),
            a_perm=np.asarray(d["a_perm"], dtype=np.float64),
        )


def _random_orthogonal(rng: np.random.Generator, n: int) -> np.ndarray:
    a = rng.normal(size=(n, n))
    q, r = np.linalg.qr(a)
    return (q * np.sign(np.diag(r))).astype(np.float64)


def _random_signed_permutation(rng: np.random.Generator, n: int) -> np.ndarray:
    perm = rng.permutation(n)
    signs = rng.choice([-1.0, 1.0], size=n)
    p = np.zeros((n, n), dtype=np.float64)
    for i, j in enumerate(perm):
        p[i, j] = signs[i]
    return p


def draw_spec(family: str, seed: int) -> EnvSpec:
    """Draw one environment of ``family`` from the disclosed ranges under ``seed``."""
    rng = np.random.default_rng(seed)
    params: dict[str, float] = {}
    for name, (lo, hi) in PARAM_RANGES[family].items():
        params[name] = float(rng.uniform(lo, hi))
    if family == "delayed_pendulum":
        params["delay"] = float(int(round(params["delay"])))
    return EnvSpec(
        family=family,
        params=params,
        q_rot=_random_orthogonal(rng, OBS_DIM),
        a_perm=_random_signed_permutation(rng, ACT_DIM),
    )


class VecEnv:
    """A batch of ``n_envs`` independent copies of one :class:`EnvSpec`.

    Episodes are fixed length (:data:`MAX_EPISODE_STEPS`); ``done`` is True only on the
    final step of an episode and the batch auto-resets on the following ``step``.
    """

    obs_dim = OBS_DIM
    act_dim = ACT_DIM
    max_episode_steps = MAX_EPISODE_STEPS

    def __init__(self, spec: EnvSpec, n_envs: int, seed: int, init_states: np.ndarray | None = None):
        self.spec = spec
        self.n_envs = int(n_envs)
        self._rng = np.random.default_rng(seed)
        self._fixed_init = None if init_states is None else np.asarray(init_states, dtype=np.float64)
        self._p = spec.params
        self._family = spec.family
        self._q = spec.q_rot
        self._ap = spec.a_perm
        self._delay = int(self._p.get("delay", 0.0)) if self._family == "delayed_pendulum" else 0
        self._state = np.zeros((self.n_envs, self.state_dim), dtype=np.float64)
        self._buf = np.zeros((self.n_envs, max(self._delay, 1)), dtype=np.float64)
        self._t = 0



    @property
    def state_dim(self) -> int:
        return {
            "double_integrator": 4,
            "pendulum_balance": 2,
            "pendulum_swingup": 2,
            "cartpole_swingup": 4,
            "reacher2": 6,
            "delayed_pendulum": 2,
        }[self._family]



    def sample_init(self, n: int, rng: np.random.Generator) -> np.ndarray:
        f = self._family
        if f == "double_integrator":
            s = np.zeros((n, 4))
            s[:, 0:2] = rng.uniform(-1.5, 1.5, size=(n, 2))
            s[:, 2:4] = rng.uniform(-0.5, 0.5, size=(n, 2))
        elif f == "pendulum_balance":
            s = np.zeros((n, 2))
            s[:, 0] = rng.uniform(-0.35, 0.35, size=n)
            s[:, 1] = rng.uniform(-0.8, 0.8, size=n)
        elif f in ("pendulum_swingup", "delayed_pendulum"):
            s = np.zeros((n, 2))
            s[:, 0] = math.pi + rng.uniform(-0.6, 0.6, size=n)
            s[:, 1] = rng.uniform(-0.5, 0.5, size=n)
        elif f == "cartpole_swingup":
            s = np.zeros((n, 4))
            s[:, 0] = rng.uniform(-0.3, 0.3, size=n)
            s[:, 1] = rng.uniform(-0.2, 0.2, size=n)
            s[:, 2] = math.pi + rng.uniform(-0.6, 0.6, size=n)
            s[:, 3] = rng.uniform(-0.3, 0.3, size=n)
        elif f == "reacher2":
            s = np.zeros((n, 6))
            s[:, 0] = rng.uniform(-math.pi, math.pi, size=n)
            s[:, 1] = rng.uniform(-2.0, 2.0, size=n)
            s[:, 2:4] = rng.uniform(-0.5, 0.5, size=(n, 2))
            radius = rng.uniform(0.4, 1.4, size=n)
            angle = rng.uniform(-math.pi, math.pi, size=n)
            s[:, 4] = radius * np.cos(angle)
            s[:, 5] = radius * np.sin(angle)
        else:
            raise KeyError(self._family)
        return s



    def _features(self, s: np.ndarray) -> np.ndarray:
        f = self._family
        n = s.shape[0]
        raw = np.zeros((n, OBS_DIM), dtype=np.float64)
        if f == "double_integrator":
            raw[:, 0:2] = s[:, 0:2] / 1.5
            raw[:, 2:4] = s[:, 2:4] / 2.0
        elif f in ("pendulum_balance", "pendulum_swingup", "delayed_pendulum"):
            raw[:, 0] = np.cos(s[:, 0])
            raw[:, 1] = np.sin(s[:, 0])
            raw[:, 2] = s[:, 1] / 8.0
        elif f == "cartpole_swingup":
            raw[:, 0] = s[:, 0] / 2.0
            raw[:, 1] = s[:, 1] / 3.0
            raw[:, 2] = np.cos(s[:, 2])
            raw[:, 3] = np.sin(s[:, 2])
            raw[:, 4] = s[:, 3] / 8.0
        elif f == "reacher2":
            raw[:, 0] = np.cos(s[:, 0])
            raw[:, 1] = np.sin(s[:, 0])
            raw[:, 2] = np.cos(s[:, 2])
            raw[:, 3] = np.sin(s[:, 2])
            raw[:, 4] = s[:, 1] / 6.0
            raw[:, 5] = s[:, 3] / 6.0
            raw[:, 6] = s[:, 4] / 1.5
            raw[:, 7] = s[:, 5] / 1.5
        return raw @ self._q.T

    def _obs(self) -> np.ndarray:
        return self._features(self._state).astype(np.float32)



    def _dynamics(self, s: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One Euler step.  Returns (next_state, reward)."""
        p = self._p
        f = self._family
        s = s.copy()
        if f == "double_integrator":
            acc = (p["force"] * u[:, 0:2] - p["drag"] * s[:, 2:4]) / p["mass"]
            s[:, 2:4] += DT * acc
            s[:, 0:2] += DT * s[:, 2:4]
            s[:, 0:2] = np.clip(s[:, 0:2], -4.0, 4.0)
            s[:, 2:4] = np.clip(s[:, 2:4], -6.0, 6.0)
            cost = (
                np.sum(s[:, 0:2] ** 2, axis=1)
                + p["vel_cost"] * np.sum(s[:, 2:4] ** 2, axis=1)
                + p["ctrl_cost"] * np.sum(u[:, 0:2] ** 2, axis=1)
            )
            return s, -cost
        if f in ("pendulum_balance", "pendulum_swingup", "delayed_pendulum"):
            torque = p["torque_frac"] * p["mass"] * p["gravity"] * p["length"] * u[:, 0]
            ml2 = p["mass"] * p["length"] ** 2
            acc = (
                p["mass"] * p["gravity"] * p["length"] * np.sin(s[:, 0]) / ml2
                - p["damping"] * s[:, 1] / ml2
                + torque / ml2
            )
            s[:, 1] = np.clip(s[:, 1] + DT * acc, -8.0, 8.0)
            s[:, 0] = s[:, 0] + DT * s[:, 1]
            ang = np.arctan2(np.sin(s[:, 0]), np.cos(s[:, 0]))
            cost = ang**2 + 0.1 * s[:, 1] ** 2 + p["ctrl_cost"] * u[:, 0] ** 2
            return s, -cost
        if f == "cartpole_swingup":
            mc, mp, l, g = p["cart_mass"], p["pole_mass"], p["length"], p["gravity"]
            force = p["force_frac"] * (mc + mp) * g * u[:, 0]
            th, thd = s[:, 2], s[:, 3]
            sin_t, cos_t = np.sin(th), np.cos(th)
            total = mc + mp
            temp = (force + mp * l * thd**2 * sin_t) / total
            thacc = (g * sin_t - cos_t * temp) / (l * (4.0 / 3.0 - mp * cos_t**2 / total))
            xacc = temp - mp * l * thacc * cos_t / total
            s[:, 1] = np.clip(s[:, 1] + DT * xacc, -8.0, 8.0)
            s[:, 0] = np.clip(s[:, 0] + DT * s[:, 1], -3.0, 3.0)
            s[:, 3] = np.clip(thd + DT * thacc, -12.0, 12.0)
            s[:, 2] = th + DT * s[:, 3]
            ang = np.arctan2(np.sin(s[:, 2]), np.cos(s[:, 2]))
            cost = ang**2 + 0.1 * s[:, 0] ** 2 + 0.02 * s[:, 3] ** 2 + p["ctrl_cost"] * u[:, 0] ** 2
            return s, -cost
        if f == "reacher2":
            tau = p["torque"] * u[:, 0:2]
            s[:, 1] = np.clip(s[:, 1] + DT * (tau[:, 0] - p["damping"] * s[:, 1]), -6.0, 6.0)
            s[:, 3] = np.clip(s[:, 3] + DT * (tau[:, 1] - p["damping"] * s[:, 3]), -6.0, 6.0)
            s[:, 0] = s[:, 0] + DT * s[:, 1]
            s[:, 2] = s[:, 2] + DT * s[:, 3]
            ex = p["l1"] * np.cos(s[:, 0]) + p["l2"] * np.cos(s[:, 0] + s[:, 2])
            ey = p["l1"] * np.sin(s[:, 0]) + p["l2"] * np.sin(s[:, 0] + s[:, 2])
            dist2 = (ex - s[:, 4]) ** 2 + (ey - s[:, 5]) ** 2
            cost = dist2 + 0.01 * (s[:, 1] ** 2 + s[:, 3] ** 2) + p["ctrl_cost"] * np.sum(
                u[:, 0:2] ** 2, axis=1
            )
            return s, -cost
        raise KeyError(f)



    def reset(self) -> np.ndarray:
        if self._fixed_init is not None:
            self._state = self._fixed_init[: self.n_envs].copy()
        else:
            self._state = self.sample_init(self.n_envs, self._rng)
        self._buf[:] = 0.0
        self._t = 0
        return self._obs()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = np.asarray(actions, dtype=np.float64).reshape(self.n_envs, ACT_DIM)
        u = a @ self._ap.T
        if self._delay > 0:
            applied = self._buf[:, 0].copy()
            self._buf[:, :-1] = self._buf[:, 1:]
            self._buf[:, -1] = u[:, 0]
            u = np.stack([applied, u[:, 1]], axis=1)
        self._state, reward = self._dynamics(self._state, u)
        self._t += 1
        done = self._t >= MAX_EPISODE_STEPS
        if done:
            obs = self._obs()
            dones = np.ones(self.n_envs, dtype=bool)
            self._state = (
                self._fixed_init[: self.n_envs].copy()
                if self._fixed_init is not None
                else self.sample_init(self.n_envs, self._rng)
            )
            self._buf[:] = 0.0
            self._t = 0
            return obs, reward.astype(np.float32), dones
        return self._obs(), reward.astype(np.float32), np.zeros(self.n_envs, dtype=bool)


def eval_init_states(spec: EnvSpec, n: int, seed: int) -> np.ndarray:
    """The fixed evaluation initial states for one environment."""
    probe = VecEnv(spec, n_envs=1, seed=0)
    return probe.sample_init(n, np.random.default_rng(seed))
