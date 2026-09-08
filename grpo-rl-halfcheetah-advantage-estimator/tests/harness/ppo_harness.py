import numpy as np
import torch
import torch.nn as nn

ENV_ID = "HalfCheetah-v4"
NUM_ENVS = 8
NUM_STEPS = 128
TOTAL_TIMESTEPS = 1_000_000
UPDATE_EPOCHS = 4
NUM_MINIBATCHES = 4
LEARNING_RATE = 2.5e-4
ADAM_EPS = 1e-5
CLIP_COEF = 0.2
MAX_GRAD_NORM = 0.5
ANNEAL_LR = True

BATCH_SIZE = NUM_ENVS * NUM_STEPS
NUM_ITERATIONS = TOTAL_TIMESTEPS // BATCH_SIZE

OBS_CLIP = 10.0
REWARD_CLIP = 10.0
RETURN_SCALE_DECAY = 0.99

RUN_WALL_CLOCK_CAP = 1500.0

HARNESS_THREADS = 1


class RunningMeanStd:
    """Welford-style running mean/variance, matching gymnasium's normalisation wrappers."""

    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot
        self.mean = new_mean
        self.var = m2 / tot
        self.count = tot


def normalize_obs(raw, obs_rms):
    """Whiten with the running observation statistics, then clip."""
    out = (np.asarray(raw, dtype=np.float64) - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8)
    return np.clip(out, -OBS_CLIP, OBS_CLIP).astype(np.float32)


def layer_init(layer, std=np.sqrt(2.0), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Policy(nn.Module):
    """Gaussian policy: tanh MLP 64-64 mean head plus a state-independent log-std."""

    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )
        self.log_std = nn.Parameter(torch.zeros(1, act_dim))

    def distribution(self, obs):
        mean = self.mean(obs)
        std = torch.exp(self.log_std.expand_as(mean))
        return torch.distributions.Normal(mean, std)

    def act(self, obs):
        dist = self.distribution(obs)
        action = dist.sample()
        return action, dist.log_prob(action).sum(-1)

    def log_prob(self, obs, action):
        dist = self.distribution(obs)
        return dist.log_prob(action).sum(-1)


def make_env_fns(env_id, num_envs):
    import gymnasium as gym

    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.ClipAction(env)
        return env

    return [thunk for _ in range(num_envs)]
