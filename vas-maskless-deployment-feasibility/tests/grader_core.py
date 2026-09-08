import json
import os
import sys
from pathlib import Path

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(1, str(TESTS_DIR / "agent_mirror"))

import mdenv as E
from reference_policy import ReferencePolicy

GRADE_SPLIT = os.environ.get("GRADE_SPLIT", "final")
HIDDEN_DIR = TESTS_DIR / "hidden_data" / GRADE_SPLIT
N_EVAL_EPISODES = 256
N_ACTIONS = E.N_ACTIONS
OBS_DIM = 514
OBS_BYTES = OBS_DIM * 4

INIT_BUDGET_S = 600.0
EPISODE_BUDGET_S = 2.0
TOTAL_BUDGET_S = 1500.0

M0 = 3.205078125
BASELINE_SCORE = 28.6
HIGHER_IS_BETTER = True

METRIC_NAME = (
    "mean episode return of the frozen MiniDungeon reference policy over the 256 sealed "
    "evaluation maps when every action-selection step is masked by the submission's "
    "predicted validity mask"
)


class MaskError(Exception):
    """The deliverable failed to produce a usable mask for the current step."""


def eval_seeds():
    return list(json.loads((HIDDEN_DIR / "eval_seeds.json").read_text())["seeds"])


def load_policy():
    """Load the frozen weights from the verifier's own root-owned copy under /tests."""
    return ReferencePolicy(TESTS_DIR / "agent_mirror" / "policy_weights.npz")


def run_episode(policy, seed, source):
    """Play one sealed episode with `source` supplying the mask at every step.

    `source` must expose begin_episode(), reset(obs) and predict(obs); predict returns
    a (24,) bool array or raises MaskError. A MaskError ends the episode and the return
    accumulated so far is kept.

    Returns (episode_return, status).
    """
    env = E.MiniDungeon(seed)
    act_rng = np.random.default_rng(np.random.SeedSequence([0xAC7, int(seed)]))
    obs = env.observation()
    total = 0.0
    source.begin_episode()
    try:
        source.reset(obs)
    except MaskError as exc:
        return total, "reset:%s" % exc
    while not env.done:
        try:
            mask = source.predict(obs)
        except MaskError as exc:
            return total, "predict:%s" % exc
        logits = policy.logits(obs)
        draw = float(act_rng.random())
        available = np.flatnonzero(mask)
        if available.size:
            z = logits[available]
            z = z - z.max()
            p = np.exp(z)
            p /= p.sum()
            k = int(np.searchsorted(np.cumsum(p), draw, side="right"))
            action = int(available[min(k, available.size - 1)])
        else:
            action = E.A_NOOP
        obs, reward, _ = env.step(action)
        total += reward
    return total, "ok"


def graded_reward(metric, valid):
    """rational_squash: 0 at the strongest trivial baseline, 0.5 at the baseline score."""
    if not valid:
        return 0.0
    sigma = 1.0 if HIGHER_IS_BETTER else -1.0
    x = sigma * (float(metric) - M0)
    x_ref = sigma * (BASELINE_SCORE - M0)
    assert x_ref > 0, "malformed task: the baseline score does not beat the trivial baseline"
    u = max(0.0, x) / x_ref
    return u / (1.0 + u)
