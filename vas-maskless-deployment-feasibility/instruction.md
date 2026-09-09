# Supply the validity mask a mask-trained policy acts under

MiniDungeon is a procedurally generated dungeon crawler with 24 discrete actions. Before each
choice, the policy needs a Boolean mask that marks which actions the environment will accept.
During training, the policy received the true mask and ignored invalid actions by setting their
logits to `-inf`. That mask is not available at deployment. Build a predictor to replace it.

## Submission requirements

Ship `/app/output/predictor.py` defining `class MaskPredictor` with exactly these three
entry points:

```python
class MaskPredictor:
    def __init__(self, artifacts_dir: str): ...
    def reset(self, first_obs: np.ndarray) -> None: ...
    def predict(self, obs: np.ndarray) -> np.ndarray: ...
```

- `predict` must return a `numpy` array of `dtype=bool` and shape `(24,)`. `True` leaves that
  action available to the policy; `False` forces its logit to `-inf`.
- `obs` is a C-contiguous `float32` array of shape `(514,)`, the same vector the frozen policy
  consumes, and the only input you get.
- `artifacts_dir` is the directory `predictor.py` was loaded from, so anything you save next to
  it is available to `__init__`.
- Only `/app/output/` is writable, and only what is under it carries over to the evaluation run:
  `predictor.py` plus every weight, table, or config it loads.
- Your deliverable directory is placed first on `sys.path`, so it may ship helper modules, but it
  must not depend on anything you leave elsewhere.
- There is no network.

## Scoring

The reference policy is deployed on a sealed set of held-out MiniDungeon maps you have never
seen, one episode per map, 500 steps each. Each episode calls `reset` with the episode's first
observation, then `predict` once per step; the first `predict` call of an episode receives that
same first observation. At every step your mask is applied to the policy's logits, an action is
sampled from the resulting distribution with a pinned RNG stream, and the environment is stepped.
Rewards are per-step achievement rewards, tiered 1 / 3 / 5 / 8, with a theoretical maximum of 98
in one episode.

- Metric: mean episode return over the sealed episodes.
- Direction: maximize. Higher is strictly better, with no cutoff at which further improvement
  stops mattering.
- Baseline: a trivial predictor (a constant mask, or one that ignores the observation) earns 0.
  Anything worse also earns 0.
- Determinism: the maps, the environment, and the action-sampling stream are all pinned, and
  `OMP_NUM_THREADS=1`, so a fixed submission gets the same score every time.

## What you are given

Every path listed here is read-only, and all of them are present at grade time with
byte-identical content at the same paths.

- `/app/policy_weights.npz` and `/app/reference_policy.py`: the frozen policy.
  `ReferencePolicy` loads the weights and exposes `logits(obs) -> (24,) float32`,
  `logits_batch(obs) -> (N, 24)`, and `masked_probs(obs, mask) -> (24,) float32`. This is the same
  code and the same weights the evaluation harness uses to produce the logits your mask is
  applied to.
- `/app/mdobs.py`: the observation encoder, and the authoritative description of the 514-d
  layout: 500 dims of a 5x5 tile view centered on the player (per tile, 16 one-hot block channels
  then 4 mob/item flag bits, tile-major, row-major), followed by 14 inventory and status dims.
  Its `build_obs_batch(crop_block, crop_flags, extras_raw)` is the function that produced every
  observation you will ever be given.
- `/app/data/train_traj.npz` and `/app/data/val_traj.npz`: trajectory banks recorded by rolling
  the frozen policy out on visible maps, 964843 and 121752 steps respectively, disjoint from
  each other and from the evaluation maps. Both hold:

  | key | dtype, shape | meaning |
  | --- | --- | --- |
  | `crop_block` | uint8 `(N, 25)` | per-tile block id of the 5x5 view |
  | `crop_flags` | uint8 `(N, 25)` | per-tile flag bits of the 5x5 view |
  | `extras_raw` | uint16 `(N, 12)` | the raw inventory/status counters |
  | `valid` | bool `(N, 24)` | the oracle's validity mask at that state |
  | `mask_used` | bool `(N, 24)` | the mask the policy actually acted under at that state |
  | `action` | uint8 `(N,)` | the action sampled |
  | `reward` | uint8 `(N,)` | the reward that step produced |
  | `episode` | uint16 `(N,)` | episode index; rows are in step order within an episode |
  | `episode_seed` | int64 `(E,)` | the map seed of each episode |

  - Pass the first three arrays for a set of rows through `build_obs_batch` to get the exact
    `(N, 514)` float32 observations.
  - `mask_used` equals `valid` on some episodes and is a corrupted version of it on others.
- The MiniDungeon implementation itself (its map generator, transition function, and validity
  rules) is not provided, and there is no simulator you can step.

## Budgets and failure

- `__init__` must return within **600 seconds** of wall clock. Exceeding it, raising, or
  failing to import scores 0 for the whole submission.
- Per episode, the cumulative wall clock spent inside `reset` plus all `predict` calls for that
  episode must stay under **2.0 seconds**, an average of 4 ms per step. Exceeding it ends that
  episode, keeps the return accumulated so far, and then **terminates the predictor process, so
  every remaining episode contributes 0**.
- The whole evaluation run is capped at **1500 seconds**; episodes not reached contribute 0.
- A `predict` call that raises, or returns anything that is not a bool array of shape `(24,)`,
  ends that episode and keeps the return accumulated so far; the process keeps running and the
  next episode starts normally. If the predictor process dies on its own, every remaining episode
  contributes 0.
- An all-`False` mask is replaced by a forced no-op for that step.

## Environment

- Installed packages: `numpy`, `scipy`, `scikit-learn`, CPU `torch`.
- Hardware: 8 CPUs, no GPU, 14 GB of memory.
- No network access.
