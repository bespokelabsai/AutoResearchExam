# Advantage estimation for a frozen PPO-clip harness

Design the advantage values used by a fixed PPO training program in Gymnasium's
`HalfCheetah-v4` environment. Your goal is to maximize the trained policy's average episode return
across hidden training and evaluation seeds.

The only part you control is the advantage assigned to each action in each rollout. An advantage
says whether an action produced a better or worse outcome than expected. PPO uses these values to
update the policy.

The program collects 128 steps from 8 environments at a time. It runs 4 training epochs per
iteration. Each epoch uses 4 batches of 256 transitions. Training ends after 976 iterations, or
999,424 environment steps.

The program trains only the policy. It does not normalize, clip, or rescale the advantages that
you return. Your array is passed directly to the training loss.

## Deliverable

Write `/app/output/advantage.py`, defining `class AdvantageEstimator` with two methods:

```python
class AdvantageEstimator:
    def __init__(self, obs_dim, act_dim, num_envs, num_steps, total_iterations):
        ...

    def estimate(self, rollout) -> numpy.ndarray:
        ...
```

The grader calls your code as follows:

- `__init__` runs once per training run, before any environment interaction, with all five values
  passed as keyword arguments: `obs_dim=17`, `act_dim=6`, `num_envs=8`, `num_steps=128`,
  `total_iterations=976`.
- `estimate` runs once per iteration and must return a C-contiguous `float32` array of shape
  `(num_steps, num_envs)` that is finite everywhere.
- The instance persists for the whole training run, so it can carry and fit its own state,
  including its own networks and optimizers, using the arrays it is handed.
- Only `/app/output/` carries over into the graded run, and it must be fully self-contained there.
  `advantage.py` can import sibling modules and read data files from that directory, but nothing
  outside `/app/output/` is available when it is graded.

`rollout` is a `dict` of freshly allocated numpy arrays:

| key | shape | dtype | meaning |
|---|---|---|---|
| `obs` | `(T, N, obs_dim)` | `float32` | the observation the policy acted on at step `t` in env `n`, after the harness's running whitening and clipping |
| `next_obs` | `(T, N, obs_dim)` | `float32` | the observation that step produced, whitened the same way. Where an episode ended at `t` this is that episode's final observation, not the first observation of the next one |
| `actions` | `(T, N, act_dim)` | `float32` | the action taken |
| `logprobs` | `(T, N)` | `float32` | its log-probability under the policy that took it |
| `rewards` | `(T, N)` | `float32` | the reward the harness optimises: the environment reward after the harness's running scaling and clipping |
| `terminations` | `(T, N)` | `bool` | the environment reported termination at that step |
| `truncations` | `(T, N)` | `bool` | the environment reported truncation at that step |
| `iteration` | - | `int` | 0-based index of this iteration |

## Scoring

The grader trains a policy with your estimator and measures its average episode return. It uses
the environment's raw reward before any scaling by the training program.

- Twelve training runs are executed, one per hidden training seed.
- Each trained stochastic policy is rolled out for 20 evaluation episodes of 1000 steps, on
  environments built from hidden evaluation seeds. These seeds are not present in this container
  and differ from any seed you use locally.
- The metric is the arithmetic mean of all 240 episode returns.
- Higher is better. Every improvement counts.
- An estimator no better than a trivial one scores zero.

During an iterated research run, the intermediate grader uses separate hidden training and
evaluation seeds. It measures the same outcome with six training runs, so its score is noisier
than the final score.

## Hard constraints

The submission scores zero if any of these conditions occurs:

- `/app/output/advantage.py` is missing or empty, or defines no `AdvantageEstimator`
- `__init__` or `estimate` raises, or `estimate` returns anything other than a finite,
  C-contiguous `float32` array of shape `(128, 8)`
- a graded training run does not finish within 1500 seconds of wall clock (it is SIGKILLed at that
  point)
- the estimator process imports `gymnasium`, `mujoco`, or any other simulator package, constructs
  an environment, or spawns a child process
- any one of the twelve graded training runs fails for any reason above

The estimator runs in its own process and receives nothing but the arrays above. It cannot reach
the policy, the optimizer, the environments, or their random state.

## Environment

- 8 CPU cores, 14336 MB memory, 0 GPUs, no network access.
- Installed packages are `torch` with CPU support, `numpy`, `gymnasium`, and `mujoco`.
- The training program uses one PyTorch thread. The estimator process uses two threads through
  `OMP_NUM_THREADS=2` and `torch.set_num_threads(2)`.
- A full 976 iteration run with a cheap estimator takes about four minutes on an idle machine.

`/app/harness/` is read-only and is restored from a pristine copy before grading. Run it yourself
with any seeds you like:

```
python3 /app/harness/train.py --seed 0 --submission /app/output \
    --ckpt /app/runs/s0/policy.pt --log /app/runs/s0/log.json
python3 /app/harness/eval_policy.py --ckpt /app/runs/s0/policy.pt --seeds 101,102,103,104,105
```

`train.py` exit codes:

- `0`: completed run
- `2`: the estimator broke its contract
- `3`: the run hit the wall-clock cap
- `4`: it reached outside the arrays it was given

`--log` records the raw episodic returns seen during training.
