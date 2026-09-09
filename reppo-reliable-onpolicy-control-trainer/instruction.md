# Train control policies that hit a performance bar on every seed and hold it

Write a reinforcement-learning training procedure at `/app/solution/trainer.py`. The score is
the **reliable-success fraction**. It is the fraction of runs whose policy reaches 90% of a
reference level and stays there for the last ten evaluation checkpoints.

The grader runs your procedure from scratch 72 times. It uses six unseen continuous-control
environments with twelve sealed random seeds each. Each run gets 400,000 environment
interactions and 45 CPU-seconds. Nothing carries over between runs. Push the reliable-success
fraction as high as you can. Higher is always better. A procedure that does no learning scores 0.

## The deliverable

`/app/solution/trainer.py` must define, at module level:

```python
def train(env, seed, budget, report) -> None
```

`env` is a vectorised environment handle owned by the harness:

| member | meaning |
| --- | --- |
| `env.reset()` | `np.ndarray (env.n_envs, env.obs_dim)` float32 |
| `env.step(actions)` | `(obs, reward, done)`; `actions` must be `(env.n_envs, env.act_dim)` and coerce to float32 with every entry in `[-1, 1]` |
| `env.n_envs` | 64 |
| `env.obs_dim` | 8 |
| `env.act_dim` | 2 |
| `env.max_episode_steps` | 100 |

Episodes have a fixed length: `done` is true only on an episode's final step, and the batch
auto-resets on the next `step`. Once the interaction budget is spent, `env.step` raises
`RuntimeError`.

`seed` is the run's integer seed. `budget` carries `budget.total_env_steps` (400000),
`budget.cpu_seconds` (45.0), and `budget.n_checkpoints` (40).

`report(policy_fn)` hands the harness the deterministic policy it should evaluate from then
on. `policy_fn` maps `np.ndarray (B, 8)` float32 to `(B, 2)` with every entry in `[-1, 1]`,
and must have no side effects, since the harness calls it on evaluation states that never
touch your training environment. Call `report` as often as you like; only the most recently
reported policy is evaluated.

Actions from `policy_fn` and from `env.step` are read as float32: any array whose values
convert exactly to float32 and whose size is `B * 2` is accepted, regardless of dtype or
memory order. Values are then clipped into `[-1, 1]` with a float32 tolerance of `1e-4`, so
rounding at the boundary costs nothing; going further outside ends the run.

You may add helper modules under `/app/solution/`. That directory is everything that reaches
the grader, so anything your trainer imports or loads must live inside it, subject to:

- no symbolic or hard links
- at most 2000 files
- at most 64 MB total

## How a run is scored

The harness evaluates at 40 checkpoints, one every 10,000 environment steps. At checkpoint
`k` it runs the most recently reported `policy_fn` on that environment's 32 fixed evaluation
initial states, one full 100-step episode each, and records the mean undiscounted return
`R_k`. This evaluation does not consume your interaction budget.

`R_k` is normalised against two per-environment constants:

$$s_k = \mathrm{clip}\left(\frac{R_k - R_{\text{rand}}}{R_{\text{ref}} - R_{\text{rand}}}, 0, 1\right)$$

`R_rand` is the return of uniform-random actions and `R_ref` is a fixed reference level. The
constants for the graded environments are not disclosed.

A run is **reliably performant** when $s_k \geq 0.9$ at all of the last ten checkpoints, i.e.
over the final 100,000 environment steps of the run. The metric is the fraction of the 72
runs that are reliably performant.

A checkpoint reached before your first `report` call scores $s_k = 0$, and so does every
checkpoint after a run has been ended by the harness. If `train` returns early, the remaining
checkpoints are evaluated with the policy you reported last.

Each run gets a private, empty working directory as its current directory and as `TMPDIR` and
`HOME`. Nothing written there survives the run. `/app/solution` is read-only while your
trainer executes, and no other path is writable, so the 72 runs are independent and cannot
pass state to one another. POSIX shared memory (`/dev/shm`) is also not writable inside a
graded run, so `multiprocessing` primitives that need it are unavailable. Threads and
`subprocess` work. The CPU meter sums every process your run creates. Each run executes under
its own uid, and System V IPC, POSIX message queues, socket binding, ptrace and kernel keyrings
are refused with `EPERM` inside a graded run.

## What ends a run

Each of the following stops the run at that point, so every checkpoint from then on scores
$s_k = 0$:

- spending more than 45 CPU-seconds. The meter covers every process your run creates, including
  helper processes and the work your `policy_fn` does answering evaluation queries, and it
  starts when `train` is entered; interpreter and library import time is not charged.
- taking more than 150 seconds of wall-clock time.
- returning an action array of the wrong size, containing a non-finite value, or with an
  entry outside `[-1, 1]`.
- raising an exception out of `train` or out of `policy_fn`.

A deliverable that is missing, empty, unparseable, or that does not define a module-level
`train` taking four positional arguments scores 0 outright.

## The environments

`/app/envs.py` holds all six families, with their disclosed parameter ranges and dynamics:

- a 2-D double integrator
- an inverted-pendulum balance task
- a pendulum swing-up
- a cart-pole swing-up
- a planar two-link reacher
- a pendulum swing-up with a one- or two-step actuation delay

Every family is presented through the same 8-dimensional observation and 2-dimensional action
interface: families that drive one actuator ignore the second action component, and families
with fewer than eight state features occupy a subspace of the observation.

Each environment also carries two transforms drawn with its own parameter seed: a fixed
orthogonal rotation applied to the observation vector, and a fixed signed permutation applied
to the action vector before the dynamics see it. The graded environments use parameter draws,
transforms, evaluation states, and seeds that you do not have.

## Iterating locally

`/app/public_panel.json` holds six public environments, one draw from each family, with their
own `R_rand` and `R_ref` constants and three public seeds. These are not the graded ones.

```
python3 /app/run_public.py
python3 /app/run_public.py --envs env2 --seeds 0 --workers 4
```

It prints, per run, the mean evaluation return at each of the 40 checkpoints together with
that environment's `R_rand` and `R_ref`, so you can apply the normalisation above yourself.

`/app/panel.py` and `/app/child_main.py` are the same harness the grader runs, including the
meters and the evaluation protocol. `/app/envs.py`, `/app/panel.py`, `/app/child_main.py`,
`/app/run_public.py`, and `/app/public_panel.json` are inputs; only `/app/solution/` is yours
to change.

## The machine

- 8 CPU cores, 14 GB of RAM, no GPU, no network
- packages installed: `numpy`, and a CPU build of `torch`
- one thread per process (`OMP_NUM_THREADS=1`); extra threads will not buy you compute
- the 72 graded runs are executed 8 at a time
