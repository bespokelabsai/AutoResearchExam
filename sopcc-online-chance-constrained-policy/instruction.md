# Online stochastic orienteering under a budget-violation cap

A robot moves over a complete graph on 40 vertices, starting at vertex 0 and finishing at vertex
1, collecting the reward of every distinct vertex it visits along the way. Vertex rewards,
coordinates, and the distance matrix are known exactly before the run starts. What a traversal
actually costs is random and is revealed only through the budget left afterward:

$$\text{cost}(u, v) = \text{dist}[u, v] \cdot E, \qquad E \sim \text{Exponential}(\text{mean}=1)$$

drawn independently for each traversal, where $\text{dist}$ is the Euclidean distance between the
two vertices.

Write an online policy that decides, one move at a time, where to go next.

Episode rules:

- The episode ends the moment the robot travels to vertex 1.
- If the remaining budget is still non-negative on arrival at vertex 1, the episode delivers the
  sum of the rewards of the distinct vertices visited.
- If the budget goes negative on any traversal, including the last one, the episode delivers
  nothing (0.0).
- Vertices 0 and 1 carry reward 0.0, so nothing is collected for free.

## Instance family

Every instance, including the graded ones, is drawn like this:

- 40 vertices; coordinates i.i.d. $U[0,1]^2$; vertex rewards i.i.d. $U[0,1]$, then `rewards[0]`
  and `rewards[1]` are overwritten with `0.0`.
- `dist` is the full 40x40 Euclidean distance matrix; the graph is complete.
- `budget = 2.0`, `start = 0`, `goal = 1`.
- Each episode of an instance pre-draws its own vector of 48 i.i.d. $\text{Exponential}(1)$
  values before the episode starts. The $k$-th traversal of that episode ($k$ counted from 0)
  costs $\text{dist}[u, v] \cdot E_k$. The same instances and the same pre-drawn vectors are
  reused for every submission, so two submissions are compared on identical randomness.
- An episode is capped at 48 traversals; reaching that cap without arriving at the goal is a
  failure.

`/app/harness/sopcc.py` is the generator and the episode dynamics, and it is exactly the code
that runs at grade time. `/app/harness/` is read-only.

## What to build

`/app/solution/policy.py`, exposing a class named `Policy` with exactly these three methods:

```python
class Policy:
    def __init__(self, seed: int) -> None: ...
    def prepare(self, instance: dict) -> None: ...
    def act(self, obs: dict) -> int: ...
```

`prepare` is called once per instance, before any of its episodes, with

| key | type |
|---|---|
| `coords` | `float64` ndarray, shape `(40, 2)` |
| `rewards` | `float64` ndarray, shape `(40,)`, with `rewards[0] == rewards[1] == 0.0` |
| `dist` | `float64` ndarray, shape `(40, 40)` |
| `start`, `goal` | `int`, `0` and `1` |
| `budget` | `float`, `2.0` |
| `p_fail` | `float`, `0.1` |
| `cost_model` | `str`, `"exponential_mean_equals_euclidean"` |

`act` is called once per decision with

| key | type |
|---|---|
| `current` | `int`, the vertex the robot is on |
| `visited` | `bool` ndarray, shape `(40,)`, true for every vertex already visited, including the start |
| `remaining_budget` | `float`, what is left of the 2.0 |
| `collected_reward` | `float`, the reward banked so far this episode |
| `step` | `int`, the number of traversals already made this episode; `0` on the first decision of every episode |

and must return the integer index of the next vertex to travel to. Returning the goal index ends
the episode; any other return value must be an unvisited vertex other than the current one.

Each of the following ends that episode immediately as a failure (delivered reward 0.0, counted
toward the failure rate):

- returning a visited vertex, the current vertex, or an index outside `0..39`
- returning a non-integer (booleans included)
- raising any exception

Submission format:

- deliverable: `/app/solution/policy.py`, plus any further modules and data files anywhere under
  `/app/solution/`, up to 50 MB total
- `/app/solution` is on `sys.path`, so helper modules are imported as top-level modules
- the tree must contain only regular files and directories: no symbolic links, no hard links
- grading runs in a separate, freshly built environment that receives only `/app/solution`;
  nothing else about how you produced it (installed packages, other files, environment
  variables, running processes) carries over, so the solution must be fully self-contained in
  `/app/solution`

## How it is scored

The policy is graded on a sealed panel of 48 instances drawn from the generator above, using seed
material not present in this image, with 32 episodes each, for 1536 graded episodes total.

- $R$ = mean delivered reward over all 1536 episodes, counting a failed episode as 0.0.
- $F$ = fraction of the 1536 episodes that failed.
- $p = 1.0$ if $F \le 0.10$, otherwise $p = \max(0, 1 - 10 (F - 0.10))$, so $p$ falls linearly to
  zero at $F = 0.20$.
- Score: $\text{score} = R \cdot p$.

The objective is to make the score as large as possible:

- Higher is strictly better; there is no cutoff beyond which further improvement stops mattering.
- A policy no better than the trivial reference in `/app/harness/baseline_policy.py` earns
  nothing at all; anything above it earns strictly more, with no upper limit on the scale.
- A deliverable that is missing, unimportable, or unable to answer a decision earns nothing.

## Grade-time resources

Each instance is graded in a fresh process that imports your module, so no state survives from
one instance to the next.

- **At most 75 seconds of CPU time per instance**, covering the import of your module, `__init__`,
  `prepare`, and all 32 episodes of that instance. This cap is charged to every process and
  thread you start; once it is used up, the remaining episodes of that instance are failures.
- **At most 150 seconds of wall-clock time per instance**, with the same consequence.
- 6 GiB of address space, 64 MiB of total file writes, and 256 open files per process.
- `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS` are all `1`, and instances are
  graded 8 at a time on 8 vCPUs.
- No network access, at grade time or while you work.
- Anything your code writes to stdout is redirected to stderr, so do not print to fd 1 expecting
  to see it.
- Packages installed: `numpy` 2.5.2, `scipy` 1.18.0, CPU `torch` 2.13.0. Nothing else may be
  installed.

## Working locally

`python3 /app/harness/dev_eval.py --solution /app/solution` scores your policy the same way the
grader does: same protocol, same caps, on the public dev panel of 4 independent draws of 50
instances each, generated from the public seeds in `sopcc.py`.

It reports:

- the score per draw and averaged over draws
- the failure rate
- the largest per-instance CPU time
- why episodes failed

`--draws`, `--instances`, and `--episodes` trade runtime for precision. Draws differ from each
other, and the graded panel is another draw from the same generator, so a result from one small
draw is a weak estimate of the graded score.
