# Label-efficient risk estimation

A classifier `f` assigns a probability distribution over `C` classes to each of `N = 2000`
unlabelled pool examples. Its risk on that pool is

$$
R = \frac{1}{N} \sum_i -\log \text{target\_probs}[i, y_i]
$$

where `y_i` is the true (unknown) class of example `i`. You may buy at most `M` of these
labels, one at a time, choosing each next example after seeing every label you've already
bought. Your job is to estimate `R`.

You're also given, for the same pool and row order, the predictive distributions of a second,
cheaper classifier over the same `C` classes. You aren't told how good it is, and that varies
from pool to pool.

## Deliverable

Write `/app/output/estimator.py`, defining a class named exactly `Estimator`:

```python
class Estimator:
    def __init__(self, pool: dict, budget: int, rng: numpy.random.Generator) -> None: ...
    def next_index(self) -> int: ...
    def observe(self, index: int, label: int) -> None: ...
    def estimate(self) -> float: ...
```

`pool` has exactly these keys:

| key | type |
| --- | --- |
| `target_probs` | C-contiguous `float64` ndarray, shape `(n_pool, n_classes)`, rows sum to 1 |
| `surrogate_probs` | C-contiguous `float64` ndarray, shape `(n_pool, n_classes)`, rows sum to 1 |
| `n_pool` | `int`, always 2000 |
| `n_classes` | `int`, always 2 |

Requirements:
- Helper modules can live anywhere under `/app/output/`, as long as they're importable from
  `estimator.py`.
- Allowed packages: `numpy`, `scipy`. Nothing else is installed, and there's no network access.
- Grading runs in a separate environment from this one. Only the contents of `/app/output`
  cross over, so your solution must be fully self-contained there.

The harness, not your code, drives the loop: it calls `next_index()` exactly `budget` times,
and right after each call it calls `observe(index, label)` with the true class of the row you
named. Naming a row you've already bought is legal and still uses up one unit of budget. After
the last `observe`, `estimate()` is called once.

## How it's scored

- One *cell* is one held-out pool paired with one label budget `M`, where `M` is one of
  `{50, 100, 200, 400}`.
- Every cell is replayed over 4000 pinned seeds.
- For each seed, the harness builds a fresh `Estimator` and records the squared error

$$
(\text{estimate}() - R)^2
$$

  against the exact pool risk `R`, which it knows and you don't.
- A cell's score is

$$
\text{score} = \frac{\text{median over the 4000 seeds of your squared error}}{\text{median over the SAME 4000 seeds of the squared error of the reference}}
$$

- The reference draws its `M` rows uniformly at random, without replacement, from the same
  `rng` seed you're given, and returns the plain mean of their `-log target_probs[i, y_i]`.
- The graded metric is the **median of the cell scores** across every held-out cell.
- **Lower is better.** Matching the reference scores zero; every value strictly below the
  reference scores strictly higher, with no cutoff where further reduction stops counting. Push
  it as low as you can.

The held-out pools span several source corpora, classifier pairs, and calibration regimes,
including pools where the target classifier is accurate and well-calibrated and pools where
it's badly miscalibrated. No single fixed design wins on all of them.

## Development data

`/app/data/dev/` holds twelve labelled pools you can use however you like. Each
`/app/data/dev/<pool_id>/` contains:

- `target_probs.npy` - `float64` `(2000, n_classes)`
- `surrogate_probs.npy` - `float64` `(2000, n_classes)`
- `labels.npy` - `int64` `(2000,)`, values in `[0, n_classes)`
- `meta.json` - `{"pool_id", "n_pool", "n_classes"}`

The held-out pools come from disjoint shards of the same corpora and are never shipped to you.

## Execution contract

- Your module runs as an unprivileged user with **no writable directory**, including
  `/app/output`. It's re-executed from source for **every seed**, so no state survives from one
  seed to the next: do per-pool precomputation in `__init__`, not at module scope. The pool row
  order is re-randomised on every seed, so row indices mean nothing across seeds.
- `/app/output` must hold at most 2000 files totalling at most 256 MB.
- Wall clock limits: **5 seconds per seed**, and **900 seconds for a whole cell** (all 4000
  seeds of one pool-budget pair). A cell runs with one CPU core available to it.
- Any of the following makes that seed score the worst-case squared error: an exception, a
  `next_index()` outside `[0, n_pool)`, a non-finite `estimate()`, or exceeding the per-seed
  clock. Exceeding the per-cell clock, or trying to obtain more than `budget` labels in a seed,
  scores the worst case for the whole cell.
- A missing, empty, or unimportable `/app/output/estimator.py` scores zero.
