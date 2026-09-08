# Unlabeled-pool pruning for active learning

An active-learning loop trains a binary sentiment classifier over a large unlabeled pool of
documents. Each round, before the acquisition function runs, your code picks which subset of
the still-unlabeled documents the acquisition function is allowed to consider. Everything else
about the loop is fixed and already built for you.

Write `/app/output/prune.py`. Your goal is to push the mean final-iteration macro-F1 as high as
possible.

Target metric: the mean final-iteration macro-F1 (in percent) on the hidden grading pools'
evaluation sets, averaged over 24 seeded end-to-end runs.

## The loop, per run

1. `prune` is called and returns at most 3125 pool indices (25% of the 12500-document pool).
2. The acquisition function picks 125 of the returned documents and reveals their stored
   labels. On iteration 0 there is no model yet, so it picks the 125 uniformly at random from
   what you returned. After that, it takes the 125 documents the current model is least
   confident about, breaking ties by ascending pool index. If you return fewer than 125
   indices, only that many documents get labelled that round.
3. The classifier (multinomial logistic regression, `C=1.0`, `lbfgs`, `max_iter=1000`, over
   fixed 128-dimensional frozen text features) is refit from scratch on every label acquired so
   far, then scored on a held-out evaluation set.

There are five iterations, so 625 labels at most. The graded number is the arithmetic mean,
over 24 seeded end-to-end runs, of the macro-F1 in percent the classifier reaches on the
evaluation set after the fifth iteration. Grading uses pools built by the same procedure as the
ones you're given, but from disjoint source documents you've never seen.

## The entry point

`/app/output/prune.py` must define, at module top level:

```python
def prune(pool_texts, available, labeled_indices, labeled_labels,
          keep, iteration, rng_seed) -> list[int]
```

- `pool_texts` - the whole pool as a `list[str]` of raw documents, length 12500. The same object
  is passed on every call within a run.
- `available` - `list[int]`, the pool indices not yet labelled.
- `labeled_indices`, `labeled_labels` - `list[int]`, what's been labelled so far in this run and
  the labels that came back. Both empty on iteration 0.
- `keep` - `3125`, the largest selection allowed.
- `iteration` - `0` to `4`.
- `rng_seed` - this run's seed. Use it if you want your own randomness to be reproducible.

The return value must be a `list`, `tuple`, or 1-D integer numpy array of distinct ints, every
one present in `available`, with length between `1` and `keep` inclusive. Returning fewer than
`keep` (pruning harder) is allowed. Order doesn't matter: the simulator sorts the indices and
treats the result as a set.

The module is imported once per run in a fresh process, so module-level state persists across
that run's five calls but is discarded before the next run starts.

## Budgets and hard limits

- Module import: 60 s per run. Each `prune` call: 20 s of wall clock.
- 4 GiB of address space per run. Threading libraries are pinned to one thread; the grading
  container has 8 vCPUs, 14 GiB of RAM, no GPU, and no network.
- The 24 runs are independent. Your module gets a read-only copy of the deliverable directory.
  The only writable path is the private scratch directory named by `TMPDIR` (also `HOME`),
  which is deleted when the run ends. `/tmp`, `/var/tmp`, and `/dev/shm` are not writable.
  Nothing you write during scoring survives into another run.
- Anything else scores 0: no `/app/output/prune.py`, an empty one, an import error, a wrong
  function name or arity, an exception, a blown budget, a return value of the wrong type or
  dtype, a duplicate index, an index not in `available`, or a length of 0 or above `keep`. A
  single failing run invalidates the whole submission.
- A submission containing a symlink or hard link anywhere under `/app/output` is rejected and
  scores 0.

## What crosses to the grader

- Only `/app/output/` travels, in its entirety. Every module, model, vocabulary, threshold
  table, cached statistic, and data file your `prune` loads at grade time must live inside that
  directory; nothing you write anywhere else survives.
- Packages available at grade time: `numpy`, `scipy`, `scikit-learn`, `joblib`,
  `threadpoolctl`, and the standard library.
- Absent when you're scored: `/app/data/`, `/app/harness/`, and `/app/run_dev.py`. Do not
  import from them at grade time.
- Your `prune` receives raw text only. The frozen feature matrices belong to the simulator, not
  to you.

## What you are given

- `/app/data/pool_{a,b,c}.jsonl.gz` - three development pools, one JSON object per line with
  `text` and the `label` the oracle would return for it. 12500 lines each.
- `/app/data/pool_{a,b,c}_X.npy` - the frozen 128-d features for each pool, `float32`, shape
  `(12500, 128)`; `/app/data/pool_{a,b,c}_y.npy` - the same labels as `int8`.
- `/app/data/dev_eval.jsonl.gz`, `/app/data/dev_eval_y.npy`, `/app/data/eval_a_X.npy`,
  `/app/data/eval_b_X.npy`, `/app/data/eval_c_X.npy` - a development evaluation slice of 2500
  held-out labelled documents and its features under each pool's feature map.
- `/app/harness/al_loop.py` - the simulator, byte-identical to the one that scores you, plus the
  process isolation and budget enforcement it runs under.
- `/app/run_dev.py` - runs the loop end to end on the development pools with your pruner and
  prints per-pool and overall means:

  ```
  python3 /app/run_dev.py --draws a,b,c --seeds 0-11
  ```

The simulator, the acquisition function, the classifier, and the evaluation are all fixed. The
grader uses its own copies, so editing anything under `/app/harness/` changes nothing about how
you're scored.

## Scoring

A mean final-iteration macro-F1 of 77.9 percent or lower scores 0. Above that floor, every
improvement receives a higher score.
