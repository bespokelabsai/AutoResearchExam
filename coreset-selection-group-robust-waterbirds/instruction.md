# Coreset selection against an unlabeled slice

You're given embeddings for a pool of bird photos, produced once by a frozen image backbone.
Each embedding is 512-dimensional. Your job is to pick 240 of these rows to form the training
set for a fixed downstream classifier.

The classifier is scored on a held-out evaluation set split into four slices: the two class
labels crossed with a binary attribute. That attribute is never disclosed and never labeled
anywhere in the material you're given.

## Data provided

| path | contents |
|---|---|
| `/app/data/dev_pool_features.npy` | `float32`, shape `(2397, 512)`, development pool embeddings, not L2-normalized |
| `/app/data/dev_pool_labels.npy` | `int64` in `{0, 1}`, shape `(2397,)`, row-aligned to the features |
| `/app/data/dev_draws.npy` | `int64`, shape `(4, 2038)`, four subsamples of the development pool, drawn by the same stratified procedure used for the pools you'll be graded on |

These files are read-only. You may only write to `/app/output/`.

## Deliverable

Write `/app/output/selector.py`, with exactly one entry point:

```python
def select_coreset(features, labels, budget, seed):
    ...
```

Contract:

- `features`: `float32` array of shape `(N, 512)`, embeddings of a graded pool you've never
  seen, from the same backbone as the development pool. `N` is roughly 2000.
- `labels`: `int64` array of shape `(N,)`, values in `{0, 1}`, row-aligned to `features`.
- `budget`: `int`, always `240`.
- `seed`: `int`, yours to use for any randomness of your own.
- Returns: an `int64` array of shape `(240,)` with **unique** indices in `[0, N)`, of which
  **exactly 120 have label 0 and exactly 120 have label 1**.

Helper modules can live beside `selector.py` in `/app/output/`. Everything in that directory
travels with the entry point at grading time, so anything your entry point loads at call time
needs to be there too.

## Runtime constraints

- Your function is called once per graded pool draw, with 150 seconds of wall-clock time per
  call.
- Runs as an unprivileged user: no network, no access to the evaluation set, no attribute
  labels.
- Runs with `OMP_NUM_THREADS=1`, on a machine with 8 CPUs, 14336 MB of memory, and 0 GPUs.
- Available packages: `numpy`, `scipy`, `scikit-learn`. Nothing else is added at grading time.
- A call that raises an exception, exceeds the time budget, or returns anything that violates
  the contract above scores zero for the whole submission.
- A missing or unimportable `/app/output/selector.py` also scores zero for the whole
  submission.

## The fixed downstream classifier

You cannot change it, and it is not shipped to you. On each graded pool draw, it:

- Trains only on your selected rows.
- Standardizes those rows by their own per-dimension mean and standard deviation, flooring any
  standard deviation below `1e-6` to `1e-6` before dividing.
- Fits a two-class multinomial logistic regression from a seeded initialization: weights drawn
  i.i.d. from a zero-mean normal distribution with standard deviation `0.01`, biases starting
  at zero.
- Trains for 100 epochs of mini-batch SGD: batch size 32, learning rate 0.05, momentum 0.9,
  weight decay 1e-4, data shuffled each epoch.
- Applies the trained classifier to the evaluation embeddings.

## Metric

- Each run's score is the classifier's accuracy, in percent, on the weakest of the four slices
  described above.
- The submission's score is the mean of that worst-slice accuracy over 7 x 5 = 35 runs (seven
  graded pool draws, five pinned classifier seeds), in percent.
- Above a floor set a small margin above the trivial baselines, higher is strictly better, with
  no ceiling where further improvement stops mattering.
- A uniform random class-balanced draw scores zero, as does any selection up to that floor.
