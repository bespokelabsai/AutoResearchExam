# 16-bit hash codes for multi-label image retrieval

Build a binary encoder that maps a frozen 512-dimensional image feature vector to a 16-bit code,
so that the Hamming distance between two codes reflects how much semantic content the two images
share.

## Data

- `/app/data/train_features.npy`: shape `(10000, 512)`, `float32`. Pooled features from a frozen
  convolutional backbone, one row per image. The backbone itself is gone, so these vectors are
  the only information you have about the images.
- `/app/data/train_labels.npy`: shape `(10000, 80)`, `uint8`. Multi-hot category labels for the
  same rows, categories in ascending order. Each image has between 1 and 15 of the 80 categories;
  the mean is 2.9.

## What to submit

Write `/app/output/hash_model.py` with exactly these two functions:

```python
def train(train_features, train_labels, seed):
    """train_features: (N, 512) float32, train_labels: (N, 80) uint8, seed: int.
    Returns any object (the 'state') that encode() needs."""

def encode(state, features):
    """features: (M, 512) float32. Returns an (M, 16) int8 array,
    every element either -1 or +1."""
```

Rules for the submission:

- You can put helper modules and learned weight files anywhere under `/app/output/`. That whole
  directory is copied to the evaluation environment and put on the import path; nothing outside
  it is.
- `/app/output/` must stay under 1 GiB.
- At evaluation time `numpy`, `scipy`, `scikit-learn`, and CPU PyTorch are available, at the
  same versions as in this environment.

## How the evaluation run works

- `train` is called once, on the 10,000 training rows above, with `seed=0`.
- `encode` is then called several times, once per feature matrix: a query set of images held
  out from everything else, and a 25,000-row retrieval database, which contains the 10,000
  training images plus 15,000 more.
- Your code never sees labels for the query or database images.
- Each `encode` call is independent; no two matrices are ever passed together.
- Permutation requirement: one `encode` call receives a row-permuted copy of a matrix that
  another call also receives. The codes must satisfy `encode(state, X[P]) == encode(state,
  X)[P]` row for row, i.e. the code for an image depends only on that image, never on its
  position or on what else is in the batch.
- Time limit: the whole run (the one `train` call plus all `encode` calls) is SIGKILLed 1200
  seconds after it starts.
- Environment: non-root user, scratch working directory, no network, 8 CPU cores, 14 GiB RAM, no
  GPU.
- `train_features.npy` and `train_labels.npy` are passed to `train` as arrays; don't read them
  from `/app/data` during evaluation, and don't rely on anything else from the filesystem.

## Scoring

Ranking: for each query, rank the 25,000 database images by ascending Hamming distance between
codes (ties broken by ascending database row index), and keep the top 5,000. A database image is
relevant to a query if they share at least one of the 80 categories.

Metric (mAP@5000), averaged over all queries:

$$AP = \frac{\sum_{k=1}^{5000} \text{precision@}k \cdot \text{relevant@}k}{\text{number of relevant items inside the top 5000}}$$

A query with no relevant item in the top 5,000 scores `AP = 0`. Higher mAP@5000 is better, with
no ceiling on how much improvement counts; a degenerate encoder (constant code, per-image random
code, fixed random projection) scores zero.

The run scores zero if any of the following happen:

- It crashes.
- It exceeds the 1200-second budget.
- `encode` returns an array of the wrong shape or dtype.
- `encode` returns a value other than -1 or +1.
- The permutation requirement above is violated.
- `/app/output/hash_model.py` is missing or empty.
