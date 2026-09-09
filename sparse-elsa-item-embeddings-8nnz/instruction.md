# Item embeddings under a hard 8-nonzero budget

Build compact item embeddings that retrieve a user's held-out items from their known interactions.

`/app/data/train.npz` holds a binarized user-item interaction matrix. `scipy.sparse.load_npz`
loads it as a `csr_matrix` of shape `(48366, 10000)`, dtype float32, every stored value `1.0`.
Rows are users, columns are items. Item ids are anonymized and carry no meaning outside this
matrix.

Retrieval uses one item-embedding matrix `A` with 10,000 rows and `d` columns, applied through a
fixed rule. Let `A_norm` be `A` with every row rescaled to unit L2 norm (an all-zero row stays
all-zero), and let `x` be a user's binary interaction vector over the 10,000 items:

```
r = x @ A_norm @ A_norm.T - x
```

Items are recommended to that user in order of descending `r`.

Build `A` to serve that rule as well as you can, under a hard storage budget:

- at most 8 nonzero entries per row of `A`
- `1 <= d <= 65536`; you choose `d` in that range, and its value neither rewards nor penalizes you directly
- rows with fewer than 8 nonzeros, including empty rows, are allowed
- stored zeros don't count against the budget

## Deliverable

`/app/output/solution.py`, exposing:

```python
def build_item_embeddings(X_train, n_items, max_nnz_per_item, max_dims, seed):
    ...
```

It's called exactly once, positionally, so parameter names are yours to choose:

- `X_train` - a `scipy.sparse.csr_matrix`, float32, with sorted indices, identical in content and
  layout to `/app/data/train.npz`
- `n_items` = `10000`, `max_nnz_per_item` = `8`, `max_dims` = `65536`
- `seed` - a fixed integer, the same on every graded run

Return value:

- a `scipy.sparse` matrix (any format) or a 2-D `numpy.ndarray`
- shape `(10000, d)`
- real, finite values

Submission format:

- `/app/output/` is everything that crosses over to grading
- `solution.py`, plus every module, weight file, and config it reads at run time, must live inside
  `/app/output/`
- it's imported as the top-level module `solution`, with its own directory first on `sys.path`
- that directory gets copied to another path before running, so resolve anything you load relative
  to `__file__`, never by an absolute path under `/app`
- nothing else from `/app` exists at grade time - `/app/data/` in particular is gone, so pull
  anything you need from the training data out of `X_train`, not the file

Runtime environment:

- fresh process, unprivileged user, 8 CPU cores, 14336 MB RAM, no GPU, no network
- OpenMP/MKL/OpenBLAS/NumExpr thread counts set to 8
- process killed 2400 seconds after it starts, including the time to import `solution`

## How `A` is scored

Metric: mean nDCG@100 over sealed evaluation rows.

Evaluation rows:

- built from 2,500 users present in neither `/app/data/train.npz` nor `/app/data/val.npz`
- each of those users' interaction rows is split into an input part and a held-out part, five
  times
- each fold hides $\lceil 0.2 \cdot \text{row\_nnz} \rceil$ of the user's items, chosen without
  replacement, leaving the rest as input
- each fold contributes one evaluation row per user

Per-row score, with `x` the input part:

- score all 10,000 items by `r` as defined above
- force `r` to $-\infty$ on the items already in that row's input part
- rank items by descending `r`, breaking ties by ascending item index, and keep the top 100
- $\mathrm{DCG} = \sum \dfrac{1}{\log_2(\text{rank} + 1)}$ over held-out items that land in that
  top 100, rank counted from 1
- $\mathrm{IDCG} = \sum_{i=1}^{\min(100,\, k)} \dfrac{1}{\log_2(i + 1)}$, where `k` is the number
  of held-out items
- the row's nDCG is $\mathrm{DCG} / \mathrm{IDCG}$

Final score: the per-row values, averaged uniformly. Higher is strictly better, with no cutoff
where further improvement stops mattering. A submission that does no better than a trivial
baseline earns nothing.

A submission scores 0 if any of the following holds:

- `/app/output/solution.py` is missing or fails to import
- it exposes no callable `build_item_embeddings`
- the call raises
- it hasn't returned within its 2400-second budget
- the returned object isn't a 2-D sparse matrix or `ndarray`
- it doesn't have 10,000 rows
- `d` is outside `1 .. 65536`
- it contains a non-real or non-finite value
- any row has more than 8 nonzeros

## Local material

- `/app/data/val.npz` - `csr_matrix` (2500, 10000), same format as the training matrix: 2,500
  further users, disjoint from the 48,366 training users and from the 2,500 evaluation users.
  Nothing about the evaluation rows themselves is available locally.
- `/app/check_contract.py` - runs your entry point the way grading does and reports the shape,
  the per-row nonzero counts, `d`, and the wall clock it used. It computes no metric.

Packages installed: `numpy`, `scipy`, `scikit-learn`, CPU `torch`.

Filesystem: `/app/output` and `/tmp` are writable; `/app/data` is read-only.
