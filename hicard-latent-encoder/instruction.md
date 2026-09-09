# Fixed-width encoding for a very high-cardinality categorical column

A tabular regression dataset has 10 numeric covariates `X` and one categorical column `G`.
`G` has `n_categories = 1600` levels with very different frequencies. Write an encoder that
replaces `G` with **at most 8 numeric columns** without ever seeing the target. A fixed random
forest will use your encoding and the numeric covariates to predict a held-out target. Your
encoding should beat a full one-hot encoding of `G`.

## Deliverable

`/app/output/encoder.py`, defining:

```python
class Encoder:
    def __init__(self, n_categories: int, max_width: int): ...
    def fit(self, X: np.ndarray, G: np.ndarray) -> None: ...
    def transform(self, X: np.ndarray, G: np.ndarray) -> np.ndarray: ...
```

* `X` is C-contiguous `float64` of shape `(m, 10)`. `G` is `int64` of shape `(m,)` with values
  in `[0, n_categories)`. Some levels are absent from a fitting pool but can still appear in
  rows passed to `transform`.
* `fit` receives an unlabeled pool. **No target is ever passed to your code, at fit time or
  at transform time.**
* `transform` returns a `float64` NumPy array of shape `(n, k)`, one row per input row, with
  `0 <= k <= max_width` (`max_width` is `8`). Every entry must be finite and within the
  `float32` range. `k` must be the same on every call for a given instance. You choose the
  width, column order, and scale; `k = 0` is legal and means the forest sees only the
  covariates.
* `/app/output/` is the only directory that reaches the grading environment, so any helper
  modules, fitted parameters, or any other file `encoder.py` loads must live there too. Load them using a path
  derived from `__file__`, since the working directory at grading time is unspecified.
* `/app/data/` is read-only.

## How your encoder is scored

For each withheld instance, drawn from the same generator as the development instances, in a
fresh process:

1. `Encoder(n_categories, 8)` is constructed and `fit(X_train, G_train)` is called on that
   instance's 75% training rows, with no targets.
2. `E_train = transform(X_train, G_train)` and `E_test = transform(X_test, G_test)`.
3. `RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=8)` is fit on
   `hstack([X_train, E_train])` against `y_train`, and its mean squared error is measured on
   `hstack([X_test, E_test])` against the withheld `y_test`. The design matrix is cast to
   `float32` before fitting, matching what scikit-learn's trees do internally; `y` stays
   `float64`.
4. The instance's score is `100 * (mse_onehot - mse_yours) / mse_onehot`, where `mse_onehot`
   is the same forest fit on `hstack([X, onehot(G)])` with all `n_categories` indicator
   columns.

The reported metric is the unweighted mean of these per-instance percentages, computed in
float64 with no rounding. **Maximize it.** Matching the one-hot reference, or beating it by no
more than a trivial encoding would, is worth nothing. Above that, higher is strictly better,
with no ceiling. Per-instance percentages can vary by several points.

Per instance, `fit` and both `transform` calls run in one process, under these limits:

* Wall-clock: SIGKILLed 120 seconds after launch.
* Memory: 8 GiB address-space limit.

Any of the following zeroes the whole run, with no partial credit across instances:

* A missing entry point, an import error, an exception, or a timeout.
* An output that is not a `float64` rank-2 NumPy array.
* A width above 8, or a width that differs between the two `transform` calls.
* A row count that doesn't match the input.
* A single entry that is non-finite or outside the `float32` range.

## Development data

`/app/data/dev/instance_00000.npz` through `/app/data/dev/instance_00023.npz` are 24 instances generated the
same way as the withheld ones, but with unrelated level identities, so nothing memorized about
a specific level transfers. Each file holds:

| key | dtype | shape | meaning |
|---|---|---|---|
| `X` | `float32` | `(16000, 10)` | the covariates; cast to `float64` before use |
| `G` | `int16` | `(16000,)` | the categorical column; cast to `int64` before use |
| `y` | `float64` | `(16000,)` | the target, for your own experiments only |
| `train_idx` | `int32` | `(12000,)` | row indices of the 75% training pool |
| `test_idx` | `int32` | `(4000,)` | row indices of the 25% held-out rows |
| `n_categories` | `int64` | scalar | `1600` |
| `onehot_mse` | `float64` | scalar | that instance's reference error, from step 4 above |

Container environment:

* CPUs: 8 cores.
* Network: none.
* `OMP_NUM_THREADS=1`.
* Packages installed: `numpy`, `scipy`, `scikit-learn`.
