# Imputing a severely and heterogeneously incomplete firm-characteristic panel

Recover missing values in unseen firm characteristic panels as accurately as possible. Each panel
is an array with shape `(60 months, N firms, 45 characteristics)`.

Data:

* Every value has been cross-sectionally ranked within its `(month, characteristic)` slice
  and rescaled to `[-0.5, 0.5]`.
* Roughly 83% of the entries are missing and appear as `NaN`.
* Missingness is not random: firms differ enormously in how much of their history survives.

## Deliverable

`/app/solution/impute.py`, exposing exactly one entry point:

```python
def impute(panel: np.ndarray, seed: int) -> np.ndarray:
    ...
```

* `panel` - a C-contiguous `float64` array of shape `(60, N, 45)`, with `NaN` at every
  entry you cannot see and the observed value elsewhere. `N` varies between panels and is
  between 350 and 450.
* `seed` - an `int`. Use it for any stochastic initialization, so a rerun on the same panel
  returns the same array.
* Return a real-valued floating array of shape `(60, N, 45)`, finite everywhere (no `NaN`,
  no `inf`), with every value inside `[-1.0, 1.0]`.

`/app/solution/` is yours: put any helper modules, cached artifacts, or configuration there
and they travel with the submission. Only `impute.py` is imported, and only `impute` is
called. Nothing outside `/app/solution/` is preserved. The submitted tree must contain no
symbolic links and no hard links: either one gets the whole submission rejected. Grading
runs in a separate environment from wherever you develop this: `/app/solution/` is the
only artifact carried over, so your solution must be fully self-contained inside it.
Nothing else from this workspace persists, and there is no network access at grading time
to fetch anything you didn't already put there.

## Evaluation

* You are scored on panels you have never seen.
* In each sealed panel, a set of cells that were originally observed has been blanked
  before you receive the array. Those held-out cells appear as `NaN`, indistinguishable
  from cells that were never observed. Nothing tells you which cells they are.
* Metric: out-of-sample imputation $R^2$, pooled over one panel's held-out cells:

$$R^2_{imp} = 1 - \frac{\sum_m (x_m - \hat{x}_m)^2}{\sum_m (x_m - \bar{x})^2}$$

  where $x$ are the true values at those cells, $\hat{x}$ your returned values there, and
  $\bar{x}$ the mean of the true values.
* Reported score: the arithmetic mean of $R^2_{imp}$ over the sealed panels.
* A trivial imputer that explains none of the held-out variance scores zero. Any strictly
  higher $R^2_{imp}$ scores strictly higher, with no cutoff where further improvement
  stops mattering.
* Maximize the reported score.

## Development data

`/app/data/dev/instance_00.npz` through `/app/data/dev/instance_07.npz` are eight panels
drawn by the same procedure as the sealed ones, each with ground truth included so you can
score yourself. Every file holds four arrays:

| key | dtype, shape | meaning |
| --- | --- | --- |
| `panel` | `float64 (60, N, 45)` | exactly what `impute` receives |
| `eval_mask` | `bool (60, N, 45)` | `True` at the held-out cells |
| `eval_true` | `float64 (M,)` | the true values at those cells, in C order of `eval_mask` |
| `seed` | `int64` | the `seed` argument passed alongside `panel` |

The sealed panels are drawn with independently redrawn generator hyperparameters, so the
dev panels do not pin them down.

## Execution contract

* Your module is imported and `impute` is called once per sealed panel, in a fresh
  unprivileged process per panel, with no network.
* Timeout: each call is killed 240 seconds after the process is launched.
* Threads: `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and
  `NUMEXPR_NUM_THREADS` are all pinned to `1`.
* Concurrency: up to four panels are graded concurrently on the 8 vCPUs, 14336 MB (14 GB)
  of RAM, and 0 GPUs the container provides.
* Packages: `numpy`, `scipy`, and `scikit-learn` are installed. There is nothing to
  download.
* Failure handling: a panel whose call raises, times out, returns nothing, or returns
  anything that violates the shape, dtype, finiteness, or range contract above, or whose
  saved output exceeds 64,000,000 bytes, scores that panel as the trivial imputer would.
