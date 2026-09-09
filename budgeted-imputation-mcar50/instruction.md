# Recovering missing values under 50% missingness, on a fixed CPU budget

Fill in the missing values in unseen numeric tables as accurately as possible.
Each value is missing independently with 50% probability, and each call has a 30 second
wall clock limit.

Write an imputer at `/app/output/impute.py`.

## Submission format

- File: `/app/output/impute.py`.
- It must define exactly this function:

```python
def impute(X_train: np.ndarray, X_eval: np.ndarray, seed: int) -> np.ndarray
```

- One call handles one dataset.
- `X_train`: shape `(n_train, d)`, dtype float64.
- `X_eval`: shape `(n_eval, d)`, dtype float64.
- `X_train` and `X_eval` are row-disjoint samples from the same numeric tabular dataset.
- Both matrices have `np.nan` at their missing entries.
- Missingness is MCAR at a rate of 0.50, drawn independently for each matrix.
- Features were transformed to a normal output distribution by a quantile transform fitted on the incomplete training matrix.
- `seed`: an integer you may use for any randomness you need.
- Nothing in the two matrices reveals the true value of a missing entry.

Return value:

- The completed evaluation matrix.
- Must be a `numpy.float64` array, or something `numpy.asarray` can convert to one.
- Shape `(n_eval, d)`, with row `i` and column `j` aligned to row `i` and column `j` of `X_eval`.
- Every entry must be finite: no `nan`, no `+inf`/`-inf`.
- Only entries that were missing in `X_eval` are scored.
- A return value with the wrong shape or dtype, or containing a `nan` or an infinity, invalidates the call (see "Execution and budget" for the consequence).

## Metric

- Target metric: imputation $R^2$, computed on hidden evaluation datasets not included in this container (see "Execution and budget").
- For each call and feature $j$, over that feature's masked evaluation entries:

$$R^2_j = 1 - \frac{\sum_{\text{masked}} (\text{true} - \text{returned})^2}{\sum_{\text{masked}} (\text{true} - \overline{\text{true}})^2}$$

- $\overline{\text{true}}$ is the mean of the true values over the same masked entries.
- Skip features with fewer than two masked entries, or where the masked true values' sum of squared deviations from their mean is at most 1e-12.
- Call score: unweighted mean of $R^2_j$ over the remaining features. $R^2$ can be negative.
- Reported metric: that score averaged unweighted, first over several independent mask draws per dataset, then over all evaluation datasets.
- Filling every missing entry with the mean of that column's observed training values scores 0.
- Anything strictly better scores strictly higher, with no ceiling, so keep pushing the metric up.

## Execution and budget

- Each call runs in a fresh process that imports your module and passes it matrices from a tabular dataset not present in this container.
- Time budget: 30 seconds of wall clock per call, covering interpreter start-up, importing your module, and the `impute` call together.
- The process is SIGKILLed at 30 seconds.
- These threading env vars are fixed to `1` in that process: `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`.
- A call that overruns the budget, raises an error, or returns something the contract above rejects scores 0 (the column-mean score) for that call.
- Each call gets a fresh working directory, `HOME`, and `TMPDIR`. Anything it leaves elsewhere is removed before the next call starts, so nothing survives between calls.
- The 30 seconds are per call and cannot be banked or carried over.
- Evaluation matrix sizes: `n_train` up to 10000, `n_eval` up to 2500, `d` up to 42 (as low as 3 on some datasets).

## What is here

- `/app/data/dev/`: six complete numeric tabular datasets as `.npz` files, plus `manifest.json` describing each one. These are development material only; none is an evaluation dataset.
- `/app/cellspec.py`: provides `build_cell(X, seed, n_train, n_eval)`, which builds `(X_train, X_eval, X_eval_true, mask_eval)` from a complete matrix using the row split, MCAR masks, and quantile transform described above. Also provides `imputation_r2(pred, true, mask)`, the metric function. The evaluation harness uses this same file to build its cells.
- `/app/cellrunner.py`: loads one call's inputs, imports `impute.py`, and saves what it returns. The evaluation harness runs this same file.
- `/app/devkit.py`: runs `/app/output/impute.py` over the development panel under the same 30-second budget and prints per-draw, per-dataset, and panel scores. Run `python3 /app/devkit.py --list` to show the panel.
- `/app/output/impute.py` currently contains column-mean imputation.

## Constraints

- Only `/app/output/` is collected for evaluation.
- Anything your module imports or loads at evaluation time must live under `/app/output/`.
- Limits under `/app/output/`: at most 2000 files and 512 MB total.
- Symbolic links under `/app/output/` are not copied.
- No network access.
- Container resources: 8 CPUs, 14 GB memory, 0 GPUs.
