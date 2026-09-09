# Access-request risk scoring from high-cardinality ID columns

Rank unseen access requests by how likely they are to be granted.

Employee access requests are recorded as nine nominal ID columns (resource, manager,
role rollups, department, title, family) and a binary outcome: the request was granted
or it was not. Nothing about the columns is ordinal or interpretable; each value is an
opaque identifier, and the cardinalities run from 67 to over 7000.

Write `/app/output/solution.py` exposing exactly one entry point:

```python
def fit_predict(train_X, train_y, pool_X, eval_X, seed) -> numpy.ndarray:
    ...
```

The verifier calls it once per split with data it holds back from this container, and
scores the array it returns.

## Arguments

| name | type | shape | contents |
|---|---|---|---|
| `train_X` | `pandas.DataFrame` | 8500 x 9 | `int64` columns `f0`..`f8`, the nine nominal codes |
| `train_y` | `numpy.ndarray` | (8500,) | `int8`, values in `{0, 1}`, aligned to `train_X` rows |
| `pool_X` | `pandas.DataFrame` | 15300 x 9 | same nine columns, **no labels** |
| `eval_X` | `pandas.DataFrame` | 5000 x 9 | same nine columns, the rows to be scored, in scoring order |
| `seed` | `int` | | the split's identifier; pin your own randomness with it |

`pool_X` is the row-shuffled union of 10300 held-out unlabelled rows and the 5000 rows
of `eval_X`, all drawn from the same source as `train_X`. Which rows are which is not
marked. You may use `pool_X` however you wish, or not at all. No label for any row of
`pool_X` or `eval_X` exists anywhere in this container.

## Return value

A `numpy` array of shape `(5000,)`, finite, real-valued, where a larger value means a
higher predicted probability that the request was granted. Only the induced ranking is
used, so no calibration is needed and any monotone rescaling is equivalent.

## Score

The score is the mean ROC-AUC over twelve fixed splits of a sealed evaluation set.
`fit_predict` is called once per split. The grader compares its output with the held-back
labels using `sklearn.metrics.roc_auc_score`. All twelve splits have equal weight.

Maximise it. There is no target value and no point at which further improvement stops
mattering. A trivial predictor that does no modelling (a constant, or a ranking read off
a single column) earns zero credit, and so does any submission that does not clear the
best such trivial baseline by a margin. Once past that margin, a higher mean ROC-AUC is
strictly better and every genuine improvement earns strictly more.

## Hard constraints

- `/app/output/solution.py` must exist, be non-empty, and define `fit_predict` with
  exactly the five positional parameters above. Anything else scores zero.
- Everything `fit_predict` needs at call time must live under `/app/output`, which is
  the only directory that reaches the verifier. Its whole tree must be **at most 131072
  bytes** and must contain **no symbolic link and no hard link**; either one scores zero.
- Each call has **100 seconds of wall-clock time**. The process is SIGKILLed at that
  point. A call that exceeds it, raises, exits non-zero, or returns something that is
  not a finite real array of shape `(5000,)` contributes a ROC-AUC of 0.5 for that split
  and the panel mean is reported as usual.
- The whole panel shares a budget of 1200 seconds of submitted-code wall-clock time;
  splits not reached inside it also contribute 0.5.
- Every call is a fresh process. Nothing written during one call survives into the next,
  so state cannot be carried between splits except through the files under `/app/output`.

## Environment

8 CPU cores, no GPU, no network, **14336 MB of memory**. `OMP_NUM_THREADS`,
`MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS` are `8` in the calling process. Installed
and importable: `numpy`, `pandas`, `scipy`, `scikit-learn`, `catboost`, `lightgbm`.

## What you have locally

- `/app/data/dev.csv`: 5000 labelled rows, columns `f0`..`f8` plus `y`. These rows are
  always part of `train_X` at scoring time; they are yours to use however you like.
- `/app/local_panel.py`: builds `(train_X, train_y, pool_X, eval_X, eval_y)` triples out
  of those labelled rows, with the same nine columns and the same
  `len(pool_X) == 1.8 * len(train_X)` geometry as a scored call, scaled down to what 5000
  labelled rows support. Run `python3 /app/local_panel.py` to evaluate your
  `/app/output/solution.py` over several independent local draws. It prints each draw's
  ROC-AUC and their mean. A single draw is noisy, and the local panel is much smaller than
  a scored one.

The sealed evaluation rows are not in this container and are not derivable from
`/app/data/dev.csv`.
