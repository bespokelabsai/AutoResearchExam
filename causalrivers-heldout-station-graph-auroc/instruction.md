# Directed edge scores for river gauge subgraphs

A river gauge network is a directed graph over discharge measuring stations. An edge `i -> j`
means water measured at station `i` flows onward to station `j`, with no gauged station in
between. Given five years of discharge data from five stations at a time, score how likely each
ordered pair is to be such an edge.

## Submission format

- Write `/app/output/predict.py` defining a function `score_edges(series, freq_minutes) ->
  numpy.ndarray`.
- `series`: a C-contiguous `float32` array of shape `(175296, 5)`, 15-minute samples from
  2019-01-01 00:00 to 2023-12-31 23:45, raw discharge in $m^3/s$.
- Missing measurements are `NaN` (a few per cent overall, ranging from 0% to 66% for an
  individual station). A handful of readings are physically impossible negatives.
- `freq_minutes` is the int `15`.
- The column order of `series` is a random permutation of the five stations. No identifier,
  coordinate, river name, or other ordering hint is passed.
- Return a `(5, 5)` array of real numbers. Entry `(i, j)` scores the hypothesis "column `i` is
  directly upstream of column `j`" - higher means more likely.
- Every off-diagonal entry must be finite. The diagonal is ignored. Only the ranking induced by
  the 20 off-diagonal entries is read, so any scale or offset works.

## Metric

- Each graded sample is a connected 5-node induced subgraph of a held-out gauge network.
- Its 20 off-diagonal scores are ranked against the true directed edges of that subgraph, using
  a tie-aware AUROC (self-links excluded, so a constant matrix scores 0.5).
- The task metric is the unweighted mean of that AUROC over the graded samples.
- Objective: maximize. Higher is strictly better, with no threshold or cutoff past which further
  improvement stops counting.
- A submission no better than cheap heuristic baselines earns 0 reward; reward rises strictly
  with the metric above that.

## Packages

numpy, pandas, scipy, scikit-learn, statsmodels, networkx and torch (CPU build) are installed
and importable in both the development and the graded environment. No other packages and no
network access are available.

## Development data

`/app/data` holds two labelled gauge networks. Their stations are disjoint from each other and
from the graded stations, and each region is gauged by a different set of agencies.

| file | contents |
| --- | --- |
| `/app/data/dev_a_series.npy` | float32 `(175296, 494)`, same time axis, units and defects as `series` |
| `/app/data/dev_a_edges.csv` | `cause,effect` - column indices of the 490 true directed edges of region a |
| `/app/data/dev_b_series.npy` | float32 `(175296, 259)` |
| `/app/data/dev_b_edges.csv` | the 200 true directed edges of region b |

- Each edge list is the induced graph over the stations of its own region: a pair absent from
  the list has no edge between those two stations.
- `/app/dev_data.py` has loaders.
- Nothing in this container describes the graded stations.

## Graded run

- 500 samples per graded run, in groups of at most 40. Each group runs in a fresh process that
  imports `/app/output/predict.py` once, then calls `score_edges` once per sample in that group.
  Nothing kept in module state or on disk carries across groups.
- The whole graded run (every import plus every call) must finish inside 1200 s of wall clock.
  Overrunning that scores 0.
- Hardware: 8 vCPUs, 14 GB RAM, 0 GPUs, no network. `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS` and `NUMEXPR_NUM_THREADS` are all `1` in the graded process, and
  `torch.set_num_threads(1)` is called before your module is imported.
- Grading runs in a separate environment from this one: `/app/output` is the only artifact that
  crosses over, so the solution must be self-contained inside it. Nothing else in this
  container - files elsewhere, installed state, anything not under `/app/output` - is available
  when grading runs.
- `/app/output` is copied to a directory the graded process can read but not write, and its
  contents must total at most 50 MB.
- Modules and data files shipped next to `predict.py` can be imported and read normally; that
  directory is on `sys.path`. Symbolic links are not carried across the copy.
- The only writable location during grading is `$TMPDIR`, which is empty at the start of every
  group.
- Scoring 0 for the whole run: a missing, empty or unimportable `predict.py`, a missing or
  non-callable `score_edges`, a group process that dies, or an overrun time budget.
- Scoring 0.0 for a single sample: a call that raises, or that returns anything other than a
  `(5, 5)` matrix with finite off-diagonal entries.
- No network access is available at any point, in either environment.
