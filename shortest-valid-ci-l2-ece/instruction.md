# Shortest valid confidence interval for the squared l2 calibration error

Given predicted class probabilities and their true labels, return a confidence interval for the
squared calibration error defined below. The interval should be as short as possible while still
covering the true value in at least 88% of repeated samples.

## Submission

- File: `/app/output/interval.py`, plus any helper module it imports. Nothing outside
  `/app/output` reaches the grader.
- It must expose:

```python
def ece2_interval(Z, Y, k, alpha, rng) -> tuple[float, float]:
    ...
```

- The function must return a `1 - alpha` confidence interval for the squared top-1-to-k l2
  expected calibration error of the predictions in `Z`.
- Return `(lo, hi)`: two finite floats with `0.0 <= lo <= hi <= k`.

## The estimand

For a K-class predictor, let `Z` hold predicted probability vectors and let `r_1, ..., r_k` be
the classes with the k largest entries of a row, so `Z_(1:k)` are those entries in
non-increasing order and `Y_{r_(1:k)}` the matching label indicators. With the expectation over
the joint law of the row and its label,

$$
\mathrm{ECE}^2_{1:k} = \mathbb{E} \left\| \mathbb{E}\left[ Y_{r_{(1:k)}} - Z_{(1:k)} \,\middle|\, Z_{(1:k)} \right] \right\|_2^2
$$

This is zero exactly when the top-k predicted probabilities are calibrated, and it lies in
`[0, k]`.

## Arguments and return

| name | contract |
|---|---|
| `Z` | C-contiguous `float64` array, shape `(n, K)`, rows are probability vectors summing to 1 |
| `Y` | `int64` array, shape `(n,)`, realised class indices in `[0, K)` |
| `k` | `int` in `{1, 2, 3}`, always `k < K` |
| `alpha` | `0.10` |
| `rng` | a freshly seeded `numpy.random.Generator`, one per call |

- A call counts as a non-covering interval of length `k` if it raises, returns an invalid value,
  or runs for more than 2.0 seconds.
- `rng` is the only random number generator supplied by the grader. The grader draws the same
  data each time. Seed any other source of randomness so repeated grading gives the same result.

## How it is scored

- Your routine runs against a sealed panel of data-generating settings that is not in this
  container.
- Each setting is graded over `R = 2000` seeded replications. Each replication is an independent
  draw of `(Z, Y)` with its own true `ECE^2_{1:k}`. The grader calls your routine once for each
  replication in a fresh process.
- For setting `s` with `k = k_s`:
  - `coverage_s` = fraction of the 2000 replications whose interval contains that
    replication's true `ECE^2_{1:k}`
  - `relative_length_s = mean(hi - lo) / k_s`
  - score:

$$
\text{score}_s =
\begin{cases}
\text{relative\_length}_s & \text{if } \text{coverage}_s \ge 0.88 \\
1.0 & \text{otherwise}
\end{cases}
$$

- The grader reports the unweighted mean of `score_s` over all settings as `metric`.

Lower is better. The interval `(0.0, k)` always covers the true value, but it gives a metric of
`1.0` and earns zero credit. Every metric below `1.0` earns more credit as it decreases.

## The sealed panel

Settings are drawn from the generator in `/app/panel.py`. Across the panel:

* `K` is in `{2, 3, 5, 10, 50}`, `k` in `{1, 2, 3}` with `k < K`, `n` in `{200, 500, 2000, 10000}`.
* The distribution of `Z` mixes a uniform distribution on the simplex with a Dirichlet or
  logit-normal distribution. The density of `Z_(1:k)` is bounded away from zero on
  `{z_1 >= ... >= z_k >= 0, k/K <= sum z <= 1}`.
* The calibration curve is a smooth per-rank over- or under-statement of the predicted
  probability, and labels depend on a row only through its top-k entries.
* Each setting carries a fixed 41-point grid of true squared calibration errors, and every
  replication draws one of them, so the true value **varies between replications of the same
  setting**. Some settings include exactly calibrated replications; others reach a squared
  error of order 0.1.

## Data notes

* `/app/panel.py` is the generator the grader uses. `draw_replication(spec, theta, rng)`
  returns `(Z, Y)` for a setting `spec` whose true `ECE^2_{1:k}` is exactly `theta`.
  `theta_for(spec, rng)` picks a replication's `theta` from the setting's grid.
  `estimate_a(spec, n_draws, seed)` and `sigma0_sq(K, k)` are the same Monte-Carlo helpers used
  to build the panel.
* `/app/data/dev_panel.json` holds 24 development settings, drawn three independent times from
  that same generator, each with its `A`, its `theta_grid`, and the rest of its parameters. Use
  it to generate replications with known true values and measure coverage and length yourself.
  The three draws differ, so a choice tuned to one need not work for another.

## Packages and environment

* `python3` with `numpy` and `scipy` is available.
* There is no network.
* Each grade-time process serves at most one replication per setting and is destroyed
  afterward, with no writable location surviving it, so nothing carries over from one call to
  the next.
