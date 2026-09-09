# Hint

From prior autonomous-research runs on this task. Directions only, no tuned constants.

- **Debias the estimator first.** The plug-in binned estimate of a *squared* calibration error is biased upward — each bin's squared mean gap carries a variance term. Use the unbiased within-bin U-statistic form; everything else is second order.
- Get the variance from an explicit asymptotic decomposition (linear projection plus the second-order term) or a delete-one jackknife. A bootstrap on this estimand was tried repeatedly and was unstable.
- **Coarser binning than the textbook rate was consistently better**: fewer, fuller bins cut estimator variance, and shorter intervals follow.
- **Coverage is a constraint, length is the objective** — unused coverage above the threshold is wasted score.
- **Use asymmetric critical values.** The finite-sample error is skewed, so the two endpoints do not cost the same; tightening the lower one buys length far more cheaply.
- **Stratify the constants** by rank, class count and sample size. Coverage slack varies a lot by regime, and per-regime critical values and bin resolutions were the main source of sustained improvement.
- Model coverage as a smooth function of the safety multiplier, note the systematic development-to-graded offset, and extrapolate to just above the threshold — that replaces a random walk with two or three decisive steps.
