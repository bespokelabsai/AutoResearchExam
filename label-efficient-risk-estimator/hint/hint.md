# Hint

From prior autonomous-research runs on this task. Directions only, no tuned constants.

- **Frame it as survey sampling: design plus model-assisted estimator.** Stratify the pool by a proxy for each example's loss and allocate the budget by a Neyman-type rule. The best proxy is the *gap* between the two possible losses, which is what drives variance.
- The surrogate's pool mean is known exactly for every row, so use it as a **control variate / GREG correction** on the sampled residuals.
- **Robustness beats aggression.** Surrogate quality varies per pool and you are not told how good it is. A **within-stratum** control variate cancels exactly regardless of calibration, and this was a large jump for a run whose aggressive regression estimator kept winning on development pools and losing on held-out ones.
- **Shrink the residual correction slightly below one** — the metric is a median squared error, so a small bias bought with a variance reduction is a good trade, and the optimum is consistently on the shrunken side.
- Fit a small calibration model on the labels you buy, weight its observations by the loss gap, and blend rather than committing to one estimator; select between robust and aggressive variants by leave-one-out CV on the purchased labels.
- Unlabelled pool statistics identify the regime, so shrinking toward a regime-appropriate prior (strength decaying in the budget, gated by similarity) helps — but gate it hard and validate on pools held out at the *source* level.
