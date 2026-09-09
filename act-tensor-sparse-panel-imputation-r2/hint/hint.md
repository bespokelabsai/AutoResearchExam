# Hint

From prior autonomous-research runs on this task. Directions only, no tuned constants.

- **It is a tensor — use all three modes.** A single unfolding captures one kind of structure. Combine the characteristic mode (cross-sectional covariance), the firm mode (each firm's whole trajectory) and the time mode (macro factor structure), alternating or cyclically projecting between them, and add a genuine low-rank tensor factorisation on top.
- **Rank is much larger than it first looks.** Runs that swept upward kept improving well past any obvious stopping point. Ensemble over a spread of ranks and regularisation strengths with validation weighting instead of picking one.
- **Model the residuals after the factor stage.** Stacked and each additive: an autoregressive temporal smoother over observed neighbours, a cross-characteristic ridge, a short-window then global cross-time ridge, and augmenting those regressions with the **missingness mask** — informative because firm histories differ enormously in coverage.
- **Calibrate the output scale.** Low-rank reconstructions are shrunk toward zero, which costs R² directly; refit a scale against observed entries, dilate the variance, and clip into range.
- Values are cross-sectional ranks rescaled to a fixed interval, so each `(month, characteristic)` slice has a known marginal shape — moment or quantile matching per slice is information a generic imputer ignores.
