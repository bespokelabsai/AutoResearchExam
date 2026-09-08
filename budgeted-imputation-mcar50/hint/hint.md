# Hint

From prior autonomous-research runs on this task. Directions only, no tuned constants.

- **Two stages, blended.** A joint-Gaussian conditional mean (covariance by EM or pairwise-complete estimation, repaired to positive-definite, plus shrinkage) is a strong low-variance base; a per-column nonlinear model with native `NaN` handling, fitted on its **residuals**, adds the rest.
- Combine them with **per-feature** weights estimated on held-out entries, not one global weight.
- **Use the evaluation matrix.** Both matrices come from the same distribution under MCAR, so the eval rows are legitimate unlabelled data for the covariance. Fitting transductively on the stacked matrices is a real gain.
- **Watch for leakage in the blend calibration:** if calibration rows already had the target imputed by the same models, the weight is biased toward the flexible one. Build leakage-free "shadow" rows first.
- Once the covariance is repaired, an extra ridge on the Gaussian conditional solve only biases predictions toward zero — removing it was one of the larger single gains.
- Chained imputation helps when columns are ordered by aggregate absolute correlation, with damped carry-over between passes. Add missingness indicators as predictors.
- **The hidden datasets are wider than the development ones.** Profile a synthetic worst-case at maximum dimension; several runs were tuned in a regime the dev panel never reached, where rows-per-fit becomes the binding constraint.
