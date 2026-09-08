# Hint

From prior autonomous-research runs on this task. Directions only, no tuned constants.

- **This is not a generic CATE problem.** Meta-learners, doubly-robust estimators and tabular foundation models all plateau early. The data comes from a highly structured semi-synthetic generator, and every large jump came from identifying more of that structure: two simple parametric outcome surfaces that **share a coefficient vector**, whose entries come from a **small discrete grid with a large point mass at zero**, with a population quantity calibrated to a **known constant** and its own discrete law for the intercepts.
- Recover this from the data (fit the surfaces, see where estimated coefficients pile up, check the calibration identity numerically). The task then collapses from estimation to **discrete identification** — get the latent vector exactly right and that dataset's error is essentially zero.
- **Impose the calibration constraint exactly, not as a penalty.** Softening it leaves a systematic bias.
- Search the grid by profiled coordinate descent plus escape moves over pairs and triples, with iterated local search; independent rounding of continuous estimates leaves many datasets wrong. Keep a **smooth** likelihood for the search and use the **exact** one only to re-rank near-optimal candidates.
- **Match the aggregator to the metric.** The score averages a per-dataset RMSE — an L2 norm, not a squared error — so the right aggregate over posterior candidates is a weighted **geometric median**, not a mean.
