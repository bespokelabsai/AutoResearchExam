# Hint

From prior autonomous-research runs on this task. Directions only, no tuned constants.

- You never see the target, so the only signal about a level is **how the numeric covariates behave inside it**. Compute per-level centroids, denoise them, project to the allowed width. Everything good here refines that.
- A level's centroid has noise variance proportional to `1/count`. Whiten in the between-level eigen-coordinates so the noise is isotropic, then shrink.
- **The prior over level centroids is clustered, not Gaussian.** Replacing Gaussian empirical-Bayes shrinkage with a nonparametric mixture (NPMLE) posterior mean — atoms placed by clustering the high-count levels, weights by EM — was the single largest jump in several runs, and rare levels benefit most.
- **Smooth locally too:** pull each level's centroid toward the mean of its nearest neighbouring level centroids.
- A **nonlinear (polynomial-kernel) projection** of the denoised centroids beat plain linear PCA by a wide margin.
- `transform` receives a whole batch, which is unlabelled data about level frequencies — pooling train and test rows at transform time to refine level statistics gained repeatedly. For fitting-pool rows, exclude the row's own contribution or the encoding leaks.
- The graded pools are smaller than the development ones; adopt a change only when it wins the dev panel **and** a small-pool stress panel.
