# Hint

From prior autonomous-research runs on this task. Directions only, no tuned constants.

- **Fit the low-rank branch, don't take it once.** Round-to-nearest on the weights followed by an SVD of the remainder is the obvious construction and it is not the good one. **Alternate**: quantise the weight minus the low-rank term, recompute the low-rank term from the new residual, repeat a few times. A handful of passes roughly halved the weight reconstruction error against a single-shot decomposition and produced a large image-quality gain.
- **Smoothing between activations and weights is the other half.** Under 4-bit activations, per-channel activation outliers dominate the error. Migrate difficulty into the weights with a per-channel scaling derived from calibration statistics — a geometric blend of activation magnitude statistics and per-column weight extremes, normalised per quantisation group, with conservative clamping.
- **Use all the calibration prompts you are allowed.** Refitting the same smoothing statistics on the full permitted set, changing nothing else, transferred cleanly to unseen prompts. Statistics from a small sample are what overfits here.
- **Fix the deliverable contract before optimising quality.** Runs lost whole iterations to scoring zero on structural checks — a stray bytecode directory in the submission path, an extra registered tensor — with a quantisation implementation that was already fine.
- Evaluate on prompts outside the calibration set: decoded-image error on the calibration prompts moves the wrong way once the smoothing starts fitting them.
