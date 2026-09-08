# Hint

From prior autonomous-research runs on this task. Directions only, no tuned constants.

- **Separate support selection from coefficient fitting.** The winning structure is not a plain sparse autoencoder trained end to end: a learned top-k encoder decides *which* atoms are active, and the coefficients are then obtained by an exact **least-squares refit on the chosen support** at inference. Learned nonlinear support selection captures far more variance than any linear code of the same width, and the refit removes all the error the encoder's own coefficient estimates would contribute. Both halves matter.
- With the refit in place the remaining error is almost entirely **support error**. Iterative refinement paid off steadily: multiple passes examining the current residual and proposing atom swaps, scored by deletion cost (what does dropping each active atom cost after refitting?). **Narrow proposals beat wide ones** — offering a couple of candidate atoms per pass gave better-conditioned, more conservative swaps.
- **Split the refit regularisation.** Wide over-complete solves used during support *selection* want noticeably stronger regularisation than the final narrow reconstruction solve; separating that single constant into two was worth a real gain.
- **The sparsity constraint is on the mean, not per activation.** Allocating slightly more atoms to hard activations and fewer to easy ones, while hitting the mean exactly, is free score.
- Check the least-squares refit's runtime at the graded batch size — that is the part that scales, not the encoder.
