# Hint

From prior autonomous-research runs on this task. Directions only, no tuned constants.

- **Train against the metric.** Losses that optimise pairwise similarity in the abstract underperform ones that optimise ranking directly. A differentiable **average-precision surrogate over the Hamming distance histogram, restricted to the truncation window the metric uses**, was the largest single improvement in more than one run. Second best: supervised contrastive with **soft** targets from label overlap rather than a binary similar/dissimilar split.
- **Sixteen bits is tiny, so wasting any is expensive.** All three helped: a decorrelation penalty pushing the bit covariance toward identity, a bit-balance penalty so no bit is near-constant, and a quantisation penalty pulling pre-sign activations off zero.
- Two-stage construction — a continuous low-dimensional semantic embedding, then a learned rotation before taking signs — cuts quantisation loss further and is cheap.
- **Blend a k-nearest-neighbour vote** over training semantic codes with the parametric bit predictor: the local neighbourhood carries information the small global head cannot represent.
- L2-normalise the features; Gaussian input noise regularises the encoder.
- `encode` must be **permutation-equivariant and batch-independent** — it is checked, and any statistic computed over the batch breaks it.
- Training used a small fraction of the limit in every run; spend it on more epochs and seeds. The local proxy's noise floor is a few thousandths of the metric.
