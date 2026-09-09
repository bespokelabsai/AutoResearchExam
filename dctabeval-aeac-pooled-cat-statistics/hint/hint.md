# Hint

From prior autonomous-research runs on this task. Directions only, no tuned constants.

- **Use the unlabelled data.** Counting over train, pool and eval together is the richest feature family: occurrence counts of singles, pairs and triples; **directional conditional-frequency ratios** (a pair's count over each member's, which expose the hierarchy among identifier columns in a way raw counts do not); per-value distinct-counts; pointwise-mutual-information scores; and log-bucketed versions of all of them.
- **Two model families, blended on the rank scale** (the metric is AUROC): a regularised sparse logistic regression over explicit one-hot singles, pairs and frequency-cut triples, and gradient boosting over the frequency features with several seeds averaged. Averaging several regularisation strengths and both balanced and unbalanced fits hedges the choice and cuts variance.
- Target-derived features help — out-of-fold target encoding with several smoothing strengths averaged, and pairwise target log-odds — but **cross-fit everything that touches the label**, and **down-weight pair terms involving near-duplicate columns**, which otherwise double-count evidence.
- Diffusing label signal a few hops through the value co-occurrence graph built from the unlabelled pool is genuinely orthogonal to the flat models and blended in for a real gain.
- Per-call runtime at graded geometry is much larger than on development draws — time it there before committing to an ensemble size.
