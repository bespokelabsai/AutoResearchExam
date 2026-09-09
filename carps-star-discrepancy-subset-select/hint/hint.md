# Hint

From prior autonomous-research runs on this task. Directions only, no tuned constants.

- **Exact star discrepancy is computable, not just approximable** — the supremum is attained on a finite grid built from the points' own coordinates. Build that evaluator first and make it fast (cumulative-count grids, in-place prefix accumulation, preallocated buffers, incremental update after a swap): everything else is a search that calls it, so its speed *is* the score. The same machinery yields the **critical boxes** that make the search guided rather than blind.
- **Two stages.** Initialise with the cheap closed-form L2 surrogate under greedy-plus-swap descent — after an accepted swap only the incoming point's kernel column needs recomputing, which bought a large fraction of the budget in one run. Then refine against the exact objective with one-point exchanges targeting the current critical boxes. Weighted variants of the L2 objective give genuinely complementary local optima.
- **Spend the budget on restarts, not one long descent.** At larger subset sizes a long tabu search almost never improves an L2 local optimum; best-of-many-starts with a patience rule, and iterated local search that perturbs the incumbent and re-descends, beat it consistently.
- Decide deliberately how to treat points on a boundary coordinate — blanket filtering was tried and **hurt** on skewed clouds, where those points reach the achievable floor.
- Always return a valid subset: a hard internal deadline with margin plus a fallback.
