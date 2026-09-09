# Hint

From prior autonomous-research runs on this task. Directions only, no tuned constants.

- **Where you skip matters far more than how you reconstruct.** Skip damage is steeply front-loaded: early steps commit the global structure, late steps only refine, and consecutive late skips are nearly free. The uniform/alternating schedule everyone writes first is close to the worst legal allocation.
- The recipe that worked: an unbroken run of full evaluations at the start, every skip pushed into the tail, and the final step always exact. Moving from alternating to tail-loaded was the single largest jump in multiple runs — larger than any reconstruction improvement.
- Derive the placement empirically with a **single-skip sensitivity sweep** (skip exactly one step, measure the loss, repeat), then check pairs, since adjacent skips are not independent.
- **Reconstruct in clean-sample space**, not raw velocity/noise space — it is scheduler-aware and far more stable across a skip.
- **Holding the last cached value beat linear extrapolation**, which systematically overshoots; if you do extrapolate, the useful coefficient is small and was sometimes negative.
- Add hindsight error feedback: at each full evaluation, measure how far the cached prediction had drifted and correct later cached steps by that residual. The guidance residual also varies more slowly than either branch, so caching *it* while refreshing the conditional branch is a cheap alternative.
- Verify the schedule costs exactly the budget by construction and keep the fallback generic for other step-count/budget pairs.
