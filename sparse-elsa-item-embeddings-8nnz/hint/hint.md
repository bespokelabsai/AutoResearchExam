# Hint

From prior autonomous-research runs on this task. Directions only, no tuned constants.

- **Start from a dense teacher**: the closed-form item-item ridge on the interaction Gram matrix with the diagonal constrained away. That matrix is the target behaviour and its performance is your ceiling — useful to know early.
- **The realisation trick:** set the embedding width to the number of items and use each dimension as a shared channel between one *pair* of items. A nonzero pair then costs one slot in each row, and the per-row budget becomes a **capacity-constrained edge orientation** problem, with one slot reserved for the row's self-weight.
- **Support selection is where the remaining score is, not values.** Once values are refit on a fixed support, further value tuning gives very little. Every sustained climb came from better support: score candidate edges by teacher importance weighted by the value the edge would carry, run a capacity-aware greedy orientation, then **iterate** — refit, re-score, re-select over several rounds. Replacing the greedy with an **exact augmenting-path** assignment gained again, because the greedy silently rejects feasible high-priority pairs.
- **The self-weight is a free per-row parameter** — under row normalisation it sets that row's shrinkage. Making it a function of the row's fitted neighbour magnitudes rather than a constant was a clean gain.
- **Match the evaluation's masking.** Scoring feeds a masked fraction of each row as input, so the second-moment matrix the model faces is a specific reweighting of the Gram matrix with an inflated diagonal — fit the teacher against *that*.
- Refitting values with a **listwise ranking objective** on the fixed support beat squared error.
