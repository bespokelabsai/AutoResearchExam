# Hint

From prior autonomous-research runs on this task. Directions only, no tuned constants.

- Start from correct GAE with a learned critic, correct **termination-versus-truncation** bootstrapping, and per-batch advantage standardisation — then stop tuning the GAE constants. Sweeping them yields very little; several runs spent a dozen iterations there for a fraction of a point.
- **The critic is the bottleneck, not the advantage formula.** The per-iteration rollout is small and fitting a value function on it alone badly underfits. Each of these was individually a large jump: a **FIFO replay of recent iterations'** observations and return targets refit over the pooled buffer; real critic capacity (wider than the reflexive small MLP) with several epochs and a long warm-up fit on the first iteration; and **normalising the value targets** with running statistics, since returns here span a huge range.
- **The real failure mode is seed variance, not mean performance.** Runs bifurcate — some seeds find the fast gait, others settle into a low-return attractor — and the score is a mean over hidden seeds. So the highest-value change is anything that raises the *fraction of seeds that escape*: a small **annealed** exploration term folded into the advantage did exactly that, and was the second-largest gain after the critic work.
- Note the interface is unconstrained — an arbitrary array goes straight into the policy loss, so the advantage channel can carry shaping signals other than a value baseline.
- Single-seed comparisons are worthless; use exactly-paired multi-seed batches and report the escape count alongside the mean.
