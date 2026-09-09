# Hint

From prior autonomous-research runs on this task. Directions only, no tuned constants.

- **Derive the covariance, don't measure it.** The iterate error follows a linear recursion driven by the sketch and the gradient noise, so its stationary second moment satisfies a Lyapunov / discrete moment equation. Write that down and solve it. Replaying the optimiser to measure the spread is slower and far noisier.
- The sketch's projection moments have **closed form** — propagate them analytically rather than sampling.
- That leaves only two things to estimate from the stream: the data covariance and the gradient-noise scale.
- **Fit the covariance with its parametric structure**, by maximum likelihood on the minimal sufficient statistics. This has dramatically lower variance than a free sample covariance and was worth more than any other single change. Recover the noise variance from an exact streaming least-squares residual.
- A small analytic correction for the bias from applying the nonlinear sketch factor to an *estimated* covariance adds a little more.
- **Throughput is a statistical lever**: accuracy improves with records consumed, and the limit is enforced under multi-process contention. Accumulate sufficient statistics from preallocated blocks, and set the internal deadline by profiling under that same concurrency — a guard tuned solo truncates the stream under load.
- Micro-tuning a final scalar against the visible score chases the noise of one fixed seed set and will not transfer.
