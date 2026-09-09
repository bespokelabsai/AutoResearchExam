# Hint

From prior autonomous-research runs on this task. Directions only, no tuned constants.

- **Generic deep RL cannot win this.** The metric is *reliable* success — above the bar at every one of the last several checkpoints, on every seed, within a small interaction and CPU budget. PPO, TD3, ARS, CEM and evolution strategies were each implemented carefully in different runs and all plateaued around a third to a half of the score, because a stochastic learner occasionally has a bad seed and a bad seed scores zero.
- Every run that reached the top switched to **system identification plus classical control**:
  1. **Classify the family** from observation dimension / empirical rank plus cheap probing — this can be made essentially error-free.
  2. **Identify the latent system** by regressing dynamics from a short probing rollout: latent coordinates, physical constants, actuator sign and scale, and any action delay.
  3. **Apply the matching controller** — linear-quadratic for the near-linear families, energy shaping plus a linear controller for swing-up, analytic inverse kinematics with a joint controller for reaching, partial feedback linearisation for cart-pole.
- **Make the identification invariant to the hidden transform** — use quantities invariant under the unknown rotation, shift and scale. Several runs had a controller that worked on public draws and failed on hidden ones purely because a sign convention was baked in.
- **Report early and freeze.** Since the last several checkpoints must *all* pass, converge fast, report a fixed policy before the first checkpoint, and stop changing it — gated on a validation score, falling back to a generic learner when identification validates poorly.
- Identification costs seconds, leaving most of the CPU budget unused. Validate each family on many fresh parameter draws, not just the public seeds.
