# Hint

From prior autonomous-research runs on this task. Directions only, no tuned constants.

- **Plan offline, execute online, repair rather than replan.** Do the combinatorial work once before the episode — insertion heuristics scored by reward per unit of extra distance, improved by local search and randomised restarts; **beam search over insertion sequences clearly beat single-path greedy**. Online, persist that tour and repair it greedily against the realised budget instead of replanning from scratch. Keeping an **ensemble of plans** built under different assumed caps and selecting among them as the budget realises was another sizeable gain.
- The travel costs are exponential, so the probability of reaching the terminal vertex within the remaining budget has **closed form** — use it directly rather than by simulation, and use it for the continue-versus-go-home decision by backward induction.
- **The failure allowance is a resource to spend.** The decisive observation across several runs: graded failure rates were sitting well below the cap, meaning the policy was buying safety it was not charged for. Lowering the risk price until the realised failure rate sits just under the cap — with margin for the development-to-graded gap — was repeatedly the largest single improvement, often larger than any planning change. Calibrate on a full panel sweep, never one draw.
- **Endgame rule:** when the budget tightens, choose the next vertex by reward per unit of outgoing distance (accounting for getting home), not nearest-eligible.
- Evaluate over thousands of episodes across several draws, pooled — single-draw tuning of the risk parameter overfits, and that parameter is the one where overfitting costs a penalty rather than a little reward.
