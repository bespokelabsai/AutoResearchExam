# Hint

From prior autonomous-research runs on this task (one run, thinner evidence than most, but a clean monotone climb). Directions only.

- The gradient entry per candidate is the calibrated starting point and worth submitting first to establish scale — but on its own its rank correlation with true loss reduction is weak, which is exactly what the task says.
- The change that moved the score by a factor of five: **learn a per-token prior from generated data**. Substitution quality depends heavily on *which token* is swapped in, largely independently of the current prompt — some tokens are systematically good replacements. Use the development generator to produce snapshots with true rankings, fit a smoothed per-token-id rank prior, and combine it with the gradient score at a small weight, leaving the gradient dominant.
- **The rest of the climb was purely scaling that data.** Every subsequent gain came from generating more behaviour trajectories and refitting the same prior unchanged, roughly doubling the count each iteration. The mechanism is coverage: the fraction of graded candidates whose token was never observed falls steeply and the prior stops backing off. **Track that unseen-token rate — it predicts the score better than development rank correlation does.**
- Freeze the inference formula and blend weight while scaling the data, so each grading tells you about one thing.
- Group validation **by behaviour**, not by snapshot; snapshots from one trajectory are highly correlated and pooled correlation flatters you.
