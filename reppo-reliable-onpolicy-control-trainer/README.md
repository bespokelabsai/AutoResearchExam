# Training reliable control policies under tight budgets

*Category: Model training. Subcategory: Reinforcement learning.*

The agent must train a controller from scratch for each of six unseen physical-control tasks, such as balancing a pendulum or moving a robotic arm, using at most 400,000 environment interactions and 45 CPU-seconds. Every task presents eight observation values and accepts two continuous actions, but hidden rotations and permutations prevent the trainer from relying on fixed meanings for those coordinates. The agent controls the entire training procedure and periodically reports the deterministic policy that the evaluator should test. It may use stable on-policy reinforcement learning, improved value estimates, system identification followed by an analytic controller, or safeguards that preserve the best policy found so far. Success requires more than briefly reaching a high return: the policy must achieve at least 90% of a task-specific reference level at each of the final ten checkpoints. Evaluation covers 72 independent runs across six environments and twelve hidden seeds.

In v1.1, half the score is the fraction of runs meeting this reliability condition. Half is the mean improvement in raw return from the passing threshold toward zero cost, averaged over the final ten checkpoints of all 72 runs. Missing checkpoints contribute zero. Returns above the reference continue to earn credit. Zero cost is a theoretical upper bound and may not be achievable. See [the scoring formula](instruction.md#how-a-run-is-scored).

## Benchmark integration

The task returns the composite score directly as both `metric` and `reward`. It records `reliable_success_fraction` and `mean_performance_improvement` separately, along with `scoring_version`. Raw checkpoint returns are saved in `/logs/verifier/panel_results.json`, keeping `metric.json` within the harness's 64 KB limit.

The benchmark must apply a v1.1 difficulty map to the raw composite before computing AUARC. Establish a new reference anchor; the old `0.8` anchor described reliability alone. Preserve the v1.0 map for released results. The AUARC integration and best-visible-checkpoint selection rules stay the same, but the harness must select the correct map by version. The current harness resolves maps by task name alone, so do not use its v1.0 map for this score.

Before a v1.1 leaderboard release, check strong saved solutions for useful remaining performance differences, establish the mapping reference and run fresh evaluations under the new feedback. Historical reliable-success fractions cannot reconstruct the new score.

## Scoring tests

From this task's directory, run `python3 -m unittest discover -s tests -p test_grader_core.py`. These tests check scoring without running candidate training or requiring NumPy or Torch.
