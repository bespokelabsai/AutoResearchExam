# Hint

From prior autonomous-research runs on this task. Directions only, no tuned constants.

- **Pool hygiene is the biggest lever, not acquisition theory.** The pool contains out-of-domain text, malformed and truncated documents. Runs that spent their effort on uncertainty heuristics gained little; runs that first built an in-domain / cleanliness detector and pruned to genuine, well-formed examples jumped several points immediately.
- Equally cheap: look at what distinguishes the **evaluation** documents (formatting artefacts, length, markup) and bias the retained subset toward that distribution.
- **Ship a small pre-trained prior model with the pruner.** It gives pseudo-labels so round 0 — where there is no model yet and acquisition is effectively random — can return a class-balanced, high-confidence set. Blending a frozen development-trained scorer with one refit on the current pool beat either alone: stability from the first, adaptation from the second.
- **Schedule the candidate set across rounds:** narrow, confident, class-balanced early while the loop's own model is weak; wider later so uncertainty sampling has diversity; and actively exclude the most ambiguous documents late, so labels are not burned on unlearnable examples.
- Spread confident picks across a range of confidence ranks or cluster medoids — pure top-k collapses diversity and lost points every time.
- Validate with leakage-controlled leave-one-pool-out replays over many seeds; the run-to-run spread swallows a real gain.
