# Hint

From prior autonomous-research runs on this task. Directions only, no tuned constants.

- Start from task arithmetic — difference vectors against the pretrained encoder, merged with sign resolution and trimming — but keep going, because three refinements each gained:
  - **Per-task coefficients, not one global scale.** The experts interfere very unequally; a couple of them (typically those with the narrowest datasets) need weighting down hard while others carry more.
  - **Depth-dependent scaling.** Amplifying the task vectors in later encoder blocks relative to early ones changed the specialisation/interference balance enough that the per-task coefficients had to be re-optimised afterwards — do the two jointly, not in sequence.
  - **Per-task, per-depth adjustments.** An expert's contribution can help in one band of layers and interfere in the next.
- **The step change: stop merging, start fitting.** Scalar merge weights can only trade interference around. By far the largest jump came from **jointly training a replacement for the last shared transformer block** (and the final normalisation) on top of the merge, using all eight datasets at once — extending it to the last two blocks gained again. This stays within the rules (still one frozen encoder, no per-dataset routing) and improves *every* domain simultaneously, which no reweighting achieved. If you have budget for one big idea, this is it.
- Averaging two independently trained replicas of that block, with the mixing weight chosen on untouched holdouts, gains a little more and reduces overfitting risk. Select coefficients on splits held out from the ones used to fit the block, and track per-dataset accuracies, not just the mean.
