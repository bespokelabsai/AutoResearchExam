# Hint

From prior autonomous-research runs on this task. Directions only, no tuned constants.

- **Throughput is half the problem** — buy epochs first: the dataset resident on the GPU with augmentation done there, reduced-precision autocast, channels-last, a large batch, momentum SGD with Nesterov, no weight decay on normalisation and bias parameters, and a **time-based** one-cycle schedule so the run completes whatever the machine speed.
- **The attack schedule is the other half.** One-step training with a random start is cheapest but **collapses at a predictable epoch** — robust accuracy against a strong multi-step attack falls off a cliff while the training signal still looks fine. Find that epoch on a held-out split, then either stop there or switch to a small-step multi-step tail. A step-count curriculum that increases through the run beat any fixed count.
- Train at an epsilon somewhat **larger** than the evaluation epsilon. A margin-style objective beat plain cross-entropy.
- Keep an exponential moving average of the weights (or average the last several epoch states), **recalibrate batch-norm afterwards**, and select the checkpoint against a cheap multi-step attack on a held-out split rather than shipping the last epoch.
- **Clamp adversarial examples in pixel space, not normalised space** — getting this wrong produces a model that looks robust locally and collapses under the grader.
- The grader takes the latest *complete* checkpoint at the deadline: write atomically every epoch with time in reserve. Seed spread is a fraction of a point, so replicate before believing a small gain.
