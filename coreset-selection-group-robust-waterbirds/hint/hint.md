# Hint

From prior autonomous-research runs on this task. Directions only, no tuned constants.

- **The hidden attribute is recoverable without labels for it**, because its correlation with the class is what makes it findable. Fit a class predictor: the low-margin / margin-conflicting tail is enriched for the minority group where class and attribute disagree. Contrasting the two class-conditional margin tails estimates a **nuisance direction**, and clustering within each class gives latent groups directly.
- Whichever you use, the score follows from **balancing the budget across the inferred groups** — pushing the inferred rare group up toward half of each class's share was the change that kept paying.
- **Mix hard and representative.** Pure hardness collapses (the selection stops being representative and the downstream classifier degrades); pure representativeness never breaks the spurious correlation. Take a block of low-margin examples plus a block spread across the margin range, per class, with near-centroid picks inside each stratum.
- **Stability matters as much as the criterion.** The pool is redrawn per run and the downstream fit is small-data. Blend a frozen development-trained scorer with the pool-specific one, prefer a **sparse L1 logistic** proxy (it suppresses nuisance dimensions and scores more symmetrically than a dense fit), and fix the solver's random state.
- Evaluate over all development draws crossed with several downstream seeds — single-draw differences on a worst-slice metric are almost pure noise, and several runs shipped regressions by trusting one.
