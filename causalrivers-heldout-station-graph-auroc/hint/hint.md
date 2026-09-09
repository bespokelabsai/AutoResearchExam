# Hint

From prior autonomous-research runs on this task. Directions only, no tuned constants.

- Work on differenced / de-seasonalised log discharge and compute lagged cross-correlations; the **asymmetry** of the lead-lag profile is the core directional signal. Aggregate to multi-hour blocks before correlating — that beat raw sampling-interval lags clearly.
- **Absolute scale and missingness-rate features do not transfer** across gauges and regions. Drop them; use ranks and within-group standardised features, and pairwise-complete statistics for missing data.
- An edge means "flows onward with **no gauged station in between**", which no pairwise statistic can express. Add a conditional-independence term — partial correlation contrasted against marginal correlation. It is the only feature that separates a direct edge from a mediated one.
- **Decode the graph, don't score pairs independently.** The stations form a tree-like flow network: enumerate spanning arborescences and take edge marginals under a temperature-scaled softmax over trees. This was worth more than any additional feature.
- Layer region-invariant physics on the structure: **mass balance at confluences** (inflows cannot exceed the node's own discharge), the **river mouth carries the most water** so bias toward trees rooted at the highest-median-discharge station, and an **out-degree penalty**. These are exactly the constraints that survive unseen stations.
- In-region cross-validation badly overstates. Validate with station-disjoint and cross-region splits and adopt only changes that win on all of them.
