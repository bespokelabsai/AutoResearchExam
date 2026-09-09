# Hint

From prior autonomous-research runs on this task. Directions only, no tuned constants.

- **The metric is balanced accuracy, so a class you never labelled costs a full share.** Drive label acquisition by *class need*, not uncertainty: refit after every small round, weight candidates by unmet need for their predicted class over expected price, and add an explicit novelty term for classes not yet seen. Runs that fixed rare-class coverage moved tens of points; runs that tuned the classifier first moved fractions.
- At prediction time, divide the posterior by a power of the class prior — the evaluation distribution is balanced and your acquired labels are not.
- **Train under the mask you will be tested under.** You buy features per case, so the test-time missingness pattern is a consequence of your own policy. Build augmented replicas whose masks follow that same distribution — some complete, some deterministic prefixes of your purchase order, some utility-weighted random. Tuning that mix was among the most productive late axes.
- **Regularise hard.** The labelled set is tiny and the augmented replicas are correlated; every run that swept found a long monotone gain from much stronger final-model regularisation than defaults suggest.
- Score features by utility over price raised to an exponent, blending a static prior with importances learned as labels arrive, and make sure the few dominant features actually get bought.
- **Never write to stdout or stderr** — the grader speaks a line protocol over it and a stray print zeroes the score. Time the policy at the real pool scale, not the development one.
