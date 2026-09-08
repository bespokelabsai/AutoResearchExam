# Tight confidence intervals for prediction uncertainty estimation

*Category: Evaluation, calibration, and robustness. Subcategory: Metric estimation.*

The task provides labels and class probabilities from classifiers with 2 to 50 classes, and samples of 200 to 10,000 rows. The agent is tasked with returning a short confidence interval for _squared calibration error_ among the one to three most likely classes. This interval is a range for how closely the stated chances match the observed results, and the interval itself must include the _true_ error in at least some fraction of repeated samples. Sampling noise changes with the class count and the shape of the probabilities, so a narrow interval can miss the true error in some settings. One may start with a standard range based on the sample size and the observed error. Further gains may come from better estimates of sampling noise and ranges that use the most likely probabilities.
