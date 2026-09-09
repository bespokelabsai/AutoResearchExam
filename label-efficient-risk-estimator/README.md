# Estimating model performance without all labels

*Category: Evaluation, calibration, and robustness. Subcategory: Metric estimation.*

The task requires the agent to estimate the average loss of a target classifier on 2,000 test samples. It has the target classifier’s predicted probabilities for every sample but may request true labels for only up to 400 of them. The agent also has a cheaper classifier’s predictions for all 2,000 samples, which may help identify more difficult regions of the dataset. A basic approach selects examples randomly and averages their observed losses. Better approaches may select more informative examples and use the cheaper predictions to reduce noise without biasing the estimate.
