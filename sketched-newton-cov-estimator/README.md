# Estimate parameter uncertainty from a model’s optimization trajectory

*Category: Algorithms and optimization. Subcategory: Statistical methods.*

This task gives the agent one long run of a model learning from data and asks it to predict how much the final answer would change if the run were repeated with different sampled data. The agent may use the model parameters, gradients, step sizes, and curvature matrices to estimate how the final parameters would vary across different repeated runs. A simple starter approach utilizes a covariance estimate from the recorded gradients and the final average curvature matrix. Further gains may come from using changes in curvature over time and correcting for dependence between nearby updates.
