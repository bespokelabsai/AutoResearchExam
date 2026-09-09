# Unsupervised representation learning for tabular data

*Category: Model training. Subcategory: Representation learning.*

This is an unsupervised representation learning problem. The task uses regression tables with 10 numeric columns and one categorical column with 1,600 possible values. The agent must encode each category with at most eight numbers, without labels, before a fixed random forest predicts the target. Rare and unseen categories provide little evidence, and the data do not state which categories should share a representation. One may start with category frequencies and a shared value for categories absent from the fitting data. Further gains may come from grouping categories through their numeric features and smoothing estimates for rare categories.
