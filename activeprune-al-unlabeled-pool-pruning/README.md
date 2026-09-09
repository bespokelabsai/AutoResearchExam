# Selecting Useful Documents for Active Learning

*Category: Data engineering and curation. Subcategory: Data selection.*

The agent must help train a positive-versus-negative sentiment classifier using labels for at most 625 of 12,500 documents. Before each of five rounds, the agent narrows the remaining documents to at most 3,125 candidates but does not choose which 125 receive labels. In the first round, the system selects 125 candidates randomly; later, it selects those the current classifier finds most uncertain, meaning its positive and negative predictions are closest to 50–50. The agent can remove irrelevant or repetitive documents while preserving useful, varied examples, adapting its filtering as labels arrive. It is evaluated on separate unseen documents using macro-F1, which gives equal importance to performance on positive and negative sentiment.
