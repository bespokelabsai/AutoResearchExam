# Hint

From prior autonomous-research runs on this task. **Evidence is thin** — the one available run never completed a graded iteration, so this is where it got to, not a validated recipe.

- What it built: a surface-quality scorer per document (length and fragment filters, non-English and malformed-page detection, boilerplate and structured-data detection, token-set statistics) plus a mild **host-diversity** term, then a greedy pass spending the budget in score order.
- **The failure mode it had already diagnosed:** surface quality pushes the obvious garbage down — link farms, tiny fragments, machine-generated listings — but it also ranks **SEO and syndicated commercial prose highly**, because that text is clean, long and grammatical while being nearly worthless for language modelling. Any purely lexical quality heuristic will have this failure.
- The direction it identified but never reached: score by a **model-based** criterion instead — loss under a small pretrained language model, or similarity to a high-quality reference distribution in embedding space — which separates "well-formed" from "informative" in a way surface statistics cannot.
- Cost is dominated by JSON decoding and per-character Python work across hundreds of thousands of documents, not by the scoring itself; bounding the scored text sample barely helps. Even so a full pass finished in a few minutes against a budget of tens of minutes, so **an expensive scorer is affordable**.
- The comparison is against training on the whole pool, so the metric can go negative — establishing that a plain quality filter clears zero is worth one early iteration.
