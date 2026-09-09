# Rank token substitutions by true loss reduction

A prompt optimizer selects 64 possible token replacements using the gradient of its target loss. It would normally run the model once for every candidate to find which replacements help.

Build a ranker that predicts the same ordering without running the model at grade time. For each optimization snapshot, your ranker receives the gradient, candidate tokens, prompt tokens, current loss, and embedding matrix. It returns one score for each candidate.

## Deliverable

Write `/app/output/ranker.py`. It must define:

```python
def rank(features: dict) -> numpy.ndarray:
    ...
```

The grader imports `ranker.py` as the module `ranker` and calls `rank(features)` once per snapshot. You may add helper modules, weights, tables, or configuration files under `/app/output/`.

Only `/app/output/` is copied to the grading environment. Everything your ranker needs must be inside that directory. Use regular files and directories. Do not use symbolic links, hard links, or special files.

## Input

`features` is a new dictionary for each snapshot with these keys:

* `grad_row`: A `float32` array with shape `(32000,)`. Entry `j` is the gradient for vocabulary token `j`.
* `candidate_ids`: An `int64` array with shape `(64,)`. The ids are distinct, sorted, and drawn from the printable ASCII token set.
* `current_token_id`: The current token id as an integer from `0` to `31999`.
* `current_loss`: The current positive finite target loss.
* `prefix_ids`: A nonempty list of token ids before the position being changed.
* `post_ids`: A nonempty list of token ids between that position and the target.
* `target_ids`: A nonempty list of target token ids.
* `step`: The optimizer step. It is one of `10, 20, ..., 100`.
* `embedding_matrix`: A read only `float32` array with shape `(32000, 4096)`.

## Output

Return 64 finite real numbers with shape `(64,)`. A NumPy array, list, or tuple is accepted.

Output position `i` must score `features["candidate_ids"][i]`. Do not return token ids, indices, or a sorted array. Lower scores mean that a candidate is predicted to reduce the loss more. Only the ordering matters. Ties are resolved by the original candidate order.

For candidate `i`, the true value is:

```text
loss after replacing the current token with candidate i minus current_loss
```

A negative value means the replacement helps.

## Scoring

The grader compares your predicted ordering with the true loss change ordering on unseen optimization snapshots. The metric is pooled rank concordance correlation, or `pooled_rank_ccc`. Higher is better.

Leaving candidates in their given order defines the zero reward baseline. Ranking candidates by their value in `grad_row` earns a small positive reward. Better rank agreement earns a higher reward, with no upper cutoff where improvements stop counting.

If one call raises an exception or returns an invalid value, that snapshot uses the given candidate order. A missing entry point, invalid submitted tree, process crash, or timeout scores zero for the full run.

## Runtime

The full grading run imports your module and calls `rank` 80 times. It must finish within 300 seconds. The grader uses one CPU thread, 32 GiB of memory, no GPU, and no network.

The model weights are not available at grade time. The embedding matrix is passed in `features`. The Vicuna tokenizer is available at `/opt/assets/vicuna-tokenizer/` if needed.

## Development files

You may use these files while building the ranker:

* `/app/gcg_states.py` creates labeled development snapshots.
* `/app/data/dev_behaviors.csv` contains 100 development behaviors.
* `/app/data/token_allowlist.npy` contains the allowed printable ASCII token ids.
* `/app/example_ranker.py` is a minimal working example.
* `/app/work/` is scratch space.

These files do not move to the grading environment. Copy anything your ranker needs into `/app/output/`.

The development environment has one H100 GPU, 4 physical CPU cores, 32 GiB of memory, and no general internet access. NumPy, SciPy, scikit learn, PyTorch, and Transformers are installed.
