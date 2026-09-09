# Spend a pretraining token budget

Build a selector that chooses which web documents to use for language model pretraining under a fixed token budget.

## Goal

The document pool contains about 250 million GPT-2 BPE tokens. Your selector may spend 125,000,000 tokens.

The grader trains the same 124M parameter decoder-only transformer from scratch on your selected documents and on the full pool. Both runs use fixed training settings. It then measures perplexity on a hidden evaluation shard.

Your metric is the percent reduction in hidden perplexity compared with training on the full pool:

$$
100 \times \frac{\mathrm{PPL}_{\mathrm{full}}-\mathrm{PPL}_{\mathrm{selected}}}{\mathrm{PPL}_{\mathrm{full}}}.
$$

Higher is better. The metric can be negative. An invalid submission receives zero reward.

A selector that is no better than random earns very little. A selector that buys no documents earns zero.

## Deliverable

Write `/app/output/select.py`. It must define:

```python
def select(pool_dir: str, budget_tokens: int, workdir: str) -> list[int]:
    ...
```

At grading time, `budget_tokens` is `125000000`. `pool_dir` contains the document pool. `workdir` is an empty writable directory for temporary files.

Return document IDs in priority order, with the most useful document first. Return a Python list or tuple of integer IDs. IDs must be unique and between `0` and `N - 1`. A NumPy integer is accepted. A Boolean is not.

The grader walks the list in order. It accepts a document if it fits within the remaining budget and skips it otherwise. Repeated, invalid, and out-of-range IDs are ignored. Spending less than the full budget is allowed, but the accepted documents must contain at least 1,024 tokens.

## Pool format

`pool_dir` contains:

- `docs.jsonl`: one document per line, with `id`, `text`, `url`, and `ntokens`.
- `tokens.npy`: a `uint16` array containing all token IDs in document order.
- `offsets.npy`: an `int64` array of length `N + 1`. Document `i` uses `tokens[offsets[i]:offsets[i+1]]`.

Document IDs differ between the development and hidden pools. Submit a selection method, not a saved list of IDs.

## Files and limits

Only files under `/app/output` are submitted. You may include modules, weights, vocabularies, or lookup tables there. Do not use symbolic links or hard links. Keep the directory below 200,000 files.

The grader copies this directory before importing `select.py`. Locate bundled files relative to `__file__`, not by hardcoding `/app/output`.

Importing the module and running `select()` share a 2,400 second limit. A crash, timeout, invalid return value, failed training run, or non-finite metric receives zero reward.

The grading machine has one NVIDIA H100 80 GB GPU, 8 physical CPU cores, 32 GiB RAM, and a 300 GB disk. The selector has no internet access. Write temporary data only to `workdir`.

Installed packages include PyTorch, Transformers, NumPy, SciPy, scikit-learn, and Tokenizers. Offline model snapshots are available at `/opt/assets/gpt2` and `/opt/assets/pythia-160m`.

## Development files

Use these files to test your selector:

- `/app/data/dev_pool`: a development pool with the same format.
- `/app/data/dev_eval/tokens.npy`: a separate development evaluation shard.
- `/app/harness/pretrain_harness.py`: the fixed trainer and evaluator.
- `/app/harness/train_config.json`: the fixed model and training settings.
- `/app/harness/dev_cycle.py`: runs a complete development cycle. Use `--reference` once to measure the full-pool baseline.

The hidden pool and evaluation shard use different corpus shards from the development data.
