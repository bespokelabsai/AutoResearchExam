# Sparse coding of GPT-2 activations

Build a sparse dictionary that reconstructs GPT-2 small layer 8 residual stream activations.

## Goal

The grader measures normalized reconstruction error on 1,048,576 hidden activations:

$$
\frac{\sum_i \lVert x_i-\hat{x}_i\rVert^2}
{\sum_i \lVert x_i-\bar{x}\rVert^2}.
$$

Lower is better. The numerator and denominator are pooled across the full hidden set before division.

Reconstructing every activation with one fixed vector earns zero reward. A method no better than the best 32 dimensional linear code earns very little.

The average number of active dictionary entries may not exceed 32 per activation. The dictionary may contain at most 8,192 entries.

## Deliverable

Create these two files:

- `/app/output/dictionary.pt`
- `/app/output/encoder.py`

Only files under `/app/output` are submitted.

### Dictionary

Save `dictionary.pt` with `torch.save`. It must be a plain dictionary of tensors with:

- `W_dec`: a finite `float32` CPU tensor with shape `[768, n]`, where `1 <= n <= 8192`.
- `b_pre`: a finite `float32` CPU tensor with shape `[768]`.

Column `j` of `W_dec` is dictionary entry `j`. The grader reconstructs each activation as:

$$
\hat{x}=b_{\mathrm{pre}}+\sum_j \mathrm{val}_j W_{\mathrm{dec}}[:,\mathrm{idx}_j].
$$

### Encoder

`encoder.py` must define:

```python
def load(device: str) -> object:
    ...
```

The returned object must define:

```python
def encode(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```

The grader calls `load("cuda")` once. Each `encode` call receives a `float32` tensor with shape `[8192, 768]` on the GPU.

Return `(idx, val)`:

- `idx` has shape `[8192, K]`, integer dtype, and values from `0` to `n - 1`.
- `val` has the same shape, floating point dtype, and finite values.
- `K` may be between 0 and 8,192 and may change between calls.
- Entries with value exactly zero are inactive and do not count toward the sparsity budget.
- A row may not name the same dictionary entry twice.

## Data

`/app/extract_acts.py` shows how activations are created. It splits documents into non-overlapping 64 token windows, runs GPT-2 small in float32, reads the residual stream after block 8, centers each token across its 768 channels, and normalizes it to unit length.

Development data is available at:

- `/app/data/openwebtext/plain_text/train-00000-of-00080.parquet`
- `/app/data/openwebtext/plain_text/train-00001-of-00080.parquet`

The hidden activations use a different OpenWebText shard and the same extraction code.

## Grading and limits

The grader calls `encode` 128 times, in fixed order, for 1,048,576 activations total. It reconstructs in float32 and accumulates the error sums in float64.

Importing `encoder.py`, loading the encoder, and all encoding calls share a 900 second limit. The whole grading run has a 1,500 second limit.

`/app/output` may contain at most 20,000 files and directories. It may not contain symbolic links, hard links, or special files. Load bundled files relative to `encoder.py`, since the grader copies the output directory before running it.

The grading machine has one NVIDIA H100 80 GB GPU, 4 physical CPU cores, 32 GiB RAM, and about 300 GB disk. The process has no internet access and runs with one CPU thread. PyTorch, Transformers, NumPy, SciPy, and PyArrow are installed. The pinned GPT-2 checkpoint is available at `/opt/assets/gpt2`.

Any malformed output, exception, timeout, non-finite value, file contract violation, or resource limit violation receives zero reward.
