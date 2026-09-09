# A faster executor for transformer-decoder compute graphs

`/app/reference_executor.py` runs transformer-decoder compute graphs on CPU. Write an
executor that computes the same thing faster.

## Submission format

- Deliverable: `/app/output/executor.py`, plus any helper modules placed beside it in
  `/app/output/`.
- Only `/app/output/` is collected and graded. Everything else under `/app` is reference
  material. Grading happens in a separate environment, so your submission must be
  self-contained inside `/app/output/`. Other paths, running processes, and environment
  state do not carry over.
- It must expose exactly one entry point:

```python
build_executor(graph_spec: dict,
               weights: dict[str, numpy.ndarray],
               n_workers: int) -> Callable[[numpy.ndarray], numpy.ndarray]
```

- The returned callable takes `x` of shape `(T, d_model)`, dtype float32, and returns the
  final hidden states of shape `(T, d_model)`, dtype float32.
- It will be called many times with different inputs but the same weights.

## Data notes

- `graph_spec` describes a decoder stack. It contains `n_layers`, `d_model`, `d_ff`,
  `n_heads`, `n_kv_heads`, `head_dim`, `T`, `rms_eps`, a per-layer list of nodes, and a
  final node. Each node has an operation, named inputs, an output name, and a weight name.
- `weights` maps every weight name in the spec to a C-contiguous float32 array, plus
  `rope_cos`, `rope_sin` and `attn_mask`.
- `n_workers` is `8`.

`/app/reference_executor.py` defines what a correct forward pass is: its module docstring
states the semantics of every node type, and its code is the tie-breaker if anything is
ambiguous. Read it. `/app/graph_spec.py` builds instances and their weights from an
integer seed.

## Correctness

For every call, output `y` is compared against the reference's `y_ref` on the same input.
Both bounds must hold:

$$\max(|y - y_{ref}|) \le 10^{-3}$$

$$\frac{\lVert y - y_{ref} \rVert_F}{\lVert y_{ref} \rVert_F} \le 10^{-3}$$

- Reassociating a reduction (changing accumulation order) is fine; dropping work is not.
- A single instance that violates either bound scores the whole submission 0.

## How you are measured

- Your executor and the reference are built for the same instance in two separate
  processes, then timed head to head.
- The two are called alternately with a fresh input each repetition. Whichever one isn't
  being called is suspended so it can't consume cycles, and which one goes first
  alternates between repetitions.
- 3 warmup repetitions are discarded.
- Instance time = fastest of 15 timed repetitions.
- Instance speedup = `reference_time / your_time`.
- Score = geometric mean of the per-instance speedups over a sealed set of instances,
  built from seeds you've never seen, drawn from the ranges below.
- Maximize this geometric-mean speedup on the sealed set. An executor that just delegates
  to the reference scores zero; any larger speedup scores higher, with no ceiling where
  further improvement stops counting.

The sealed instances use the same generator and ranges as the public ones:

| | |
|---|---|
| `d_model` | 192, 256 or 384 |
| `d_ff / d_model` | 2.6875 or 4.0, rounded to a multiple of 32 |
| `n_layers` | 16 or 32 |
| `n_heads` | 4 or 8, `head_dim = d_model // n_heads` |
| `n_kv_heads` | `n_heads`, `n_heads // 2` or `n_heads // 4` |
| `T` | 1, 8 or 32, with 1 drawn half the time |

## Budget and environment

- `build_executor` must return within **20 seconds** per instance. Each call to the returned
  callable must finish within **120 seconds**. Time spent in `build_executor` is not included
  in the runtime metric.
- The container has **8 CPUs**, **14 GiB** of memory, **0 GPUs** and no network. Each
  executor process is pinned to 8 CPUs and starts with
  `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=NUMEXPR_NUM_THREADS=8`. The only
  third-party Python packages installed are `numpy` and `threadpoolctl`.
- Threads are yours to use, and so are helper processes, but the harness suspends your
  whole process tree whenever the reference is on the clock, and a process that detaches
  from that tree stays suspended for good.

## Developing

`python3 /app/bench.py` measures whatever is in `/app/output/` against the reference on
the 16 public instances (seeds 0-15). It uses the same protocol, tolerances, and worker
budget as the sealed run. It prints per-instance times, ratios, equivalence errors, and
the geometric mean. `python3 /app/bench.py --seeds 0,4,9` restricts it to those seeds.
Public instances are for development only; nothing about them is graded.

A submission scores 0 if any of the following holds:

- `/app/output/executor.py` is missing.
- It fails to import.
- It has the wrong signature.
- It raises.
- It exceeds a budget.
- It returns the wrong shape or dtype.
- It returns values that are not finite.
- It violates either equivalence bound on any instance.
