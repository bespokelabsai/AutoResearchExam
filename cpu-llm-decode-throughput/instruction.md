# Serve a concurrent generation workload faster

Build a generation engine that serves the concurrent workload faster while returning the same
greedy token ids as the reference engine.

`/app/reference/reference_engine.py` serves generation requests one at a time on a 12-layer
causal language model (GPT-2 124M geometry, fp32). Its weights and `config.json` are at
`/opt/model`.

## Submission

- Write your engine to `/app/output/engine.py`, plus any module it imports.
- `/app/output/` is put on `sys.path` before your module is loaded.
- Your module must expose:

```python
class Engine:
    def __init__(self, model_dir: str) -> None: ...
    def generate(self, requests: list[dict]) -> list[list[int]]: ...
```

## Request format

`requests` is the whole workload, handed over in one call. Each element is a dict with:

| key | type | range |
| --- | --- | --- |
| `id` | `int` | unique within the call |
| `prompt_token_ids` | `list[int]` | 16 to 160 ids, each in $[0, 50257)$ |
| `max_new_tokens` | `int` | 4 to 64 |

## Output format

- `generate` returns a list the same length as `requests`, in the same order.
- Element `i` is a `list[int]` of exactly `requests[i]["max_new_tokens"]` token ids.
- Element `i` is the greedy continuation of prompt `i`.
- No early stop, no padding, nothing extra.

## Metric

Your score is the generated-token throughput speedup $S$ of your engine over the reference
engine, measured on held-out request sets you never see.

- There are three timed rounds. Each round the grader draws a different held-out workload of
  128 requests (same distributions as above: prompt lengths from $\{16, 32, 48, 64, 96, 160\}$,
  new-token counts from $\{4, 8, 16, 24, 40, 64\}$), runs the reference engine on it, then runs
  your engine on the same workload.
- Throughput of a run: total requested new tokens divided by wall-clock seconds inside that
  run's `generate()` call.
- $S$ is the median, over the three rounds, of your throughput divided by the reference's
  throughput for that round.
- Submitting the reference engine unchanged scores nothing. Every improvement over it scores
  strictly more, with no ceiling, so push $S$ as high as you can.

## How a run is measured

Each timed run is a fresh process, pinned to the machine's 8 cores, running as an unprivileged
user from a read-only copy of your deliverable. In that process the harness:

1. imports your module and calls `Engine(model_dir)` with `model_dir = "/opt/model"`,
2. calls `generate` once on a small warm-up workload,
3. starts the clock and hands your engine the graded workload,
4. stops the clock when the process exits, after writing out your returned token ids.

- Steps 1 and 2 are untimed and share a 180-second budget.
- Before your module is imported, the harness sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and
  `OPENBLAS_NUM_THREADS` to the core count, and calls `torch.set_num_threads` with it. Your
  engine may change any of that.
- Your engine cannot write into the copy it is running from.
- `$HOME`, `$TMPDIR`, and the working directory all point at one empty scratch directory that
  is destroyed after the run.
- The graded workload does not exist anywhere your process can read until the clock is
  already running.
- Each round's workload differs from every other round's.

## What scores nothing

- No `/app/output/engine.py`, an empty one, or one without a working `Engine`.
- A symlink, hard link, or special file anywhere under `/app/output`, or more than 4 GiB there.
- An import error, an exception, a return value of the wrong type, length, or element length,
  or a token id outside $[0, 50257)$.
- Construction plus warm-up taking more than 180 seconds.
- A timed run taking more than twice the reference engine's wall-clock on the same workload,
  with a minimum limit of 60 seconds. The run is killed when it reaches this limit.
- Token ids that agree with the reference engine's at a rate below **0.99**, counted position
  by position over every request of every round.

## Environment

- Hardware: 8 CPU cores, 14 GiB RAM, no GPU, no network.
- Packages installed: `torch` (CPU build), `numpy`, `safetensors`.
- `/app/output/` is what gets graded; it's the only thing carried over to the grading run.
- `/app/reference/`, `/app/harness/`, and `/app/data/` are inputs. The grading run reads the
  reference engine, harness, and workloads from its own copies, so changes you make to yours
  only affect what your local numbers mean.
- `/opt/model/` is read-only for everyone.

## Dev workloads

`/app/data/` holds four dev workloads drawn the same way as the held-out ones.

```
python3 /app/harness/bench.py
python3 /app/harness/bench.py --draws all
python3 /app/harness/bench.py --draws 2 --limit 32
```

- This measures `/app/output/engine.py` against the reference on the dev workloads, using the
  same protocol, timing boundary, and agreement rule as the grader.
- Per-round numbers vary between draws; the grader's do too.
