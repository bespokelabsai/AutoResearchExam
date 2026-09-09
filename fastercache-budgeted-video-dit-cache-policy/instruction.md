# Reduce video diffusion transformer evaluations

Video diffusion models are slow because they run a large transformer many times while refining
each video. CogVideoX 2b uses 50 diffusion steps and evaluates its transformer twice at each step,
once for the conditional prompt and once for the unconditional prompt. An uncached run therefore
uses 100 transformer evaluations.

Build a caching policy that reduces transformer evaluations by at least 38%, from 100 to no more
than 62. Within this budget, keep the final video as close as possible to the uncached result.

## Deliverable

Write `/app/output/policy.py`. It must define:

```python
class CachePolicy:
    def __init__(self, num_steps: int, budget: int, device: torch.device,
                 dtype: torch.dtype): ...

    def decide(self, step: int, state: dict) -> str: ...

    def reconstruct(self, step: int, mode: str, computed: dict,
                    state: dict) -> tuple: ...
```

The grader creates a new `CachePolicy` for each prompt with `num_steps=50`, `budget=62`, the visible H100 device, and `torch.bfloat16`.

Only `/app/output/` moves to the grading environment. You may add helper modules, weights, tables, or configuration files there. Use regular files and directories. The full directory must be no larger than 256 MiB.

## Choose transformer evaluations

The grader calls `decide(step, state)` once at each step. Return one of:

* `"full"`: Evaluate both conditional and unconditional predictions. Cost 2.
* `"cond_only"`: Evaluate only the conditional prediction. Cost 1.
* `"skip"`: Evaluate neither prediction. Cost 0.

The total cost over 50 steps must not exceed 62.

After any requested evaluations, the grader calls `reconstruct(step, mode, computed, state)`. Return `(eps_cond, eps_uncond)` in that order. Values already present in `computed` are used directly, so the matching return item may be `None`. Any missing prediction must be a finite `torch.bfloat16` tensor on the given device with shape `(1, 13, 16, 60, 90)`.

## State

Both methods receive a dictionary with:

* `step`: Current step from `0` to `49`.
* `num_steps`: Always `50`.
* `timestep`: Current scheduler timestep.
* `prev_timestep`: Scheduler timestep for the next step.
* `alpha_cumprod`: Current cumulative alpha.
* `alpha_cumprod_prev`: Cumulative alpha for the next step.
* `units_spent`: Budget already used.
* `units_remaining`: Budget left.
* `steps_remaining`: Steps left including the current step.
* `latents`: Current `torch.bfloat16` latent with shape `(1, 13, 16, 60, 90)`.

`computed` contains the predictions the harness evaluated. It has both `cond` and `uncond` for `"full"`, only `cond` for `"cond_only"`, and is empty for `"skip"`.

## Scoring

The grader runs your policy on unseen prompts and compares its videos with uncached 50 step videos generated from the same starting latent and seed. The score is mean PSNR in dB. Higher is better.

The zero reward baseline spends the budget on full evaluations at the start and returns zeros afterward. Every PSNR improvement above that baseline receives a higher reward.

## Limits

Importing `policy.py` must finish within 180 seconds. Policy construction and all calls for one prompt must use at most 60 seconds. One full prompt run must finish within 900 seconds.

The grading environment has one H100 GPU, 2 physical CPU cores, 16 GiB of memory, and no network. The model stays loaded while your policy runs, so your tensors share GPU memory with the model.

## Development files

* `/app/prompts/` contains public prompts, seeds, and text embeddings.
* `/app/examples/example_policy.py` is a working example.
* `/app/harness/run_eval.py` runs the public evaluation.
* `/app/dev_work/` is scratch space.

Run:

```text
/usr/bin/python3 /app/harness/run_eval.py --solution /app/output --limit 2
```

These development files do not move to the grading environment.
