# Dual-market budgeted classification

Build a classification policy. It must learn while label prices move, and it must choose which
feature values to buy while feature prices move. The goal is to maximize balanced accuracy: higher
is always better, with no cutoff where further improvement stops counting. A constant-class policy
is the trivial baseline and scores 0. Any solution that crashes, times out, or produces
unparseable output also scores 0.

## Data

`/app/data/train_features.npy` is a public, unlabeled development pool with shape `(15000, 54)`
and dtype `float32`. `/app/data/meta.json` describes that pool. Use it for development only; it is
disjoint from both grading splits.

At grading time, your policy is constructed with a new `data_dir`. That directory contains:

- `train_features.npy`: a sealed unlabeled pool with shape `(60000, 54)` and dtype `float32`.
- `meta.json`: contains `n_features = 54`, `n_classes = 7`, `n_train = 60000`,
  `feature_cost = 6.0`, `label_cost = 2000.0`, the 54 initial feature prices in `costs`, and
  `dynamic_price = "both"`.

The intermediate and final graders each use their own training pool and their own production
cases, all drawn from the same anonymized classification distribution but disjoint from each
other and from the development pool. Each production split caps every class at 300 cases.
Production features and every label stay sealed; your code only sees labels and feature values
through the calls described below.

## Deliverable

Create `/app/output/policy.py`, defining:

```python
class Policy:
    def __init__(self, data_dir: str):
        ...

    def select_queries(
        self,
        labeled: dict[int, int],
        budget_left: float,
        prices: dict[int, float],
    ) -> list[int]:
        ...

    def select_next(
        self,
        observed: dict[int, float],
        budget_left: float,
        prices: dict[int, float],
    ) -> int | None:
        ...

    def predict(self, observed: dict[int, float]) -> int:
        ...
```

The whole `/app/output` directory is delivered to each verifier, so `policy.py` can load weights,
config, or helper modules stored next to it. At grading time it must not depend on files
elsewhere in `/app`; use the `data_dir` argument to find the sealed pool.

## Label market

The grader constructs `Policy` once, then calls `select_queries` in rounds.

- `labeled` maps every pool row id you've already purchased to its revealed class id.
- Valid pool row ids run from 0 through 59999.
- `budget_left` starts at 2000.0 money units.
- `prices` maps each class id (0 through 6) to its current advertised label price.
- Return new pool row ids in purchase order, or `[]` to stop.

Each accepted row reveals its hidden class and charges the current advertised price for that
class. Buying repeatedly from one class moves later advertised prices; classes bought less often
can get cheaper. If the remaining budget can't cover the full price of the last purchase, it's
charged only what's left; the grader never spends more than 2000.0 total. A round that buys no new
valid row ends acquisition. After acquisition ends, `select_queries` is called once more with
`budget_left = 0.0` so the policy can do its final fit.

## Feature market

The grader then serves the sealed production cases in a fixed order. Each case starts with
`observed = {}` and a fresh budget of 6.0 money units.

- `observed` maps purchased feature ids to their revealed float values.
- `prices` maps every feature id (0 through 53) to its current advertised price.
- Return an unobserved feature id whose advertised price is no more than `budget_left`, or `None`
  to stop buying for this case.

The advertised price is charged exactly, and prices refresh after every accepted purchase. Buying
a feature repeatedly moves its later prices, including in later production cases; features bought
less often can get cheaper. Purchases that are duplicate, out of range, or unaffordable are
invalid. Once buying stops, `predict` gets the purchased values and must return a Python integer
class id from 0 through 6.

Balanced accuracy is the mean of the seven per-class recalls, so every class carries equal weight.
Your policy has 240 seconds of wall-clock time per grading split, covering construction, label
acquisition, fitting, all feature decisions, and all predictions.

## Environment

8 CPU cores, 14336 MB memory, no GPU, 150 GB disk, no network access. `numpy==2.1.3` and
`scikit-learn==1.5.2` are installed in both the development and verifier images.
`/app/POLICY_TEMPLATE.py` is a minimal interface template.
