# Limiting covariance of an online sketched Newton method

Estimate the optimizer's limiting covariance matrix $\Xi$ from the records of one run.

The fixed optimizer in `/app/harness/runner.py` processes one linear regression sample at a time.
Its final estimate varies across repeated runs. After scaling by the final step size, that
variation approaches a normal distribution with covariance matrix $\Xi$:

$$
\sqrt{1/\alpha_t}\,\left(x_t - x^\star\right) \;\Rightarrow\; N(0, \Xi) \quad \text{as } t \to \infty,
$$

The grader checks the variance that your matrix gives for the fixed direction
$w = (1/d, \ldots, 1/d)$.

At each step, the optimizer gives you its current estimate, step size, gradient, Hessian, running
Hessian average, and sampled sketch indices. `/app/harness/runner.py` defines this process.

Write `/app/output/estimator.py` with this function:

```python
def estimate(stream, config) -> numpy.ndarray:
```

The function gets one pass over the records from one run. It must return an estimate of $\Xi$.
The direction $w$ is also available as `config["w"]`.

## The stream

`stream` is an iterator over exactly `T` records, one per iteration, in order. Each record is a
dict whose arrays are read-only and stay valid for as long as you hold them.

| key | shape | meaning |
|---|---|---|
| `t` | int | iteration index, `0 .. T-1` |
| `x` | `(d,)` | `x_{t+1}`, the iterate after this step's update |
| `alpha` | float | the realized stepsize `alpha_t` |
| `g` | `(d,)` | the stochastic gradient at `x_t` |
| `H` | `(d, d)` | the sample Hessian at `x_t` |
| `B` | `(d, d)` | the Hessian average `B_t` this step's inner solve used |
| `sketch_idx` | `(tau,)` | the Kaczmarz indices this step's inner solve drew |

`config` also contains `d`, `T`, `tau`, `q`, `sketch`, `beta`, `c_beta`, `loss`, `x0`, `B0`, and
`w`. It does not reveal the instance's design covariance, noise scale, `x_star`, or seed.

## Deliverable contract

- Entry point: `estimate` in `/app/output/estimator.py`. Everything else under `/app/output/`
  travels with it: helper modules are importable, and `.npy`/`.npz` files load normally. Nothing
  outside that directory reaches the grader.
- Return a `(d, d)` array that can be converted to `float64`. Every entry must be finite. The
  grader reads it as `float64` and uses only $w^\top \hat\Xi w$.
- A run fails if the function returns the wrong shape, an invalid value, a non-finite entry, or no
  array. It also fails if the function raises an exception.
- The grader uses a fresh process for each run. No state carries over between runs.
- The process is stopped after 90 seconds. This limit includes the time needed to produce the
  stream.
- `numpy` and `scipy` are installed. There is no network.
- Each run uses one thread. The machine has 8 CPUs and 14 GB of memory, and several runs may occur
  at the same time.

## Scoring

- The sealed instances come from the same family as the development instances, but they are drawn
  independently.
- The grader runs each instance 20 times with fixed seeds.
- Let $v = w^\top \Xi w$ be the instance's exact limiting value, computed in closed form from
  that instance's parameters.

$$
\text{error(instance)} = \left| \operatorname{mean}_{\text{replications}}
\left( \frac{w^\top \hat\Xi w - v}{v} \right) \right|,
\qquad
\text{metric} = \operatorname{mean}_{\text{instances}}\big[\text{error(instance)}\big]
\quad \text{(lower is better)}
$$

- If any run for an instance fails, that instance gets an error of `1.0`. This is the same error
  as an all-zero matrix.
- An all-zero matrix earns zero reward. Every lower metric earns more reward.

## Development data

`/app/harness/dev/` has:

- `inst_*.json`: full parameters (`equi_r`, `sigma`, `x_star`, `tau`, `seed_base`).
- `inst_*_xi_star.npy`: that instance's exact `Xi`.

Build a stream for one of them with

```python
cfg    = runner.make_config(inst["d"], inst["T"], inst["tau"])
stream = runner.make_stream(inst, cfg, sample_seed, alg_seed)
```

The grader builds its own streams from the same frozen runner and grades its own copy, so
editing `/app/harness/` changes nothing.

Every instance, sealed or development, comes from one procedure:

- $d = 20$, $T = 300000$
- $\tau$ drawn from $\{10, 20, 40\}$
- design covariance: equi-correlation, $\Sigma_{ii} = 1$, $\Sigma_{ij} = r$, with
  $r \in [0.10, 0.30]$
- Gaussian noise scale $\sigma \in [0.5, 2.0]$
- $x^\star$ uniform on the sphere of radius $1/\sqrt{d}$
