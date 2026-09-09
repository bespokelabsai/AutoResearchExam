# Branching environment

`/app/submission/` is the deliverable: it is the only thing that crosses to the
evaluation. The evaluation runs its own copy of everything else in this tree, so
changing a file here changes what you measure locally and nothing about how you
are scored.

## What runs, and where

One evaluation run solves one set covering instance with a pinned SCIP
configuration. The solver lives in its own process and counts the nodes; your
policy lives in a second process that never gets a solver object. At every
branching decision the solver process sends the state of the current node LP to
your process and reads back the index of the candidate to branch on.

```
python3 /app/rollout.py --seeds 0-9 --shifts 0,1 --jobs 4
```

runs exactly that loop on instances generated from seeds 0..9, using
`/app/submission/policy.py`, and prints the node count of every run plus the
1-shifted geometric mean over them. Use `--submission DIR` to try another
directory. Instance seeds 0..1999 are yours; the graded instances come from 40
seeds outside that range which are not in this container and are not guessable.

Panels of a few dozen instances differ a lot in difficulty, so a single small
panel is a noisy estimate: compare policies on several disjoint panels.

## Files

| path | what it is |
|---|---|
| `/app/rollout.py` | the loop above |
| `/app/bnbenv/setcover.py` | instance generator and the pinned solver configuration |
| `/app/bnbenv/features.py` | builds the observation from the node LP |
| `/app/bnbenv/runner.py` | solver-side driver: one run, node count in, policy process out |
| `/app/bnbenv/host.py` | the process that loads `policy.py` and answers decisions |
| `/app/bnbenv/protocol.py` | the framing between those two processes |
| `/app/submission/policy.py` | your deliverable (a first-candidate stub to start from) |

The generator is a Balas-Ho style set covering construction:

- rows (covering constraints): 500
- columns (binary variables): 1000
- nonzero density: 0.05
- objective coefficients: uniform on {1, ..., 100}
- every column appears in at least two rows

`generate_instance(seed)` returns the costs and the constraint matrix in CSR
form. `build_model(seed, shift)` returns the configured `pyscipopt` model,
where `shift` sets both `randomization/permutationseed` and
`randomization/randomseedshift`.

## The observation

`select(obs)` is called once per branching decision. `obs` is a dict whose numpy
arrays are C-contiguous and read-only. `n_cols` and `n_rows` are the column and
row counts of the **current node LP**, so `n_rows` includes cuts added at the
root and both counts can differ from 1000 and 500.

| key | type | meaning |
|---|---|---|
| `var_feats` | float32 `(n_cols, 14)` | one row per LP column, layout below |
| `cons_feats` | float32 `(n_rows, 6)` | one row per LP row, layout below |
| `edge_index` | int32 `(2, nnz)` | `edge_index[0]` row positions, `edge_index[1]` column positions of the node LP nonzeros |
| `edge_vals` | float32 `(nnz,)` | the coefficient of that nonzero, divided by the L2 norm of its row |
| `cand_idx` | int32 `(k,)` | LP column positions of the fractional branching candidates |
| `cand_feats` | float32 `(k, 14)` | `var_feats[cand_idx]` |
| `depth` | int | depth of the current node |
| `n_nodes` | int | nodes processed so far in this run |
| `n_lps` | int | node LPs solved so far in this run |
| `dual_bound` | float | current global dual bound |
| `primal_bound` | float | current incumbent objective, `inf` if none |
| `gap` | float | current relative gap |

`var_feats` columns:

| # | name | meaning |
|---|---|---|
| 0 | `obj_coeff` | objective coefficient / L2 norm of the objective over LP columns |
| 1 | `lb_local` | local lower bound |
| 2 | `ub_local` | local upper bound |
| 3 | `sol_val` | value in the node LP solution |
| 4 | `sol_frac` | `sol_val - floor(sol_val)` |
| 5 | `sol_at_lb` | 1.0 if the LP value is at the local lower bound |
| 6 | `sol_at_ub` | 1.0 if the LP value is at the local upper bound |
| 7 | `reduced_cost` | reduced cost / L2 norm of the objective |
| 8 | `basis_lower` | basis status one-hot: at lower |
| 9 | `basis_basic` | basis status one-hot: basic |
| 10 | `basis_upper` | basis status one-hot: at upper |
| 11 | `basis_zero` | basis status one-hot: free at zero |
| 12 | `age_norm` | column age / (1 + `n_lps`) |
| 13 | `degree_norm` | number of LP rows the column appears in / `n_rows` |

`cons_feats` columns:

| # | name | meaning |
|---|---|---|
| 0 | `bias` | row bound / L2 norm of the row (right-hand side when finite, else left-hand side) |
| 1 | `has_lhs` | 1.0 if the row has a finite left-hand side |
| 2 | `is_tight` | 1.0 if the row is tight at the node LP solution |
| 3 | `dual_norm` | dual value / (row norm * objective norm) |
| 4 | `age_norm` | row age / (1 + `n_lps`) |
| 5 | `obj_cos_sim` | cosine similarity between the row and the objective |

## Installed

- `numpy` 2.3.4
- `scipy` 1.16.3
- `scikit-learn` 1.7.2
- `torch` 2.9.1+cpu
- `pyscipopt` 6.2.1 (SCIP 10.0)
- no network

`pyscipopt` is unrestricted in this container, so you can drive the solver
yourself while you work. Inside the policy process, at decision time, three
things are refused, and `/app/rollout.py` refuses them here too so that a local
run behaves like a graded one:

- importing `pyscipopt`, `gurobipy`, `mip`, `pulp`, `highspy`, `ortools`,
  `cplex`, `docplex`, `xpress` or `cylp`;
- loading a compiled extension module or shared library from outside the
  installed Python stack, so a native library carried in with the submission
  cannot be loaded;
- starting another process.

The first two raise, and any of the three ends the run at the cap.
