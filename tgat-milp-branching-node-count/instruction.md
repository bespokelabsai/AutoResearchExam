# Branching policy for a mixed integer linear program

Build a branching policy that lets a mixed integer linear programming (MILP) solver prove
optimality in as few search nodes as possible on unseen set covering instances.

The solver uses branch and bound. At each search node, it asks which fractional variable to branch
on. You receive only arrays that describe the current linear programming (LP) relaxation.

## What to write

Write `/app/submission/policy.py`. It must define:

- `class Policy` with `__init__(self)` and `select(self, obs) -> int`. The solver creates
  one `Policy` instance per instance, so it may keep state across the decisions for that
  instance. It calls `select` at every branching decision.
- Optionally, a module-level `def load() -> None`. The solver calls this once per run,
  before touching the instance, so you can load anything you trained.

`select` must return an index `i` into `obs['cand_idx']`; the solver then branches on
`obs['cand_idx'][i]`. Valid returns are a Python `int` or a numpy integer with
`0 <= i < len(obs['cand_idx'])`. Anything else - a float, a `bool`, an out-of-range value -
ends that run.

`obs` is a dict of read-only, C-contiguous numpy arrays and Python scalars describing the
node's LP relaxation: `var_feats` `(n_cols, 14)`, `cons_feats` `(n_rows, 6)`, `edge_index` `(2, nnz)`,
`edge_vals` `(nnz,)`, `cand_idx` `(k,)`, `cand_feats` `(k, 14)`, plus `depth`, `n_nodes`,
`n_lps`, `dual_bound`, `primal_bound`, `gap`.

`/app/README.md` documents every feature column, the instance generator, and the pinned
solver configuration. `/app/rollout.py` runs your policy on instances you generate
yourself, the same way the evaluation runs it.

## How you're scored

- Evaluation set: 40 held-out instances from the same generator, each solved twice under
  solver seed shifts 0 and 1 - 80 runs total.
- A run that proves optimality is charged the solver's own node count.
- Node cap: 20000 nodes. Time cap: 120 seconds of wall clock. A run that fails to prove
  optimality within either cap is charged 20000 nodes.
- Score is the 1-shifted geometric mean of the node counts over the 80 runs:
  $\exp\big(\operatorname{mean}(\log(\text{nodes} + 1))\big) - 1$. Lower is better.
- Floor: a one-line rule that always branches on the single most extreme candidate of one
  feature scores zero. Every improvement above that floor is worth strictly more, and
  there's no point past which improvement stops counting.

## Rules the evaluation enforces

- Only `/app/submission/` crosses over to the evaluation - your solution must be
  self-contained there, since nothing else you create in `/app` is carried over. It must
  be at most 200 MB, and a symlink inside it that points outside it is rejected. You can
  work anywhere else in `/app`, but the evaluation uses its own copy of the generator, the
  feature extraction, and the run driver, so editing those local copies changes nothing
  about how you're scored.
- Per run, `load()` plus every import must finish within 30 seconds; the solve itself gets
  120 seconds of wall clock, and your decision time counts against that. Exceeding the
  30-second, 120-second, or 20000-node limit charges the node cap - so does any exception,
  a non-integer or out-of-range return value, or a policy process that spawns another
  process.
- Your policy runs in a process with no solver object and no access to a MILP solver:
  importing `pyscipopt`, `gurobipy`, `mip`, `pulp`, `highspy`, `ortools`, `cplex`,
  `docplex`, `xpress`, or `cylp` inside it raises `ImportError`, and so does loading a
  compiled extension module or shared library from outside the installed Python stack.
  `numpy`, `scipy`, `scikit-learn`, and `torch` are available. `pyscipopt` is installed and
  unrestricted in this container, so you can use it while developing - only the graded
  decision process is sealed off from it.
- You have 8 CPUs, 14 GB of memory, 0 GPUs, and no network; the evaluation has the same,
  and it grades eight runs concurrently with threads pinned to one per process
  (`OMP_NUM_THREADS=1`). The whole evaluation has a wall-clock budget of 1900 seconds; any
  run not finished by then is charged the cap.
- Instance seeds 0 to 1999 are yours - generate as many instances from them as you like.
  The 40 graded instances come from seeds outside that range, aren't in this container,
  and aren't guessable, so nothing you precompute per instance will match them.
