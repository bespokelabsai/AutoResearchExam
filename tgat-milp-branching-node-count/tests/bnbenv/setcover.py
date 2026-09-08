from __future__ import annotations

import numpy as np

N_ROWS = 500
N_COLS = 1000
DENSITY = 0.05
MAX_COEF = 100

NODE_LIMIT = 20000
TIME_LIMIT = 120.0


def generate_instance(seed, n_rows=N_ROWS, n_cols=N_COLS, density=DENSITY,
                      max_coef=MAX_COEF):
    """Return (costs, row_indptr, row_indices) for one set covering instance.

    `costs` is an int array of length n_cols. The constraint matrix is returned
    in CSR form over rows: row r covers columns
    `row_indices[row_indptr[r]:row_indptr[r + 1]]`.
    """
    rng = np.random.default_rng(int(seed))
    nnzrs = int(n_rows * n_cols * density)
    if nnzrs // n_rows < 2 or nnzrs // n_cols < 2:
        raise ValueError("density too low for the requested shape")

    indices = rng.integers(0, n_rows, size=nnzrs)
    col_of_nnz = rng.integers(0, n_cols, size=nnzrs)
    col_of_nnz[: 2 * n_cols] = np.repeat(np.arange(n_cols), 2)
    _, col_nnzs = np.unique(col_of_nnz, return_counts=True)
    indices[:n_rows] = rng.permutation(n_rows)

    i = 0
    indptr = [0]
    for n in col_nnzs:
        if i >= n_rows:
            indices[i:i + n] = rng.choice(n_rows, size=n, replace=False)
        elif i + n > n_rows:
            remaining = np.setdiff1d(np.arange(n_rows), indices[i:n_rows],
                                     assume_unique=True)
            indices[n_rows:i + n] = rng.choice(
                remaining, size=i + n - n_rows, replace=False)
        i += n
        indptr.append(i)

    costs = rng.integers(1, max_coef + 1, size=n_cols)

    col_indptr = np.asarray(indptr, dtype=np.int64)
    row_of_nnz = np.asarray(indices[:i], dtype=np.int64)
    col_ids = np.repeat(np.arange(n_cols, dtype=np.int64), np.diff(col_indptr))
    order = np.lexsort((col_ids, row_of_nnz))
    row_sorted = row_of_nnz[order]
    row_indices = col_ids[order]
    row_counts = np.bincount(row_sorted, minlength=n_rows)
    row_indptr = np.concatenate(([0], np.cumsum(row_counts))).astype(np.int64)
    return costs.astype(np.int64), row_indptr, row_indices


def build_model(seed, shift, node_limit=NODE_LIMIT, time_limit=TIME_LIMIT,
                **gen_kwargs):
    """Build the pinned pyscipopt Model for (instance seed, seed shift)."""
    from pyscipopt import Model, quicksum

    costs, row_indptr, row_indices = generate_instance(seed, **gen_kwargs)
    n_cols = len(costs)

    model = Model(f"setcover-{seed}")
    model.hideOutput()
    xs = [model.addVar(vtype="B", obj=float(costs[j]), name=f"x{j}")
          for j in range(n_cols)]
    for r in range(len(row_indptr) - 1):
        cols = row_indices[row_indptr[r]:row_indptr[r + 1]]
        model.addCons(quicksum(xs[int(j)] for j in cols) >= 1, name=f"c{r}")

    model.setIntParam("separating/maxrounds", 0)
    model.setIntParam("presolving/maxrestarts", 0)
    model.setIntParam("randomization/permutationseed", int(shift))
    model.setIntParam("randomization/randomseedshift", int(shift))
    model.setIntParam("timing/clocktype", 2)
    model.setLongintParam("limits/nodes", int(node_limit))
    model.setRealParam("limits/time", float(time_limit))
    model.setParam("display/verblevel", 0)
    return model, xs
