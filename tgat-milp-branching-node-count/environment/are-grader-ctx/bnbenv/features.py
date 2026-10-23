from __future__ import annotations

import numpy as np

VAR_FEATURE_NAMES = (
    "obj_coeff",
    "lb_local",
    "ub_local",
    "sol_val",
    "sol_frac",
    "sol_at_lb",
    "sol_at_ub",
    "reduced_cost",
    "basis_lower",
    "basis_basic",
    "basis_upper",
    "basis_zero",
    "age_norm",
    "degree_norm",
)

CONS_FEATURE_NAMES = (
    "bias",
    "has_lhs",
    "is_tight",
    "dual_norm",
    "age_norm",
    "obj_cos_sim",
)

N_VAR_FEATURES = len(VAR_FEATURE_NAMES)
N_CONS_FEATURES = len(CONS_FEATURE_NAMES)

_BASIS = {"lower": 8, "basic": 9, "upper": 10, "zero": 11}


def extract(model, cands, cands_sol):
    """Build the observation dict for the current node of `model`.

    `cands` are the fractional branching candidates (pyscipopt Variables) and
    `cands_sol` their LP values, as returned by getLPBranchCands().
    """
    cols = model.getLPColsData()
    rows = model.getLPRowsData()
    n_cols = len(cols)
    n_rows = len(rows)
    n_lps = model.getNLPs()
    inv_lps = 1.0 / (1.0 + n_lps)

    obj = np.empty(n_cols, dtype=np.float64)
    lb = np.empty(n_cols, dtype=np.float64)
    ub = np.empty(n_cols, dtype=np.float64)
    sol = np.empty(n_cols, dtype=np.float64)
    age = np.empty(n_cols, dtype=np.float64)
    redcost = np.empty(n_cols, dtype=np.float64)
    basis = np.zeros((n_cols, 4), dtype=np.float32)
    lp_pos_of = {}
    for k, col in enumerate(cols):
        var = col.getVar()
        obj[k] = col.getObjCoeff()
        lb[k] = col.getLb()
        ub[k] = col.getUb()
        sol[k] = col.getPrimsol()
        age[k] = col.getAge()
        redcost[k] = model.getVarRedcost(var)
        slot = _BASIS.get(col.getBasisStatus())
        if slot is not None:
            basis[k, slot - 8] = 1.0
        lp_pos_of[var.getIndex()] = k

    obj_norm = float(np.linalg.norm(obj))
    if obj_norm <= 0.0:
        obj_norm = 1.0

    edge_rows = []
    edge_cols = []
    edge_vals = []
    cons = np.zeros((n_rows, N_CONS_FEATURES), dtype=np.float32)
    for r, row in enumerate(rows):
        row_cols = row.getCols()
        row_vals = row.getVals()
        norm = float(row.getNorm())
        if norm <= 0.0:
            norm = 1.0
        constant = row.getConstant()
        lhs = row.getLhs()
        rhs = row.getRhs()
        activity = 0.0
        dot = 0.0
        for col, val in zip(row_cols, row_vals):
            pos = col.getLPPos()
            if pos < 0:
                continue
            edge_rows.append(r)
            edge_cols.append(pos)
            edge_vals.append(val / norm)
            activity += val * sol[pos]
            dot += val * obj[pos]
        has_lhs = 1.0 if lhs > -1e19 else 0.0
        bound = rhs if rhs < 1e19 else lhs
        if bound > 1e19 or bound < -1e19:
            bound = 0.0
        tight = 0.0
        act = activity + constant
        if (rhs < 1e19 and abs(act - rhs) < 1e-6) or \
           (lhs > -1e19 and abs(act - lhs) < 1e-6):
            tight = 1.0
        cons[r, 0] = (bound - constant) / norm
        cons[r, 1] = has_lhs
        cons[r, 2] = tight
        cons[r, 3] = row.getDualsol() / (norm * obj_norm)
        cons[r, 4] = row.getAge() * inv_lps
        cons[r, 5] = dot / (norm * obj_norm)

    edge_index = np.empty((2, len(edge_rows)), dtype=np.int32)
    edge_index[0] = np.asarray(edge_rows, dtype=np.int32)
    edge_index[1] = np.asarray(edge_cols, dtype=np.int32)
    edge_values = np.asarray(edge_vals, dtype=np.float32)
    degree = np.bincount(edge_index[1], minlength=n_cols).astype(np.float32)

    var_feats = np.empty((n_cols, N_VAR_FEATURES), dtype=np.float32)
    var_feats[:, 0] = obj / obj_norm
    var_feats[:, 1] = lb
    var_feats[:, 2] = ub
    var_feats[:, 3] = sol
    var_feats[:, 4] = sol - np.floor(sol)
    var_feats[:, 5] = (np.abs(sol - lb) < 1e-9).astype(np.float32)
    var_feats[:, 6] = (np.abs(sol - ub) < 1e-9).astype(np.float32)
    var_feats[:, 7] = redcost / obj_norm
    var_feats[:, 8:12] = basis
    var_feats[:, 12] = age * inv_lps
    var_feats[:, 13] = degree / max(1, n_rows)

    cand_idx = np.array([lp_pos_of[v.getIndex()] for v in cands],
                        dtype=np.int32)
    cand_feats = np.ascontiguousarray(var_feats[cand_idx])

    primal = model.getPrimalbound()
    dual = model.getDualbound()
    obs = {
        "var_feats": np.ascontiguousarray(var_feats),
        "cons_feats": np.ascontiguousarray(cons),
        "edge_index": np.ascontiguousarray(edge_index),
        "edge_vals": np.ascontiguousarray(edge_values),
        "cand_idx": cand_idx,
        "cand_feats": cand_feats,
        "depth": int(model.getDepth()),
        "n_nodes": int(model.getNNodes()),
        "n_lps": int(n_lps),
        "dual_bound": float(dual),
        "primal_bound": float(primal),
        "gap": float(model.getGap()),
    }
    return obs
