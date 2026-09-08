import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp





try:
    import torch

    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "8")))
except ImportError:
    pass


def main() -> int:
    code_dir, train_path, out_path, n_items, max_nnz, max_dims, seed = sys.argv[1:8]
    n_items, max_nnz, max_dims, seed = int(n_items), int(max_nnz), int(max_dims), int(seed)

    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
    except NameError:
        pass

    sys.path.insert(0, str(Path(code_dir).resolve()))
    try:
        import solution
    except Exception as exc:
        print(f"import of solution.py failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    fn = getattr(solution, "build_item_embeddings", None)
    if not callable(fn):
        print("solution.py exposes no callable build_item_embeddings", file=sys.stderr)
        return 4

    X_train = sp.load_npz(str(train_path)).tocsr().astype(np.float32)
    X_train.sort_indices()

    A = fn(X_train, n_items, max_nnz, max_dims, seed)

    if sp.issparse(A):
        A = sp.csr_matrix(A)
    elif isinstance(A, np.ndarray) and A.ndim == 2:
        A = sp.csr_matrix(A)
    else:
        print(f"entry point returned {type(A)!r}, not a 2-D sparse matrix", file=sys.stderr)
        return 5

    A = A.tocsr()
    if A.shape[0] < 1 or A.shape[1] < 1:
        print(f"entry point returned a degenerate shape {A.shape}", file=sys.stderr)
        return 6
    A.sum_duplicates()
    A.eliminate_zeros()

    n_rows, n_dims = int(A.shape[0]), int(A.shape[1])
    idx = np.full((n_items, max_nnz), -1, dtype=np.int64)
    val = np.zeros((n_items, max_nnz), dtype=np.float64)
    nnz_per_row = np.zeros(n_items, dtype=np.int64)

    counts = np.diff(A.indptr)
    for r in range(min(n_rows, n_items)):
        lo, hi = A.indptr[r], A.indptr[r + 1]
        cols = A.indices[lo:hi]
        vals = A.data[lo:hi].astype(np.float64)
        nnz_per_row[r] = int(counts[r])
        if len(cols) > max_nnz:
            keep = np.argsort(-np.abs(vals))[:max_nnz]
            cols, vals = cols[keep], vals[keep]
        idx[r, : len(cols)] = cols
        val[r, : len(vals)] = vals

    np.savez(
        str(out_path),
        idx=idx.astype(np.int32),
        val=val,
        nnz_per_row=nnz_per_row,
        n_rows=np.array([n_rows], dtype=np.int64),
        n_dims=np.array([n_dims], dtype=np.int64),
    )
    print(json.dumps({"n_rows": n_rows, "n_dims": n_dims, "max_nnz": int(counts.max(initial=0))}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
