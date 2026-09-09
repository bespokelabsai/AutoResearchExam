import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

N_ITEMS = 10000
MAX_NNZ_PER_ITEM = 8
MAX_DIMS = 65536


def main() -> int:
    code_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "/app/output").resolve()
    sys.path.insert(0, str(code_dir))
    import solution

    fn = getattr(solution, "build_item_embeddings", None)
    if not callable(fn):
        print("FAIL: solution.py exposes no callable build_item_embeddings")
        return 1

    X_train = sp.load_npz("/app/data/train.npz").tocsr().astype(np.float32)
    X_train.sort_indices()

    start = time.monotonic()
    A = fn(X_train, N_ITEMS, MAX_NNZ_PER_ITEM, MAX_DIMS, 815)
    elapsed = time.monotonic() - start
    print(f"build_item_embeddings returned in {elapsed:.1f}s")

    if sp.issparse(A):
        A = sp.csr_matrix(A)
    elif isinstance(A, np.ndarray) and A.ndim == 2:
        A = sp.csr_matrix(A)
    else:
        print(f"FAIL: returned {type(A)!r}, not a 2-D sparse matrix or ndarray")
        return 1
    A = A.tocsr()
    A.sum_duplicates()
    A.eliminate_zeros()

    ok = True
    counts = np.diff(A.indptr)
    print(f"shape={A.shape}  d={A.shape[1]}  nonzeros={A.nnz}")
    print(f"nonzeros per row: max={counts.max(initial=0)} mean={counts.mean():.3f} "
          f"empty rows={int((counts == 0).sum())}")
    if A.shape[0] != N_ITEMS:
        print(f"FAIL: {A.shape[0]} rows, expected {N_ITEMS}")
        ok = False
    if not 1 <= A.shape[1] <= MAX_DIMS:
        print(f"FAIL: d={A.shape[1]} outside 1..{MAX_DIMS}")
        ok = False
    if counts.max(initial=0) > MAX_NNZ_PER_ITEM:
        print(f"FAIL: row {int(np.argmax(counts))} has {counts.max()} nonzeros, budget is "
              f"{MAX_NNZ_PER_ITEM}")
        ok = False
    if not np.isfinite(A.data).all():
        print("FAIL: A contains a non-finite value")
        ok = False
    print("contract OK" if ok else "contract VIOLATED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
