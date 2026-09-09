import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    in_path, out_path, solution_dir = sys.argv[1], sys.argv[2], sys.argv[3]

    with np.load(in_path, allow_pickle=False) as z:
        columns = [str(c) for c in z["columns"]]
        train_X = pd.DataFrame(z["train_X"], columns=columns)
        train_y = z["train_y"]
        pool_X = pd.DataFrame(z["pool_X"], columns=columns)
        eval_X = pd.DataFrame(z["eval_X"], columns=columns)
        seed = int(z["seed"])

    sys.path.insert(0, solution_dir)
    import solution

    scores = solution.fit_predict(train_X, train_y, pool_X, eval_X, seed)

    arr = np.asarray(scores)
    if arr.dtype == object:
        raise TypeError(f"fit_predict returned a non-numeric object array: {arr.dtype}")
    np.save(out_path, arr, allow_pickle=False)
    Path(out_path).chmod(0o644)
    return 0


if __name__ == "__main__":
    sys.exit(main())
