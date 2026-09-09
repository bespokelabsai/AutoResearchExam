import sys

import numpy as np


def main() -> int:
    in_path, out_path, module_dir = sys.argv[1], sys.argv[2], sys.argv[3]

    with np.load(in_path, allow_pickle=False) as data:
        X_train = np.ascontiguousarray(data["X_train"], dtype=np.float64)
        X_eval = np.ascontiguousarray(data["X_eval"], dtype=np.float64)
        seed = int(data["seed"][0])

    sys.path.insert(0, module_dir)
    import impute

    result = impute.impute(X_train, X_eval, seed)

    np.save(out_path, np.asarray(result), allow_pickle=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
