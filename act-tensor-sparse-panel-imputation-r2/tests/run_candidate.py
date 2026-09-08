import sys
from pathlib import Path

import numpy as np


def main():
    cand_dir, panel_path, seed, out_path = sys.argv[1:5]
    sys.path.insert(0, cand_dir)

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "candidate_impute", str(Path(cand_dir) / "impute.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    panel = np.load(panel_path, allow_pickle=False)
    result = module.impute(panel, int(seed))
    np.save(out_path, np.asarray(result), allow_pickle=False)


if __name__ == "__main__":
    main()
