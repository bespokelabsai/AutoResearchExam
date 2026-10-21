import os
import random
import runpy
import sys

import numpy as np
import torch


def main() -> None:
    seed = int(os.environ.get("GRADER_SEED", "0"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    entry = sys.argv[1]
    sys.argv = [entry, *sys.argv[2:]]
    runpy.run_path(entry, run_name="__main__")


if __name__ == "__main__":
    main()
