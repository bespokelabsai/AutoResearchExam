import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    args = ap.parse_args()
    work = Path(args.workdir)

    sys.path.insert(0, str(work / "harness"))
    import runner

    spec = json.loads((work / "job.json").read_text())
    config = runner.make_config(spec["d"], spec["T"], spec["tau"])

    A = np.load(work / "samples_a.npy", mmap_mode="r", allow_pickle=False)
    b = np.load(work / "samples_b.npy", mmap_mode="r", allow_pickle=False)
    stream = runner.stream_from_samples(A, b, config, spec["alg_seed"])

    sys.path.insert(0, str(work / "solution"))
    import estimator

    out = estimator.estimate(stream, config)
    out = np.asarray(out, dtype=np.float64)
    np.save(work / "out" / "result.npy", out, allow_pickle=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
