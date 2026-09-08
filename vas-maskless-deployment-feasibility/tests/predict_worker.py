import importlib.util
import os
import random
import sys
import traceback

import numpy as np

_SEED = int(os.environ.get("PYTHONHASHSEED", "0"))
random.seed(_SEED)
np.random.seed(_SEED)
try:
    import torch

    torch.manual_seed(_SEED)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
except ImportError:
    pass

OBS_DIM = 514
OBS_BYTES = OBS_DIM * 4
N_ACTIONS = 24


def _read_exact(stream, count):
    chunks = []
    got = 0
    while got < count:
        piece = stream.read(count - got)
        if not piece:
            return None
        chunks.append(piece)
        got += len(piece)
    return b"".join(chunks)


def main():
    deliverable = os.path.abspath(sys.argv[1])
    sys.path.insert(0, deliverable)
    sys.path.append("/app")

    inp = sys.stdin.buffer
    out = sys.stdout.buffer

    try:
        entry = os.path.join(deliverable, "predictor.py")
        spec = importlib.util.spec_from_file_location("submission_predictor", entry)
        if spec is None or spec.loader is None:
            raise ImportError("cannot load %s" % entry)
        module = importlib.util.module_from_spec(spec)
        sys.modules["submission_predictor"] = module
        spec.loader.exec_module(module)
        predictor = module.MaskPredictor(deliverable)
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        out.write(b"F")
        out.flush()
        return 1

    out.write(b"I")
    out.flush()

    while True:
        op = inp.read(1)
        if not op or op == b"Q":
            return 0
        if op not in (b"R", b"P"):
            return 1
        raw = _read_exact(inp, OBS_BYTES)
        if raw is None:
            return 1
        obs = np.frombuffer(raw, dtype=np.float32).copy()
        try:
            if op == b"R":
                predictor.reset(obs)
                out.write(b"K")
            else:
                mask = predictor.predict(obs)
                if not isinstance(mask, np.ndarray):
                    raise TypeError("predict returned %r, not a numpy array" % type(mask))
                if mask.dtype != np.bool_:
                    raise TypeError("predict returned dtype %s, not bool" % mask.dtype)
                if mask.shape != (N_ACTIONS,):
                    raise ValueError("predict returned shape %r, not (24,)" % (mask.shape,))
                out.write(b"K" + mask.astype(np.uint8).tobytes())
        except BaseException:
            traceback.print_exc(file=sys.stderr)
            out.write(b"E")
        out.flush()


if __name__ == "__main__":
    sys.exit(main())
