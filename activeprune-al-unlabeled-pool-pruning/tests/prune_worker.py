from __future__ import annotations

import importlib.util
import json
import os
import random
import sys
import traceback

import numpy as np


def _private_channel():
    """Move the protocol onto duplicated descriptors, then point fd 0/1 at /dev/null and
    stderr so submitted code cannot read from or write into the channel."""
    read_fd = os.dup(0)
    write_fd = os.dup(1)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.close(devnull)
    os.dup2(2, 1)
    sys.stdin = open(os.devnull, "r")
    sys.stdout = sys.stderr
    return os.fdopen(read_fd, "r", encoding="utf-8"), \
        os.fdopen(write_fd, "w", encoding="utf-8")


def main():
    module_dir, entry_module, pool_path = sys.argv[1:4]
    channel_in, channel_out = _private_channel()

    def emit(payload):
        channel_out.write(json.dumps(payload) + "\n")
        channel_out.flush()

    try:
        with open(pool_path, "r", encoding="utf-8") as fh:
            pool_texts = json.load(fh)
        sys.path.insert(0, module_dir)
        spec = importlib.util.spec_from_file_location(
            entry_module, os.path.join(module_dir, entry_module + ".py"))
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {entry_module}.py from {module_dir}")
        random.seed(0)
        np.random.seed(0)
        module = importlib.util.module_from_spec(spec)
        sys.modules[entry_module] = module
        spec.loader.exec_module(module)
        prune = getattr(module, "prune")
        if not callable(prune):
            raise TypeError("prune is not callable")
    except BaseException:
        emit({"status": "startup_error", "detail": traceback.format_exc(limit=8)})
        return 0

    emit({"status": "ready"})

    for line in channel_in:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        if request.get("op") == "stop":
            return 0
        try:
            result = prune(pool_texts,
                           request["available"],
                           request["labeled_indices"],
                           request["labeled_labels"],
                           request["keep"],
                           request["iteration"],
                           request["rng_seed"])
            emit({"status": "ok", "selection": _as_int_list(result)})
        except BaseException:
            emit({"status": "call_error", "detail": traceback.format_exc(limit=8)})
            return 0
    return 0


def _as_int_list(result):
    """Serialise the return value while preserving enough type information for the harness
    to reject a wrong type.  Anything unrecognised is passed through as its type name."""
    if isinstance(result, (list, tuple)):
        items = list(result)
    else:
        shape = getattr(result, "shape", None)
        dtype = getattr(result, "dtype", None)
        if shape is None or dtype is None:
            return {"bad_type": type(result).__name__}
        if len(shape) != 1 or dtype.kind not in "iu":
            return {"bad_type": f"array(ndim={len(shape)}, dtype={dtype})"}
        items = result.tolist()
    out = []
    for value in items:
        if isinstance(value, bool):
            return {"bad_type": "element bool"}
        if not isinstance(value, int):
            try:
                as_int = value.__index__()
            except Exception:
                return {"bad_type": f"element {type(value).__name__}"}
            out.append(as_int)
        else:
            out.append(value)
    return out


if __name__ == "__main__":
    sys.exit(main())
