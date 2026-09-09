from __future__ import annotations

import importlib.util
import json
import pickle
import signal
import sys
import time


class _CallTimeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _CallTimeout()


def _load(module_dir: str, module_file: str):
    sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location("candidate_interval", module_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_file}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    batch = pickle.load(sys.stdin.buffer)
    cap = float(batch["cap"])
    out = {"import_error": None, "results": []}

    try:
        import numpy as np

        mod = _load(batch["module_dir"], batch["module_file"])
        fn = getattr(mod, "ece2_interval")
        if not callable(fn):
            raise TypeError("ece2_interval is not callable")
    except BaseException as exc:
        out["import_error"] = f"{type(exc).__name__}: {exc}"[:200]
        sys.stdout.write(json.dumps(out))
        return 0

    signal.signal(signal.SIGALRM, _on_alarm)
    for call in batch["calls"]:
        rec = {"key": call["key"]}
        rng = np.random.default_rng(int(call["seed"]))
        started = time.monotonic()
        signal.setitimer(signal.ITIMER_REAL, cap)
        try:
            lo, hi = fn(call["Z"], call["Y"], int(call["k"]), float(call["alpha"]), rng)
            rec.update(status="ok", lo=float(lo), hi=float(hi))
        except _CallTimeout:
            rec.update(status="timeout")
        except BaseException as exc:
            rec.update(status="error", detail=f"{type(exc).__name__}: {exc}"[:120])
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            rec["elapsed"] = time.monotonic() - started
        if rec.get("status") == "ok" and rec["elapsed"] > cap:
            rec["status"] = "timeout"
        out["results"].append(rec)

    sys.stdout.write(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
