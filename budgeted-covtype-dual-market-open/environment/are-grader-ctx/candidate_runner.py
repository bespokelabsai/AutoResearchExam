from __future__ import annotations

import contextlib
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np


def send(channel, message) -> None:
    channel.write(json.dumps(message) + "\n")
    channel.flush()


def main() -> None:
    policy_path = Path(sys.argv[1])
    data_dir = Path(sys.argv[2])
    sys.path.insert(0, str(policy_path.parent))
    spec = importlib.util.spec_from_file_location("candidate_policy", policy_path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load policy")
    random.seed(0)
    np.random.seed(0)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    policy = module.Policy(str(data_dir))
    real_stdout = sys.stdout
    labeled = {}
    with contextlib.redirect_stdout(sys.stderr):
        for line in sys.stdin:
            command = json.loads(line)
            kind = command["cmd"]
            if kind == "end":
                return
            if kind == "select":
                prices = {int(k): float(v) for k, v in command["prices"].items()}
                rows = policy.select_queries(
                    dict(labeled), float(command["budget_left"]), prices
                )
                if not rows:
                    send(real_stdout, {"act": "stop"})
                    continue
                send(real_stdout, {"act": "query", "rows": [int(row) for row in rows]})
                reply = json.loads(sys.stdin.readline())
                for key, value in reply.get("labels", {}).items():
                    labeled[int(key)] = int(value)
                continue
            if kind != "row":
                raise ValueError("unknown command")
            observed = {int(k): float(v) for k, v in command["observed"].items()}
            prices = {int(k): float(v) for k, v in command["prices"].items()}
            budget_left = float(command["budget"])
            while True:
                feature_id = policy.select_next(dict(observed), budget_left, dict(prices))
                if feature_id is None or int(feature_id) in observed:
                    break
                feature_id = int(feature_id)
                if feature_id not in prices or prices[feature_id] > budget_left + 1e-9:
                    break
                send(real_stdout, {"act": "buy", "fid": feature_id})
                reply = json.loads(sys.stdin.readline())
                if reply.get("val") is None:
                    break
                observed[feature_id] = float(reply["val"])
                budget_left -= prices[feature_id]
                prices = {int(k): float(v) for k, v in reply["prices"].items()}
            prediction = policy.predict(dict(observed))
            send(real_stdout, {"act": "predict", "label": int(prediction)})


if __name__ == "__main__":
    main()
