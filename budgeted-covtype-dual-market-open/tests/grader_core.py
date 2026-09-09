from __future__ import annotations

import json
import math
import os
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

N_CLASSES = 7
FEATURE_BUDGET = 6.0
LABEL_BUDGET = 2000.0
POLICY_TIMEOUT_SEC = 240.0
M0 = 1.0 / N_CLASSES
REFERENCE_METRIC = 0.5567
X_REF = REFERENCE_METRIC - M0

MU = 0.5
ALPHA = 0.02
RHO = 0.10
ETA_ROW = 0.02
LEVEL_LO = 0.25
LEVEL_HI = 4.0
SHARE_FLOOR = 1e-3
SIGMA = 0.10
FEATURE_PRICE_SEED = 20260803
LABEL_PRICE_SEED = 20260804


class MarketPricer:
    def __init__(self, initial, cap: float, seed: int, level_control=None) -> None:
        self.base = np.asarray(initial, dtype=float).copy()
        self.floor = self.base / 10.0
        self.cap = float(cap)
        self.seed = int(seed)
        self.share = np.full(len(self.base), 1.0 / len(self.base))
        self.level_control = level_control

    def published(self, tick: int) -> np.ndarray:
        level = 1.0 if self.level_control is None else self.level_control.value
        noise = np.random.default_rng([self.seed, int(tick)]).standard_normal(len(self.base))
        return level * self.base * np.exp(SIGMA * noise)

    def on_purchase(self, item: int) -> None:
        self.share *= 1.0 - ALPHA
        self.share[item] += ALPHA

    def on_tick(self, spent: float = 0.0, pinched: bool = False) -> None:
        log_share = np.log(np.maximum(len(self.share) * self.share, SHARE_FLOOR))
        self.base = np.clip(self.base * np.exp(RHO * MU * log_share), self.floor, self.cap)
        if self.level_control is not None:
            self.level_control.on_tick(spent, pinched)


class PerRowLevel:
    def __init__(self) -> None:
        self.value = 1.0

    def on_tick(self, spent: float, pinched: bool) -> None:
        drift = ETA_ROW * ((1.0 - spent / FEATURE_BUDGET) - float(pinched))
        self.value = float(np.clip(self.value * np.exp(drift), LEVEL_LO, LEVEL_HI))


class CandidateProcess:
    def __init__(self, policy_path: Path, data_dir: Path, timeout=POLICY_TIMEOUT_SEC) -> None:
        self.deadline = time.monotonic() + float(timeout)
        runner = os.environ.get("BUDGETED_RUNNER_PATH", "/opt/budgeted/candidate_runner.py")
        command = [
            "runuser", "-u", "agent", "--", "/usr/local/bin/python3",
            runner, str(policy_path), str(data_dir),
        ]
        if os.environ.get("BUDGETED_LIGHT_DIRECT") == "1":
            command = [sys.executable, runner, str(policy_path), str(data_dir)]
        candidate_env = os.environ.copy()
        candidate_env.pop("BUDGETED_SPLIT", None)
        self.proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid,
            env=candidate_env,
        )

    def send(self, message) -> None:
        if self.proc.stdin is None:
            raise RuntimeError("candidate stdin unavailable")
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()

    def receive(self) -> dict:
        if self.proc.stdout is None:
            raise RuntimeError("candidate stdout unavailable")
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("candidate deadline exceeded")
        ready, _, _ = select.select([self.proc.stdout], [], [], remaining)
        if not ready:
            raise TimeoutError("candidate deadline exceeded")
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("candidate exited without a protocol response")
        message = json.loads(line)
        if not isinstance(message, dict):
            raise TypeError("protocol response must be an object")
        return message

    def close(self) -> None:
        try:
            if self.proc.poll() is None:
                self.send({"cmd": "end"})
                self.proc.wait(timeout=max(0.1, self.deadline - time.monotonic()))
        except Exception:
            pass
        finally:
            if self.proc.poll() is None:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    recalls = [float(np.mean(y_pred[y_true == c] == c)) for c in range(N_CLASSES)]
    return float(np.mean(recalls))


def reward_for(metric: float) -> float:
    improvement = max(0.0, metric - M0)
    units = improvement / X_REF
    return units / (1.0 + units)


def run(split_dir: Path, policy_path: Path) -> dict:
    if not policy_path.is_file() or policy_path.stat().st_size == 0:
        raise FileNotFoundError("missing or empty /app/output/policy.py")
    meta = json.loads((split_dir / "meta.json").read_text())
    X_pool = np.load(split_dir / "train_features.npy", allow_pickle=False)
    y_pool = np.load(split_dir / "train_labels.npy", allow_pickle=False)
    X_test = np.load(split_dir / "test_features.npy", allow_pickle=False)
    y_test = np.load(split_dir / "test_labels.npy", allow_pickle=False)
    if X_pool.shape != (meta["n_train"], 54) or X_test.shape != (meta["n_test"], 54):
        raise ValueError("sealed data shape mismatch")
    with tempfile.TemporaryDirectory(prefix="budgeted_candidate_") as temp:
        candidate_data = Path(temp)
        np.save(candidate_data / "train_features.npy", X_pool)
        (candidate_data / "meta.json").write_text(json.dumps(meta))
        os.chmod(candidate_data, 0o755)
        os.chmod(candidate_data / "train_features.npy", 0o444)
        os.chmod(candidate_data / "meta.json", 0o444)
        process = CandidateProcess(policy_path, candidate_data)
        predictions = np.empty(len(y_test), dtype=np.int64)
        label_pricer = MarketPricer(np.ones(N_CLASSES), cap=8.0, seed=LABEL_PRICE_SEED)
        feature_pricer = MarketPricer(
            np.asarray(meta["costs"]), cap=2.0 * FEATURE_BUDGET,
            seed=FEATURE_PRICE_SEED, level_control=PerRowLevel(),
        )
        labeled = {}
        budget_left = LABEL_BUDGET
        purchase_tick = 0
        try:
            while budget_left > 1e-9:
                prices = label_pricer.published(purchase_tick)
                process.send({
                    "cmd": "select", "budget_left": budget_left,
                    "prices": {str(i): float(value) for i, value in enumerate(prices)},
                })
                response = process.receive()
                if response.get("act") == "stop":
                    break
                if response.get("act") != "query" or not isinstance(response.get("rows"), list):
                    raise ValueError("invalid select response")
                rows = []
                for raw in response["rows"]:
                    row = int(raw)
                    if 0 <= row < len(y_pool) and row not in labeled and row not in rows:
                        rows.append(row)
                revealed = {}
                charges = {}
                before = budget_left
                for row in rows:
                    if budget_left <= 1e-9:
                        break
                    class_id = int(y_pool[row])
                    prices = label_pricer.published(purchase_tick)
                    charge = min(float(prices[class_id]), budget_left)
                    budget_left -= charge
                    label_pricer.on_purchase(class_id)
                    label_pricer.on_tick(spent=charge)
                    purchase_tick += 1
                    labeled[row] = class_id
                    revealed[str(row)] = class_id
                    charges[str(row)] = charge
                process.send({"labels": revealed, "charges": charges, "budget_left": budget_left})
                if math.isclose(before, budget_left, abs_tol=1e-12):
                    break
            prices = label_pricer.published(purchase_tick)
            process.send({
                "cmd": "select", "budget_left": 0.0,
                "prices": {str(i): float(value) for i, value in enumerate(prices)},
            })
            process.receive()
            for row_index, row in enumerate(X_test):
                prices = feature_pricer.published(row_index)
                spent = 0.0
                purchased = set()
                process.send({
                    "cmd": "row", "budget": FEATURE_BUDGET, "observed": {},
                    "prices": {str(i): float(value) for i, value in enumerate(prices)},
                })
                while True:
                    response = process.receive()
                    action = response.get("act")
                    if action == "predict":
                        label = response.get("label")
                        if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < N_CLASSES:
                            raise ValueError("prediction must be an integer from 0 through 6")
                        predictions[row_index] = label
                        break
                    if action != "buy":
                        raise ValueError("invalid row response")
                    feature_id = int(response.get("fid"))
                    if not 0 <= feature_id < 54 or feature_id in purchased:
                        raise ValueError("invalid feature purchase")
                    charge = float(prices[feature_id])
                    if spent + charge > FEATURE_BUDGET + 1e-9:
                        raise ValueError("feature budget exceeded")
                    purchased.add(feature_id)
                    spent += charge
                    feature_pricer.on_purchase(feature_id)
                    prices = feature_pricer.published(row_index)
                    process.send({
                        "val": float(row[feature_id]),
                        "prices": {str(i): float(value) for i, value in enumerate(prices)},
                    })
                positive = prices[prices > 0]
                pinched = FEATURE_BUDGET - spent < float(np.min(positive))
                feature_pricer.on_tick(spent=spent, pinched=pinched)
        finally:
            process.close()
    metric = balanced_accuracy(y_test, predictions)
    reward = reward_for(metric)
    recalls = [float(np.mean(predictions[y_test == c] == c)) for c in range(N_CLASSES)]
    return {
        "valid": True,
        "metric_name": "balanced_accuracy",
        "metric": metric,
        "balanced_accuracy": metric,
        "reward": reward,
        "label_purchases": len(labeled),
        "test_cases": int(len(y_test)),
        "per_class_recall": recalls,
        "case_scores": (predictions == y_test).astype(int).tolist(),
    }
