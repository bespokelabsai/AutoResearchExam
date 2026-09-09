import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grader_core as G

LOG_DIR = Path("/logs/verifier")
METRIC_PATH = LOG_DIR / "metric.json"

_LAUNCH_LOCK = threading.Lock()


def _privilege_drop(cmd):
    """Prefix the command so it runs as uid 1001, whichever tool this image provides."""
    runuser = "/usr/sbin/runuser" if os.path.exists("/usr/sbin/runuser") else shutil.which("runuser")
    if runuser:
        return [runuser, "-u", "agent", "--"] + cmd, {}
    return cmd, {"user": G.AGENT_UID, "group": G.AGENT_GID}


def run_candidate(seed):
    """Train with the submitted estimator, retrying once if the run died from a signal.

    A signal death (status >= 128, e.g. a MuJoCo segfault under load) is an infrastructure
    failure of the simulator, not a contract violation: the harness reports contract breaks as
    its own exit statuses 2, 3 and 4, which are never retried. One retry, then the run counts as
    failed and the whole submission is invalid.
    """
    record = _run_candidate_once(seed)
    if isinstance(record["status"], int) and record["status"] >= 128:
        first = {"status": record["status"], "seconds": round(record["seconds"], 1)}
        record = _run_candidate_once(seed)
        record["retried_after_signal"] = first
    return record


def _run_candidate_once(seed):
    """One training run, as uid 1001, capturing only a checkpoint.

    The checkpoint descriptor is opened here, on a root-only directory, and inherited by the
    candidate. Nothing the candidate can reach afterwards is able to rewrite it.
    """
    work = G.candidate_work_dir(seed)
    ckpt = G.checkpoint_path(seed)
    log_path = G.run_log_path(seed)
    record = {"seed": seed, "status": None, "seconds": None, "stderr_tail": ""}
    started = time.monotonic()
    with open(ckpt, "wb") as sink, open(log_path, "w", encoding="utf-8") as log:
        os.chmod(ckpt, 0o600)
        cmd, extra = _privilege_drop(G.train_command(seed, sink.fileno()))
        with _LAUNCH_LOCK:
            proc = subprocess.Popen(
                cmd,
                cwd=work,
                env=G.candidate_env(work),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(sink.fileno(),),
                **extra,
            )
        try:
            proc.wait(timeout=G.RUN_WALL_CLOCK_CAP + G.RUN_KILL_GRACE)
            record["status"] = proc.returncode
        except subprocess.TimeoutExpired:
            record["status"] = "killed-at-cap"
        _kill_group(proc)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            pass
    record["stderr_tail"] = G.read_log_tail(log_path)
    record["seconds"] = time.monotonic() - started
    record["ckpt_bytes"] = os.path.getsize(ckpt) if os.path.exists(ckpt) else 0
    record["ok"] = record["status"] == 0 and record["ckpt_bytes"] > 0
    return record


def _kill_group(proc):
    """SIGKILL the whole session, so no descendant of the candidate outlives its run."""
    try:
        os.killpg(os.getpgid(proc.pid), 9)
    except OSError:
        try:
            proc.kill()
        except OSError:
            pass


def grade():
    """Grade the submission. Never raises: every failure lands on the invalid path, reward 0."""
    record = {"valid": False, "metric": None, "reward": 0.0, "invalid_reason": None,
              "runs": [], "per_seed_mean_return": {}, "episode_returns": {}}
    try:
        train_seeds, eval_seeds = G.load_seeds()
        record["num_train_seeds"] = len(train_seeds)
        record["num_eval_episodes_per_seed"] = len(eval_seeds)

        if not G.submission_entry_present():
            record["invalid_reason"] = (
                f"{os.path.join(G.SUBMISSION_SRC, G.ENTRY_NAME)} is missing or empty"
            )
            return record

        G.stage()
        runs = []
        for start in range(0, len(train_seeds), G.CONCURRENT_RUNS):
            batch = train_seeds[start : start + G.CONCURRENT_RUNS]
            with ThreadPoolExecutor(max_workers=len(batch)) as pool:
                runs.extend(pool.map(run_candidate, batch))
        for index, r in enumerate(runs, start=1):
            r["label"] = G.run_label(r["seed"], index)
            r["stderr_tail"] = G.redact_seeds(r["stderr_tail"], train_seeds)
            if G.PUBLIC_STDOUT:
                del r["seed"]
        record["runs"] = runs

        failed = [r for r in runs if not r["ok"]]
        if failed:
            record["invalid_reason"] = (
                f"{len(failed)} of {len(runs)} training runs did not complete "
                f"(first: {failed[0]['label']}, status {failed[0]['status']})"
            )
            return record

        all_returns = []
        for r, seed in zip(runs, train_seeds):
            returns = G.evaluate(G.checkpoint_path(seed), eval_seeds)
            record["episode_returns"][r["label"]] = returns
            record["per_seed_mean_return"][r["label"]] = sum(returns) / len(returns)
            all_returns.extend(returns)

        record["metric"] = sum(all_returns) / len(all_returns)
        record["num_episodes"] = len(all_returns)
        record["valid"] = True
        record["reward"] = G.graded_reward(record["metric"], True)
        return record
    except BaseException as exc:
        record["valid"] = False
        record["metric"] = None
        record["reward"] = 0.0
        record["invalid_reason"] = f"{type(exc).__name__}: {exc}"
        return record
    finally:
        G.cleanup()


@pytest.fixture(scope="session")
def graded():
    """Grade once and publish the record. grade() never raises; it fails closed."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    record = grade()
    METRIC_PATH.write_text(json.dumps(record, indent=1), encoding="utf-8")
    return record


def test_sealed_slice_is_root_only():
    """The evaluation seeds must be unreadable by the uid that runs the submitted estimator."""
    seal = G.seal_check()
    assert seal["sealed"], f"hidden_data is not sealed: {seal}"


def test_every_training_run_completed(graded):
    """Every sealed training run must finish under its wall-clock cap with a usable checkpoint."""
    assert graded["runs"], f"no training run was launched: {graded['invalid_reason']}"
    bad = [(r["label"], r["status"], r["stderr_tail"][-200:]) for r in graded["runs"] if not r["ok"]]
    assert not bad, f"training runs failed: {bad}"


def test_policies_evaluated_on_every_sealed_episode(graded):
    """The metric must come from the full sealed evaluation set, not a subset of it."""
    assert graded["valid"], f"grading did not complete: {graded['invalid_reason']}"
    expected = graded["num_train_seeds"] * graded["num_eval_episodes_per_seed"]
    assert graded["num_episodes"] == expected, (
        f"scored {graded['num_episodes']} episodes, expected {expected}"
    )


def test_metric_and_reward_recorded(graded):
    """metric.json must carry a finite metric and a reward inside [0, 1]."""
    assert METRIC_PATH.is_file(), "metric.json was not written"
    assert graded["metric"] is not None and math.isfinite(graded["metric"])
    assert 0.0 <= graded["reward"] <= 1.0, f"reward out of range: {graded['reward']}"
    if G.PUBLIC_STDOUT:
        print(
            f"mean episodic return {graded['metric']:.1f} over {graded['num_episodes']} episodes"
        )
    else:
        print(
            f"mean episodic return {graded['metric']:.1f} over {graded['num_episodes']} episodes; "
            "per-seed means "
            + ", ".join(f"{k}={v:.0f}" for k, v in sorted(graded["per_seed_mean_return"].items()))
        )
