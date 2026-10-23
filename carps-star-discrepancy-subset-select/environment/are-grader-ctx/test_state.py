import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grader_core

PY = "/usr/local/bin/python3"
RUNNER = "/opt/verifier/runner.py"
RUNUSER = shutil.which("runuser") or "/usr/sbin/runuser"
TASKSET = shutil.which("taskset") or "/usr/bin/taskset"
ARTIFACT_DIR = Path("/app/output")
CANDIDATE_DIR = Path("/opt/candidate")
ENTRY_NAME = "select_subset.py"
HIDDEN_ROOT = Path("/tests/hidden_data")
METRIC_PATH = Path("/logs/verifier/metric.json")

TIME_BUDGET_S = 10.0
KILL_GRACE_S = 5.0
AGENT_SEEDS = (0, 1, 2)
AGENT_UID, AGENT_GID = 1001, 1001

SPLIT = os.environ.get("GRADE_SPLIT", "final")
VERBOSE = os.environ.get("GRADE_VERBOSE", "0") == "1"


def _harden_shared_paths() -> None:
    """Remove every agent-writable path that could survive from one graded run to the
    next. Each call gets a fresh scratch directory and nothing else, so a submission
    cannot pool budget across instances or replay cached work into a later run."""
    for path in ("/tmp", "/var/tmp", "/dev/shm"):
        try:
            os.chmod(path, 0o755)
        except OSError:
            pass
    for path in ("/home/agent", str(ARTIFACT_DIR)):
        try:
            os.chmod(path, 0o555)
        except OSError:
            pass


def _has_link(root: Path) -> str:
    """Reject a submitted tree containing symlinks or hard links, at any depth."""
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in dirnames + filenames:
            p = Path(dirpath) / name
            st = p.lstat()
            if os.path.islink(p):
                return f"symlink {p}"
            if not os.path.isdir(p) and st.st_nlink > 1:
                return f"hard link {p}"
    return ""


def stage_candidate() -> dict:
    """Copy the submitted deliverable to a root-owned directory, links refused."""
    info = {"found": False, "note": ""}
    if not ARTIFACT_DIR.is_dir():
        info["note"] = f"{ARTIFACT_DIR} is missing"
        return info
    link = _has_link(ARTIFACT_DIR)
    if link:
        info["note"] = f"submitted tree contains a {link}; refused"
        return info
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    for child in CANDIDATE_DIR.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
    shutil.copytree(ARTIFACT_DIR, CANDIDATE_DIR, symlinks=True, dirs_exist_ok=True)
    os.chown(CANDIDATE_DIR, 0, 0)
    os.chmod(CANDIDATE_DIR, 0o755)
    for dirpath, dirnames, filenames in os.walk(CANDIDATE_DIR):
        for name in dirnames + filenames:
            p = Path(dirpath) / name
            if os.path.islink(p):
                continue
            os.chown(p, 0, 0)
            os.chmod(p, 0o755 if p.is_dir() else 0o644)
    entry = CANDIDATE_DIR / ENTRY_NAME
    if not entry.is_file():
        info["note"] = f"{ARTIFACT_DIR / ENTRY_NAME} is missing"
        return info
    info["found"] = True
    return info


def _child_env(work: Path) -> dict:
    return {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": str(work),
        "TMPDIR": str(work),
        "PYTHONHASHSEED": "0",
        "PYTHONSAFEPATH": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }


def _read_result(path: Path):
    """Read the runner's output without following a symlink planted in its place."""
    fd = os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "r") as fh:
        return json.load(fh)


def run_one(points: np.ndarray, k: int, seed: int, cpu: int) -> dict:
    """Execute the candidate once, as uid 1001, on one CPU, under a hard deadline."""
    work = Path(tempfile.mkdtemp(prefix="grade_"))
    try:
        np.save(work / "points.npy", points)
        (work / "args.json").write_text(
            json.dumps({"k": int(k), "seed": int(seed), "time_budget_s": TIME_BUDGET_S})
        )
        for p in (work, work / "points.npy", work / "args.json"):
            os.chmod(p, 0o755 if p.is_dir() else 0o644)
        os.chown(work, AGENT_UID, AGENT_GID)

        cmd = [
            TASKSET, "-c", str(cpu),
            RUNUSER, "-u", "agent", "--",
            PY, RUNNER, str(work), str(CANDIDATE_DIR),
        ]
        hard = TIME_BUDGET_S + KILL_GRACE_S
        t0 = time.monotonic()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=_child_env(work),
            cwd=str(work),
            start_new_session=True,
        )
        killed = False
        try:
            _, err = proc.communicate(timeout=hard)
        except subprocess.TimeoutExpired:
            killed = True
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except OSError:
                pass
            try:
                _, err = proc.communicate(timeout=15)
            except subprocess.TimeoutExpired:
                err = b""
        wall = time.monotonic() - t0

        if killed:
            return {"valid": False, "error": "exceeded the deadline", "wall_s": wall}
        try:
            payload = _read_result(work / "result.json")
        except Exception as exc:
            tail = (err or b"").decode("utf-8", "replace").strip().splitlines()[-1:]
            reason = tail[0][:200] if tail else f"{type(exc).__name__}"
            return {"valid": False, "error": f"no usable result ({reason})", "wall_s": wall}
        if not isinstance(payload, dict) or not payload.get("ok"):
            reason = str(payload.get("error", "unknown"))[:200] if isinstance(payload, dict) else "malformed"
            return {"valid": False, "error": reason, "wall_s": wall}
        try:
            idx = grader_core.validate_indices(payload.get("result"), points.shape[0], k)
        except ValueError as exc:
            return {"valid": False, "error": str(exc)[:200], "wall_s": wall}
        try:
            call_s = float(payload.get("call_s"))
        except (TypeError, ValueError):
            call_s = float("nan")
        return {"valid": True, "indices": idx, "wall_s": wall, "call_s": call_s}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def grade() -> dict:
    out = {
        "split": SPLIT,
        "metric": None,
        "m0": None,
        "reward": 0.0,
        "valid_runs": 0,
        "total_runs": 0,
        "candidate_found": False,
        "time_budget_s": TIME_BUDGET_S,
        "kill_grace_s": KILL_GRACE_S,
        "agent_seeds": list(AGENT_SEEDS),
        "errors": [],
    }
    try:
        _harden_shared_paths()
        manifest = json.loads((HIDDEN_ROOT / SPLIT / "manifest.json").read_text())
        instances = []
        for entry in manifest["instances"]:
            pts = np.load(HIDDEN_ROOT / SPLIT / entry["file"])
            instances.append({**entry, "points": np.ascontiguousarray(pts, dtype=np.float64)})

        baseline = {}
        for inst in instances:
            bidx = grader_core.baseline_indices(inst["baseline_seed"], inst["n"], inst["k"])
            baseline[inst["iid"]] = grader_core.star_discrepancy_3d(inst["points"][bidx])

        info = stage_candidate()
        out["candidate_found"] = info["found"]
        if info["note"]:
            out["errors"].append(info["note"])

        jobs = [(inst, seed) for seed in AGENT_SEEDS for inst in instances]
        out["total_runs"] = len(jobs)
        m0 = float(np.mean([baseline[inst["iid"]] for inst, _ in jobs]))
        out["m0"] = m0
        cpus = sorted(os.sched_getaffinity(0))
        workers = max(1, min(8, len(cpus)))
        results = [None] * len(jobs)

        if info["found"]:
            def task(job_index):
                inst, seed = jobs[job_index]
                cpu = cpus[job_index % workers]
                return job_index, run_one(inst["points"], inst["k"], seed, cpu)

            with ThreadPoolExecutor(max_workers=workers) as pool:
                for job_index, res in pool.map(task, range(len(jobs))):
                    results[job_index] = res
        else:
            results = [{"valid": False, "error": info["note"] or "no deliverable"}] * len(jobs)

        per_run, per_k, errors, walls = [], defaultdict(list), defaultdict(int), []
        for (inst, seed), res in zip(jobs, results):
            if res and "wall_s" in res:
                walls.append(res["wall_s"])
            if res and res.get("valid"):
                value = grader_core.star_discrepancy_3d(inst["points"][res["indices"]])
                out["valid_runs"] += 1
            else:
                value = baseline[inst["iid"]]
                errors[(res or {}).get("error", "unknown")] += 1
            per_run.append(value)
            per_k[inst["k"]].append(value)
            if VERBOSE:
                print(f"  {inst['iid']} k={inst['k']} seed={seed} d*={value:.4f} "
                      f"valid={bool(res and res.get('valid'))}")

        if VERBOSE:
            out["per_run"] = [
                {"iid": inst["iid"], "k": inst["k"], "seed": seed, "d": value,
                 "baseline_d": baseline[inst["iid"]]}
                for (inst, seed), value in zip(jobs, per_run)
            ]
        out["metric"] = float(np.mean(per_run))
        out["per_k_mean"] = {str(k): float(np.mean(v)) for k, v in sorted(per_k.items())}
        if walls:
            out["wall_s"] = {"max": float(np.max(walls)), "mean": float(np.mean(walls))}
        out["reward"] = grader_core.graded_reward(out["metric"], m0)
        out["errors"].extend(f"{n}x {msg}" for msg, n in sorted(errors.items(), key=lambda kv: -kv[1])[:5])
    except Exception as exc:
        out["errors"].append(f"grader error: {type(exc).__name__}: {exc}"[:300])
    finally:
        METRIC_PATH.parent.mkdir(parents=True, exist_ok=True)
        METRIC_PATH.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    return out


@pytest.fixture(scope="session")
def outcome():
    result = grade()
    print(f"\nsplit: {result['split']}")
    print(f"runs: {result['valid_runs']}/{result['total_runs']} valid "
          f"({len(result['agent_seeds'])} seeds per instance, "
          f"{result['time_budget_s']:.1f}s budget per call)")
    if result["metric"] is not None:
        print(f"mean exact L_inf star discrepancy: {result['metric']:.6f}")
        for k, v in result.get("per_k_mean", {}).items():
            print(f"  k={k}: {v:.6f}")
    print(f"score: {result['reward']:.6f}")
    for msg in result["errors"]:
        print(f"note: {msg}")
    return result


def test_deliverable_present(outcome):
    """/app/output/select_subset.py exists and was stageable (no links in the tree)."""
    assert outcome["candidate_found"], outcome["errors"]


def test_all_runs_valid(outcome):
    """Every graded call returned k distinct in-range indices inside its deadline."""
    assert outcome["valid_runs"] == outcome["total_runs"], outcome["errors"]


def test_metric_computed(outcome):
    """A finite panel metric and a reward in [0,1] were produced."""
    assert outcome["metric"] is not None and np.isfinite(outcome["metric"])
    assert 0.0 <= outcome["reward"] <= 1.0


def test_beats_trivial_baseline(outcome):
    """The submitted selector is strictly better than an unguided random subset."""
    assert outcome["metric"] < outcome["m0"], (outcome["metric"], outcome["m0"])
