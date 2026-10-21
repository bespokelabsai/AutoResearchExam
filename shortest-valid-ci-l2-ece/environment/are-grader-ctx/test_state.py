from __future__ import annotations

import functools
import json
import multiprocessing as mp
import os
import pickle
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grader_core as gc

PY = sys.executable
RUN_CALL = "/opt/candrun/run_call.py"
ARTIFACT_DIR = Path("/app/output")
ENTRY_NAME = "interval.py"
STAGING = Path("/run/candidate_src")
SANDBOX_ROOT = Path("/var/tmp/candidate_sandbox")
METRIC_PATH = Path("/logs/verifier/metric.json")
AGENT_UID = 1001
AGENT_GID = 1001

HALF = os.environ.get("GRADE_SPLIT", "final")
HIDDEN_DIR = Path(os.environ.get("GRADER_HIDDEN_DIR",
                                 str(Path(__file__).resolve().parent / "hidden_data" / HALF)))
WORKERS = int(os.environ.get("GRADER_WORKERS", "8"))
GRADE_DEADLINE_SEC = float(os.environ.get("GRADER_DEADLINE_SEC", "1200"))
REPLICATIONS = int(os.environ.get("GRADER_REPLICATIONS", str(gc.REPLICATIONS)))
SEED_OFFSET = int(os.environ.get("GRADER_SEED_OFFSET", "0"))

CHILD_ENV_BASE = {
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "LC_ALL": "C",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "PYTHONSAFEPATH": "1",
}




def _reject_unsafe(root: Path) -> list[str]:
    """Every submitted path is hostile: refuse symlinks, hard links and specials.

    A privileged copy that followed a submitted symlink would materialise verifier
    files into a directory the score is computed from, so the whole tree is checked
    at every depth BEFORE anything is copied, and the copy itself never follows
    links (`symlinks=True`).
    """
    problems: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in list(dirnames) + list(filenames):
            p = Path(dirpath) / name
            st = p.lstat()
            if stat.S_ISLNK(st.st_mode):
                problems.append(f"symlink: {p}")
            elif stat.S_ISDIR(st.st_mode):
                continue
            elif not stat.S_ISREG(st.st_mode):
                problems.append(f"not a regular file: {p}")
            elif st.st_nlink > 1:
                problems.append(f"hard link: {p}")
    return problems


def stage_deliverable() -> dict:
    """Copy the deliverable to a root-owned read-only tree and seal the original."""
    info: dict = {"entry_point": str(ARTIFACT_DIR / ENTRY_NAME)}
    if STAGING.exists():
        shutil.rmtree(STAGING)
    if not ARTIFACT_DIR.is_dir():
        info["error"] = "missing_artifact_dir"
        return info
    problems = _reject_unsafe(ARTIFACT_DIR)
    if problems:
        info["error"] = "unsafe_paths"
        info["detail"] = problems[:5]
        return info

    shutil.copytree(ARTIFACT_DIR, STAGING, symlinks=True)
    entry = STAGING / ENTRY_NAME
    if not entry.is_file() or entry.stat().st_size == 0:
        info["error"] = "missing_or_empty_entry_point"
        info["deliverable_bytes"] = entry.stat().st_size if entry.exists() else 0
        return info
    info["deliverable_bytes"] = entry.stat().st_size

    for path in [STAGING, *STAGING.rglob("*")]:
        os.chown(path, 0, 0)
        os.chmod(path, 0o555 if path.is_dir() else 0o444)
    for path in [ARTIFACT_DIR, *ARTIFACT_DIR.rglob("*")]:
        try:
            os.chown(path, 0, 0)
            os.chmod(path, 0o555 if path.is_dir() else 0o444)
        except OSError:
            pass
    return info


def prepare_sandboxes(n: int) -> list[Path]:
    if SANDBOX_ROOT.exists():
        shutil.rmtree(SANDBOX_ROOT)
    SANDBOX_ROOT.mkdir(parents=True)
    os.chmod(SANDBOX_ROOT, 0o755)
    out = []
    for i in range(n):
        d = SANDBOX_ROOT / f"w{i}"
        d.mkdir()
        os.chown(d, AGENT_UID, AGENT_GID)
        os.chmod(d, 0o700)
        out.append(d)
    return out




@functools.lru_cache(maxsize=1)
def _launcher() -> list[str]:
    """Drop privileges with runuser; fall back to a setresuid preexec if absent."""
    for candidate in ("/usr/sbin/runuser", "/sbin/runuser", "/usr/bin/runuser"):
        if os.path.exists(candidate):
            return [candidate, "-u", "agent", "--"]
    found = shutil.which("runuser")
    return [found, "-u", "agent", "--"] if found else []


def _drop_privileges():
    os.setgroups([AGENT_GID])
    os.setresgid(AGENT_GID, AGENT_GID, AGENT_GID)
    os.setresuid(AGENT_UID, AGENT_UID, AGENT_UID)


def run_batch(calls: list[dict], sandbox: Path, wall_budget: float) -> dict:
    """Run one batch of candidate calls in a fresh unprivileged process."""
    blob = pickle.dumps(
        {
            "module_dir": str(STAGING),
            "module_file": str(STAGING / ENTRY_NAME),
            "cap": gc.PER_CALL_CAP_SEC,
            "calls": calls,
        },
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    env = dict(CHILD_ENV_BASE)
    env["HOME"] = str(sandbox)
    env["TMPDIR"] = str(sandbox)

    launcher = _launcher()
    argv = [*launcher, PY, RUN_CALL] if launcher else [PY, RUN_CALL]
    preexec = None if launcher else _drop_privileges
    proc = subprocess.Popen(
        argv,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(sandbox),
        env=env,
        start_new_session=True,
        preexec_fn=preexec,
    )
    try:
        out, _err = proc.communicate(input=blob, timeout=wall_budget)
        payload = json.loads(out.decode("utf-8", "replace")) if out.strip() else {}
    except subprocess.TimeoutExpired:
        payload = {"import_error": "batch_timeout"}
    except Exception as exc:
        payload = {"import_error": f"unparseable_child_output: {type(exc).__name__}"}
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), 9)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            proc.wait(timeout=10)
        except Exception:
            pass
        for child in sandbox.iterdir():
            shutil.rmtree(child, ignore_errors=True) if child.is_dir() else child.unlink(missing_ok=True)
    return payload




_WORKER_STATE: dict = {}


def _worker_init(settings, master_seed, sandbox_paths):
    idx = int(mp.current_process().name.rsplit("-", 1)[-1]) - 1
    _WORKER_STATE["settings"] = settings
    _WORKER_STATE["master_seed"] = master_seed
    _WORKER_STATE["sandbox"] = sandbox_paths[idx % len(sandbox_paths)]


def _worker_replication(rep: int):
    settings = _WORKER_STATE["settings"]
    master_seed = _WORKER_STATE["master_seed"]
    sandbox = _WORKER_STATE["sandbox"]
    calls = []
    thetas = {}
    for si, spec in enumerate(settings):
        Z, Y, theta, seed = gc.make_replication(spec, master_seed, si, rep)
        thetas[si] = theta
        calls.append(
            {"key": si, "Z": Z, "Y": Y, "k": int(spec["k"]), "alpha": gc.ALPHA, "seed": seed}
        )
    budget = gc.PER_CALL_CAP_SEC * len(calls) + 25.0
    payload = run_batch(calls, sandbox, budget)
    records = {}
    import_error = payload.get("import_error")
    by_key = {r.get("key"): r for r in payload.get("results", []) if isinstance(r, dict)}
    for si in range(len(settings)):
        rec = by_key.get(si)
        if import_error is not None or rec is None:
            records[si] = {"status": import_error or "no_result"}
        else:
            records[si] = rec
    return rep, records, thetas


def _grade() -> dict:
    started = time.monotonic()
    doc = gc.load_panel(HIDDEN_DIR)
    settings = doc["settings"]
    master_seed = int(doc["master_seed"]) + SEED_OFFSET

    staged = stage_deliverable()
    if "error" in staged:
        return {
            "status": "invalid",
            "reason": staged["error"],
            "detail": staged.get("detail"),
            "metric": None,
            "reward": 0.0,
            "half": HALF,
        }

    sandboxes = prepare_sandboxes(WORKERS)
    results: dict = {}
    thetas: dict = {}
    truncated_at = None
    ctx = mp.get_context("fork")
    with ctx.Pool(WORKERS, initializer=_worker_init, initargs=(settings, master_seed, sandboxes)) as pool:
        for rep, recs, ths in pool.imap_unordered(_worker_replication, range(REPLICATIONS), chunksize=4):
            for si, rec in recs.items():
                results[(si, rep)] = rec
            for si, th in ths.items():
                thetas[(si, rep)] = th
            if time.monotonic() - started > GRADE_DEADLINE_SEC:
                truncated_at = len(results)
                pool.terminate()
                break

    for si, spec in enumerate(settings):
        for rep in range(REPLICATIONS):
            if (si, rep) not in thetas:
                thetas[(si, rep)] = gc.theta_only(spec, master_seed, si, rep)
                results.setdefault((si, rep), {"status": "not_run"})

    scored = gc.score_panel(settings, thetas, results, REPLICATIONS)
    scored.update(
        status="graded",
        half=HALF,
        reward=gc.graded_reward(scored["metric"]),
        deliverable_bytes=staged.get("deliverable_bytes"),
        wall_clock_sec=round(time.monotonic() - started, 1),
        truncated_at=truncated_at,
        workers=WORKERS,
    )
    return scored


@functools.lru_cache(maxsize=1)
def grade_once() -> dict:
    METRIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = _grade()
    except BaseException as exc:
        result = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"[:300],
                  "metric": None, "reward": 0.0, "half": HALF}
    METRIC_PATH.write_text(json.dumps(result, indent=1, sort_keys=True))
    return result


def _summary(result: dict) -> str:
    lines = [f"half={result.get('half')} status={result.get('status')}"]
    if result.get("status") == "graded":
        for row in result["per_setting"]:
            lines.append(
                f"  setting {row['index']}: coverage={row['coverage']:.4f} "
                f"relative_length={row['relative_length']:.5f} "
                f"coverage_ok={row['coverage_gate_met']} score={row['score']:.5f}"
            )
        lines.append(f"  call outcomes: {result['status_counts']}")
        lines.append(f"  metric (mean relative length, lower is better) = {result['metric']:.6f}")
        lines.append(f"  wall clock {result.get('wall_clock_sec')}s over {result.get('workers')} workers")
    else:
        lines.append(f"  reason={result.get('reason')} detail={result.get('detail')}")
    return "\n".join(lines)




def test_deliverable_runs():
    """The submitted entry point must exist, import, and produce usable intervals.

    Measures real behaviour: it is satisfied only if calls returned intervals inside
    the disclosed contract, not by the file existing.
    """
    result = grade_once()
    print(_summary(result))
    assert result["status"] == "graded", f"deliverable not gradeable: {result.get('reason')}"
    ok = result["status_counts"].get("ok", 0)
    assert ok > 0, f"no call returned a valid interval: {result['status_counts']}"


def test_panel_metric_recorded():
    """The coverage-constrained mean relative length is recorded for the reward map."""
    result = grade_once()
    assert METRIC_PATH.is_file(), "metric.json was not written"
    assert result.get("metric") is not None, "no metric computed"
    print(f"metric={result['metric']:.6f}")


if __name__ == "__main__":
    res = grade_once()
    print(_summary(res))
    print(json.dumps({k: v for k, v in res.items() if k != "per_setting"}, indent=1))
