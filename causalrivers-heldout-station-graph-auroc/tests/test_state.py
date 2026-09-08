import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grader_core as gc

import numpy as np

PY = "/usr/local/bin/python3"
RUNUSER = "/usr/sbin/runuser"
RUNNER = "/opt/grade/run_batch.py"
AGENT_UID = 1001
AGENT_GID = 1001

DELIVERABLE = Path("/app/output")
ENTRY_NAME = "predict.py"
WORK = Path("/grade")
LOGS = Path("/logs/verifier")

BUDGET_SEC = 1200.0
MAX_DELIVERABLE_BYTES = 50 * 1024 * 1024
FREQ_MINUTES = 15

_RESULT = None


def ignore_special_files(directory, names):
    """Skip anything that is not a directory, a regular file or a symlink.

    A FIFO, socket or device node in the submission would otherwise be opened by copytree and
    could block the verifier forever. Nothing here opens a submitted path.
    """
    skipped = []
    for name in names:
        path = Path(directory) / name
        if path.is_symlink():
            continue
        if not (path.is_file() or path.is_dir()):
            skipped.append(name)
    return skipped


def measure_tree(root):
    """Total bytes of the regular files under `root`, never following a link."""
    total = 0
    for parent, _, files in os.walk(root, followlinks=False):
        for name in files:
            path = Path(parent) / name
            if not path.is_symlink() and path.is_file():
                total += path.lstat().st_size
    return total


def stage_deliverable():
    """Copy /app/output into the grading tree without ever following a submitted link.

    copytree(symlinks=True) recreates links instead of reading through them, every link in the
    copy is then unlinked, and regular files are copied byte-wise, so a hard link in the
    submission becomes an independent file here. Nothing under /tests can be pulled in. The
    size is measured on the SOURCE first, so an oversized submission is rejected before
    anything is copied.
    """
    dest = WORK / "solution"
    if dest.exists():
        shutil.rmtree(dest)
    if not DELIVERABLE.is_dir():
        return None, "deliverable directory /app/output is missing"
    total = measure_tree(DELIVERABLE)
    if total > MAX_DELIVERABLE_BYTES:
        return None, (f"deliverable is {total} bytes, over the "
                      f"{MAX_DELIVERABLE_BYTES} byte limit")
    shutil.copytree(DELIVERABLE, dest, symlinks=True, ignore_dangling_symlinks=True,
                    ignore=ignore_special_files)
    pruned = 0
    for root, dirs, files in os.walk(dest, followlinks=False):
        for name in list(dirs) + list(files):
            path = Path(root) / name
            if path.is_symlink():
                path.unlink()
                pruned += 1
    for root, dirs, files in os.walk(dest):
        os.chmod(root, 0o755)
        for name in files:
            os.chmod(Path(root) / name, 0o644)
    entry = dest / ENTRY_NAME
    if not entry.is_file():
        return None, f"no entry point at /app/output/{ENTRY_NAME}"
    if entry.stat().st_size == 0:
        return None, f"/app/output/{ENTRY_NAME} is empty"
    return {"path": str(entry), "bytes": total, "symlinks_pruned": pruned}, ""


def seal_shared_paths():
    """Remove every writable path that would otherwise survive from one batch to the next.

    /home/agent is in this list because runuser's PAM stack overrides the HOME we pass with the
    target user's own home (measured), so the fresh per-batch HOME is not what the candidate
    sees; re-owning the real one to root is what closes it.
    """
    for path in ("/tmp", "/var/tmp", "/dev/shm", "/home/agent"):
        try:
            os.chown(path, 0, 0)
            os.chmod(path, 0o755)
        except OSError:
            pass


def fresh_dir(path, uid=None):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    if uid is not None:
        os.chown(path, uid, AGENT_GID)
        os.chmod(path, 0o700)
    return path


def run_batch(index, batch, series, module_path, remaining, seed):
    """Execute one batch as uid 1001. Returns (matrices|None, seconds, reason)."""
    in_dir = WORK / "in"
    in_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(in_dir, 0o755)
    spec_path = in_dir / f"spec_{index:03d}.json"
    input_path = in_dir / f"input_{index:03d}.npy"
    out_dir = fresh_dir(WORK / "out" / f"b{index:03d}", uid=AGENT_UID)
    output_path = out_dir / "scores.npy"
    status_path = out_dir / "status.json"

    block = np.empty((len(batch), series.shape[1], gc.K_NODES), dtype=np.float32)
    for k, sample in enumerate(batch):
        block[k] = series[sample, :].T
    np.save(input_path, block)
    del block
    spec_path.write_text(json.dumps({"module": module_path,
                                     "freq_minutes": FREQ_MINUTES,
                                     "n_samples": len(batch),
                                     "seed": seed}))
    for path in (spec_path, input_path):
        os.chmod(path, 0o644)

    home = fresh_dir(WORK / "home" / f"b{index:03d}", uid=AGENT_UID)
    tmp = fresh_dir(WORK / "tmp" / f"b{index:03d}", uid=AGENT_UID)
    cwd = fresh_dir(WORK / "cwd" / f"b{index:03d}", uid=AGENT_UID)
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": str(home),
        "TMPDIR": str(tmp),
        "MPLCONFIGDIR": str(tmp),
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    cmd = [RUNUSER, "-u", "agent", "--", PY, "-s", "-B", RUNNER,
           str(spec_path), str(input_path), str(output_path), str(status_path)]

    started = time.monotonic()
    reason = ""
    proc = subprocess.Popen(
        cmd, cwd=str(cwd), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        _, err = proc.communicate(timeout=max(1.0, remaining))
        code = proc.returncode
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.communicate()
        code, err = -9, b""
        reason = "budget_exceeded"
    elapsed = time.monotonic() - started
    input_path.unlink(missing_ok=True)

    matrices = None
    if reason == "":
        if code != 0:
            reason = f"batch process exited {code}"
        elif not output_path.exists() or not status_path.exists():
            reason = "batch process produced no output"
        elif output_path.stat().st_size > 4096 + 4 * len(batch) * gc.K_NODES ** 2 * 8:
            reason = f"batch output is {output_path.stat().st_size} bytes, far over the "\
                     f"{len(batch)} x 5 x 5 float64 matrix set it must contain"
        else:
            try:
                loaded = np.load(output_path)
                status = json.loads(status_path.read_text()[:1 << 16])
                if (loaded.shape != (len(batch), gc.K_NODES, gc.K_NODES)
                        or not status.get("import_ok")):
                    reason = "batch output has the wrong shape"
                else:
                    matrices = loaded
            except Exception as exc:
                reason = f"batch output unreadable: {type(exc).__name__}"
    if reason and err:
        reason += " | " + err.decode("utf-8", "replace")[-600:]
    for path in (out_dir, home, tmp, cwd):
        shutil.rmtree(path, ignore_errors=True)
    return matrices, elapsed, reason


def grade():
    global _RESULT
    if _RESULT is not None:
        return _RESULT
    half = os.environ.get("GRADE_SPLIT", "final")
    panel = gc.load_half(half)
    batches, edges = panel["batches"], panel["edges"]
    LOGS.mkdir(parents=True, exist_ok=True)
    fresh_dir(WORK)
    seal_shared_paths()

    result = {"half": half, "valid": False, "invalid_reason": "", "deliverable": None,
              "metric": None, "reward": 0.0,
              "budget_sec": BUDGET_SEC, "candidate_seconds": 0.0,
              "n_batches": len(batches),
              "n_samples": sum(len(b) for b in batches)}

    info, why = stage_deliverable()
    result["deliverable"] = info
    if info is None:
        result["invalid_reason"] = why
        _RESULT = result
        write_metric(result)
        return result

    series = np.load(panel["series_path"], mmap_mode="r")
    if series.ndim != 2 or series.shape[0] != panel["station_count"]:
        raise RuntimeError(f"sealed series matrix is {series.shape}, expected "
                           f"({panel['station_count']}, T)")
    matrices, used = [], 0.0
    for index, batch in enumerate(batches):
        split_seed = 10_000 if half == "intermediate" else 20_000
        mats, elapsed, reason = run_batch(index, batch, series, info["path"],
                                          BUDGET_SEC - used, split_seed + index)
        used += elapsed
        matrices.append(mats)
        if reason:
            result["invalid_reason"] = f"batch {index}: {reason}"
            break
        if used > BUDGET_SEC:
            result["invalid_reason"] = (f"candidate used {used:.1f}s of the "
                                       f"{BUDGET_SEC:.0f}s budget")
            break
    result["candidate_seconds"] = round(used, 2)

    if not result["invalid_reason"]:
        result["valid"] = True
        while len(matrices) < len(batches):
            matrices.append(None)
        result["metric"] = gc.panel_metric(matrices, batches, edges)
        result["reward"] = gc.graded_reward(result["metric"]["mean_auroc"], True)
    shutil.rmtree(WORK / "in", ignore_errors=True)
    _RESULT = result
    write_metric(result)
    return result


def write_metric(result):
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "metric.json").write_text(json.dumps(result, indent=1))
    metric = result.get("metric") or {}
    print(f"half={result['half']} samples={result['n_samples']} "
          f"batches={result['n_batches']} valid={result['valid']}")
    print(f"candidate_seconds={result['candidate_seconds']} of {result['budget_sec']:.0f}")
    if metric:
        print(f"mean_auroc={metric['mean_auroc']:.4f} sem={metric['sem_auroc']:.4f} "
              f"scored_subgraphs={metric['n_samples']}")
        print(f"call_outcomes={json.dumps(metric['call_outcomes'], sort_keys=True)}")
    if result["invalid_reason"]:
        print(f"invalid_reason={result['invalid_reason']}")


def test_deliverable_runs():
    """The submitted entry point exists, imports and produces output for every batch."""
    result = grade()
    assert result["valid"], result["invalid_reason"]


def test_every_call_returned_a_usable_matrix():
    """Every graded subgraph got a (5, 5) finite score matrix (failures score exactly 0.0)."""
    result = grade()
    outcomes = (result.get("metric") or {}).get("call_outcomes", {})
    assert set(outcomes) <= {"ok"}, f"non-ok call outcomes: {outcomes}"


def test_metric_recorded():
    """The mean per-subgraph AUROC and the mapped reward are written for the reward step."""
    result = grade()
    assert (LOGS / "metric.json").is_file()
    assert 0.0 <= result["reward"] <= 1.0
