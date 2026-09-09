import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import grader_core
from grader_core import InvalidSubmission, graded_reward, map_at_k, validate_codes

TESTS = Path(__file__).resolve().parent
HIDDEN = TESTS / "hidden_data"
DELIVERABLE = Path("/app/output")
ENTRY_NAME = "hash_model.py"
METRIC_PATH = Path("/logs/verifier/metric.json")

RUN_BUDGET_SEC = 1200
DELIVERABLE_MAX_BYTES = 1024**3
N_DB_SHARDS = 3
PERMUTATION_SEED = 4242
AGENT_UID = AGENT_GID = 1001
PY = "/usr/local/bin/python3"
GRADE_SPLIT = os.environ.get("GRADE_SPLIT", "final")

_RESULT = {}


def _sealed_split():
    query_features = np.load(HIDDEN / GRADE_SPLIT / "query_features.npy")
    query_labels = np.load(HIDDEN / GRADE_SPLIT / "query_labels.npy")
    db_features = np.concatenate(
        [np.load(HIDDEN / "database" / f"db_features_{k}.npy") for k in range(N_DB_SHARDS)],
        axis=0,
    )
    db_labels = np.load(HIDDEN / "database" / "db_labels.npy")
    train_features = np.load(HIDDEN / "train" / "train_features.npy")
    train_labels = np.load(HIDDEN / "train" / "train_labels.npy")
    return query_features, query_labels, db_features, db_labels, train_features, train_labels


def _copy_deliverable(source, target):
    """Copy regular files only. Symlinks are dropped rather than followed, so a link aimed
    at the verifier tree can never be dereferenced by this root process."""
    if not source.is_dir():
        raise InvalidSubmission(f"{source} does not exist")
    target.mkdir(parents=True)
    total = 0
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if path.is_symlink():
            continue
        if path.is_dir():
            (target / relative).mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            total += path.stat().st_size
            if total > DELIVERABLE_MAX_BYTES:
                raise InvalidSubmission("deliverable exceeds 1 GiB")
            shutil.copyfile(path, target / relative)
    entry = target / ENTRY_NAME
    if not entry.is_file():
        raise InvalidSubmission(f"no {ENTRY_NAME} in {source}")
    if entry.stat().st_size == 0:
        raise InvalidSubmission(f"{ENTRY_NAME} is empty")
    return total


def _own(path, uid, mode_dir, mode_file):
    for current, _dirs, files in os.walk(path):
        os.chown(current, uid, uid)
        os.chmod(current, mode_dir)
        for name in files:
            os.chown(os.path.join(current, name), uid, uid)
            os.chmod(os.path.join(current, name), mode_file)


def _build_sandbox(inputs):
    sandbox = Path(tempfile.mkdtemp(prefix="grade_", dir="/tmp"))
    os.chmod(sandbox, 0o711)
    size = _copy_deliverable(DELIVERABLE, sandbox / "code")
    (sandbox / "inputs").mkdir()
    (sandbox / "out").mkdir()
    for name, array in inputs.items():
        np.save(sandbox / "inputs" / f"{name}.npy", array)
    os.chmod(sandbox / "inputs", 0o755)
    for path in (sandbox / "inputs").iterdir():
        os.chmod(path, 0o444)
    shutil.copyfile(TESTS / "runner.py", sandbox / "runner.py")
    os.chmod(sandbox / "runner.py", 0o444)
    _own(sandbox / "code", AGENT_UID, 0o700, 0o600)
    _own(sandbox / "out", AGENT_UID, 0o700, 0o600)
    return sandbox, size


def _reap_agent_processes():
    """Kill anything still running as the agent uid, including a process that escaped its
    group by starting its own session."""
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid == AGENT_UID:
                os.kill(int(entry.name), 9)
        except (ProcessLookupError, PermissionError, FileNotFoundError):
            pass


def _execute(sandbox):
    runuser = shutil.which("runuser") or "/usr/sbin/runuser"
    env_bin = shutil.which("env") or "/usr/bin/env"
    timeout_bin = shutil.which("timeout") or "/usr/bin/timeout"
    child_env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": str(sandbox / "code"),
        "TMPDIR": str(sandbox / "out"),
        "PYTHONSAFEPATH": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    command = [
        runuser, "-u", "agent", "--",
        env_bin, "-i", *[f"{k}={v}" for k, v in child_env.items()],
        timeout_bin, "-s", "KILL", str(RUN_BUDGET_SEC),
        PY, str(sandbox / "runner.py"),
    ]
    process = subprocess.Popen(
        command,
        cwd=str(sandbox / "out"),
        env={"PATH": "/usr/local/bin:/usr/bin:/bin"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        text=True,
        errors="replace",
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=RUN_BUDGET_SEC + 120)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(os.getpgid(process.pid), 9)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            stdout, stderr = process.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
    finally:
        _reap_agent_processes()
    return process.returncode, timed_out, stdout or "", stderr or ""


def _load_codes(path, n_rows, what):
    if not path.is_file():
        raise InvalidSubmission(f"{what}: no codes written")
    try:
        array = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise InvalidSubmission(f"{what}: unreadable array ({type(exc).__name__})") from exc
    return validate_codes(array, n_rows, what)


def graded_run():
    if _RESULT:
        return _RESULT

    record = {
        "half": GRADE_SPLIT,
        "metric_name": "mAP@5000",
        "metric": None,
        "reward": 0.0,
        "valid": False,
        "error": None,
        "executed": False,
        "timed_out": False,
        "returncode": None,
        "contract_ok": False,
        "permutation_equivariant": False,
        "deliverable_bytes": None,
        "run_seconds": None,
        "diagnostics": None,
    }
    sandbox = None
    try:
        query, query_labels, db, db_labels, train, train_labels = _sealed_split()
        permutation = np.random.default_rng(PERMUTATION_SEED).permutation(len(query))
        sandbox, record["deliverable_bytes"] = _build_sandbox(
            {
                "train_features": train,
                "train_labels": train_labels,
                "eval_a": query,
                "eval_b": db,
                "eval_c": query[permutation],
            }
        )
        code, timed_out, stdout, stderr = _execute(sandbox)
        record["returncode"], record["timed_out"] = code, timed_out
        if stderr.strip():
            print("--- deliverable stderr (tail) ---")
            print(stderr.strip()[-2000:])
        if timed_out or code == 137:
            raise InvalidSubmission(f"run exceeded the {RUN_BUDGET_SEC}s budget")

        status_path = sandbox / "out" / "status.json"
        if not status_path.is_file():
            raise InvalidSubmission(f"run produced no status (exit {code})")
        status = json.loads(status_path.read_text())
        record["executed"] = True
        record["run_seconds"] = status.get("seconds")
        if not status.get("ok"):
            raise InvalidSubmission(status.get("error") or f"run failed (exit {code})")

        codes_query = _load_codes(sandbox / "out" / "codes_a.npy", len(query), "query codes")
        codes_db = _load_codes(sandbox / "out" / "codes_b.npy", len(db), "database codes")
        codes_perm = _load_codes(
            sandbox / "out" / "codes_c.npy", len(query), "permuted query codes"
        )
        record["contract_ok"] = True
        if not np.array_equal(codes_perm, codes_query[permutation]):
            raise InvalidSubmission("encode is not permutation-equivariant")
        record["permutation_equivariant"] = True

        metric, diagnostics = map_at_k(codes_query, codes_db, query_labels, db_labels)
        record.update(
            metric=float(metric),
            diagnostics=diagnostics,
            valid=True,
            reward=float(graded_reward(metric, valid=True)),
        )
    except InvalidSubmission as exc:
        record["error"] = f"invalid submission: {exc}"
    except Exception as exc:
        record["error"] = f"grader error: {type(exc).__name__}: {exc}"
    finally:
        if sandbox is not None:
            shutil.rmtree(sandbox, ignore_errors=True)

    METRIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRIC_PATH.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"half={record['half']} metric[mAP@5000]={record['metric']} valid={record['valid']}")
    if record["diagnostics"]:
        print("diagnostics=" + json.dumps(record["diagnostics"], sort_keys=True))
    if record["error"]:
        print("error=" + record["error"])
    _RESULT.update(record)
    return _RESULT


def test_entry_point_executes():
    """/app/output/hash_model.py exists, imports, and completes train + encode inside the
    disclosed wall-clock budget."""
    result = graded_run()
    assert result["executed"], result["error"]
    assert not result["timed_out"], result["error"]


def test_code_contract():
    """Every encode call returns an (M, 16) int8 array whose values are all -1 or +1."""
    result = graded_run()
    assert result["contract_ok"], result["error"]


def test_permutation_equivariance():
    """Codes are a function of the rows, not of their order: encode(X[P]) == encode(X)[P]."""
    result = graded_run()
    assert result["permutation_equivariant"], result["error"]


def test_metric_recorded():
    """The scorer derived mAP@5000 from sealed labels and recorded a reward in [0, 1]."""
    result = graded_run()
    assert METRIC_PATH.is_file()
    assert 0.0 <= result["reward"] <= 1.0
    assert result["valid"], result["error"]
