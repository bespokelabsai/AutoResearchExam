import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import grader_core

CHILD_PATH = "/usr/local/bin:/usr/local/sbin:/usr/bin:/usr/sbin:/bin:/sbin"


def _agent_uid_gid() -> tuple[int, int]:
    import pwd

    rec = pwd.getpwnam("agent")
    return rec.pw_uid, rec.pw_gid


def _reap_agent_processes() -> None:
    """Backstop: SIGKILL anything still running as uid `agent`.

    ``setsid`` may put the grandchild in a session of its own, so a process-group kill
    on the direct child is not by itself sufficient.  Implemented against /proc rather
    than ``pkill`` so it has no package dependency.
    """
    uid, _ = _agent_uid_gid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status = (entry / "status").read_text()
        except OSError:
            continue
        for line in status.splitlines():
            if line.startswith("Uid:"):
                if int(line.split()[1]) == uid:
                    try:
                        os.kill(int(entry.name), signal.SIGKILL)
                    except OSError:
                        pass
                break


def _run_one_seed(stage: Path, inputs: dict, seed: int, timeout: float) -> dict:
    """Execute the submission for one seed, then score its output array."""
    uid, gid = _agent_uid_gid()
    work = grader_core.WORK_DIR
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    os.chmod(work, 0o755)

    scratch = work / "scratch"
    scratch.mkdir()
    os.chown(scratch, uid, gid)
    os.chmod(scratch, 0o700)

    runner = work / "runner.py"
    shutil.copyfile(grader_core.RUNNER, runner)
    os.chmod(runner, 0o644)

    in_path = work / "inputs.npz"
    grader_core.write_inputs(in_path, inputs, seed)
    os.chmod(in_path, 0o644)
    out_path = scratch / "scores.npy"

    env = {
        "PATH": CHILD_PATH,
        "HOME": str(scratch),
        "TMPDIR": str(scratch),
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
        "NUMEXPR_NUM_THREADS": "8",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    cmd = [
        "setsid",
        "--wait",
        "runuser",
        "-u",
        "agent",
        "--",
        sys.executable,
        "-I",
        str(runner),
        str(in_path),
        str(out_path),
        str(stage),
    ]

    rec: dict = {"seed": seed}
    log = work / "child.log"
    t0 = time.monotonic()
    with open(log, "wb") as sink:
        proc = subprocess.Popen(
            cmd,
            cwd=str(scratch),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=sink,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            proc.communicate(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            _reap_agent_processes()
            try:
                proc.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
    rec["elapsed_sec"] = round(time.monotonic() - t0, 2)
    tail = log.read_bytes()[-grader_core.STDERR_TAIL_BYTES :].decode("utf-8", "replace")
    rec["stderr_tail"] = tail

    if timed_out:
        rec.update(status="timeout", auc=grader_core.FAILED_SEED_AUC)
        return rec
    if proc.returncode != 0:
        rec.update(status=f"nonzero_exit:{proc.returncode}",
                   auc=grader_core.FAILED_SEED_AUC)
        return rec

    status, auc = grader_core.score_output(out_path, inputs["eval_y"])
    rec.update(status=status, auc=auc)
    return rec


def _grade() -> dict:
    """Grade the submission over this half's pinned panel.  Never raises."""
    try:
        stage, info = grader_core.stage_deliverable()
    except grader_core.DeliverableError as exc:
        return grader_core.invalid_result(str(exc))
    except Exception as exc:
        return grader_core.invalid_result(f"staging failed: {type(exc).__name__}: {exc}")

    tables = grader_core.load_half()
    records: list[dict] = []
    spent = 0.0
    for seed in grader_core.SEEDS:
        remaining = grader_core.TOTAL_BUDGET_SEC - spent
        if remaining <= 1.0:
            records.append(
                {
                    "seed": seed,
                    "status": "total_budget_exhausted",
                    "auc": grader_core.FAILED_SEED_AUC,
                    "elapsed_sec": 0.0,
                }
            )
            continue
        inputs = grader_core.build_panel_inputs(tables, seed)
        rec = _run_one_seed(
            stage, inputs, seed, timeout=min(grader_core.SEED_TIMEOUT_SEC, remaining)
        )
        spent += float(rec.get("elapsed_sec", 0.0))
        records.append(rec)
    _reap_agent_processes()
    return grader_core.aggregate(records, info)


RESULT = _grade()
grader_core.write_metric(RESULT)


def test_metric_json_written():
    """The metric channel exists and carries a float reward, whatever happened.

    This is what lets every downstream path -- including the invalid one -- resolve to
    a number rather than to a missing file.
    """
    path = grader_core.LOGS_DIR / "metric.json"
    assert path.is_file(), "metric.json was not written"
    payload = json.loads(path.read_text())
    assert isinstance(payload.get("reward"), float)


def test_deliverable_is_usable():
    """/app/output/solution.py exists and is non-empty, the deliverable tree is inside
    the 131072-byte limit and contains no symlink or hard link, and at least one panel
    seed produced a usable score array.

    instruction.md states the entry-point path, the byte limit and the link
    prohibition, so every branch here is disclosed.
    """
    assert RESULT["valid"], RESULT.get("invalid_reason", "deliverable unusable")


def test_every_seed_returned_a_scoreable_array():
    """All 12 pinned seeds returned a finite numeric array of shape (5000,) within the
    100 s per-seed wall-clock cap.

    A failing seed is not fatal -- instruction.md states it contributes 0.5 ROC-AUC to
    the panel mean -- so this reports rather than gates.
    """
    assert RESULT["seeds_failed"] == 0, RESULT["failure_detail"]


def test_metric_and_reward_are_in_range():
    """The panel-mean ROC-AUC is in [0, 1]; the reward is in [0, 1] and is exactly 0
    whenever the metric fails to beat the trivial-baseline anchor."""
    m = RESULT["metric"]
    assert m is not None and 0.0 <= m <= 1.0, m
    r = RESULT["reward"]
    assert 0.0 <= r <= 1.0, r
    if m <= grader_core.M0:
        assert r == 0.0
