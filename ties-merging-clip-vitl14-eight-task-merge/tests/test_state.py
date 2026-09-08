from __future__ import annotations

import json
import math
import os
import shutil
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_assets as ba
import grader_core as gc

PANEL = os.environ.get("PDK_PANEL", "final")
LOGS = Path("/logs/verifier")
DELIVERABLE = Path("/app/output")
ENTRY_NAME = "merge.py"
STAGE_ROOT = Path("/tmp/pdk_candidate")
AGENT_UID = 1001
AGENT_GID = 1001
RUNNER_SRC = Path(__file__).resolve().parent / "run_merge.py"
REAP_GRACE_SEC = 60.0

_RESULT: dict | None = None




def _write_metric(payload: dict) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "metric.json").write_text(json.dumps(payload, indent=1))
    print("metric.json:", json.dumps({k: v for k, v in payload.items()
                                      if k != "candidate_stderr"}))


def _invalid(reason: str, **extra) -> dict:
    payload = {"panel": PANEL, "valid": False, "failure_reason": reason,
               "metric": None, "per_task_accuracy": None, "reward": 0.0}
    payload.update(extra)
    _write_metric(payload)
    return payload


def scan_submission(root: Path) -> str | None:
    """Reject a submitted tree containing any symlink or any hard-linked file.

    A privileged copy that followed a submitted symlink would materialize verifier files into
    the directory the score is computed from; harbor preserves symlinks across the artifact
    hop, and a read-only bind mount does not remove hard links already present.
    """
    if not root.exists():
        return "missing_deliverable_directory"
    stack = [root]
    seen = 0
    while stack:
        d = stack.pop()
        try:
            entries = list(os.scandir(d))
        except OSError as exc:
            return f"unreadable_directory:{exc.strerror}"
        for e in entries:
            st = os.lstat(e.path)
            rel = os.path.relpath(e.path, root)
            if stat.S_ISLNK(st.st_mode):
                return f"symlink_in_submission:{rel}"
            if stat.S_ISDIR(st.st_mode):
                stack.append(Path(e.path))
                continue
            if not stat.S_ISREG(st.st_mode):
                return f"non_regular_file_in_submission:{rel}"
            if st.st_nlink != 1:
                return f"hard_link_in_submission:{rel}"
            seen += 1
            if seen > 20000:
                return "too_many_files_in_submission"
    return None


def check_surface(root: Path) -> str | None:
    """Enforce the file list instruction.md declares a solution may change.

    The deliverable directory is the whole of it, and its one required member is a non-empty
    regular `merge.py` directly inside it -- not in a subdirectory, not a symlink to elsewhere.
    Extra files and nested directories under the same root are permitted and travel with it;
    nothing outside it is ever read, so nothing outside it needs rejecting.
    """
    entry = root / ENTRY_NAME
    try:
        st = os.lstat(entry)
    except OSError:
        return f"missing_entry_point:{ENTRY_NAME}"
    if not stat.S_ISREG(st.st_mode):
        return f"entry_point_not_a_regular_file:{ENTRY_NAME}"
    if st.st_size == 0:
        return f"empty_entry_point:{ENTRY_NAME}"
    return None


def stage(root: Path) -> tuple[Path, Path, Path]:
    """Fresh, root-owned read-only copy of the submission plus writable scratch for uid 1001."""
    if STAGE_ROOT.exists():
        shutil.rmtree(STAGE_ROOT)
    STAGE_ROOT.mkdir(parents=True)
    os.chmod(STAGE_ROOT, 0o755)
    solution = STAGE_ROOT / "solution"
    shutil.copytree(root, solution, symlinks=True)
    for p in [solution, *solution.rglob("*")]:
        os.chmod(p, 0o555 if p.is_dir() else 0o444)
    runner = STAGE_ROOT / "run_merge.py"
    shutil.copyfile(RUNNER_SRC, runner)
    os.chmod(runner, 0o444)
    for name in ("out", "work"):
        d = STAGE_ROOT / name
        d.mkdir()
        os.lchown(d, AGENT_UID, AGENT_GID)
        os.chmod(d, 0o700)
    return solution, runner, STAGE_ROOT


def candidate_env() -> dict[str, str]:
    env = {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME": "/home/agent",
        "LANG": "C.UTF-8",
        "PYTHONPATH": "",
        "PYTHONSAFEPATH": "1",
    }
    env.update(gc.determinism_env())
    for k, v in os.environ.items():
        if k.startswith("NVIDIA_") or k in ("LD_LIBRARY_PATH", "CUDA_HOME", "CUDA_PATH"):
            env[k] = v
    return env




def grade() -> dict:
    global _RESULT
    if _RESULT is not None:
        return _RESULT
    try:
        _RESULT = _grade()
    except BaseException as exc:
        import traceback

        traceback.print_exc()
        _RESULT = _invalid(f"verifier_error:{exc!r}")
    return _RESULT


def _grade() -> dict:
    gc.pin_determinism()

    try:
        audit = gc.audit_panel(PANEL)
    except BaseException as exc:
        return _invalid(f"verifier_split_audit_failed:{exc!r}")

    rejected = scan_submission(DELIVERABLE)
    if rejected is not None:
        return _invalid(rejected)

    off_surface = check_surface(DELIVERABLE)
    if off_surface is not None:
        return _invalid(off_surface)

    solution, runner, root = stage(DELIVERABLE)
    argv = ["setsid", "runuser", "-u", "agent", "--", sys.executable, str(runner),
            "--solution", str(solution),
            "--unlabeled", str(gc.EVAL_ROOT / PANEL),
            "--assets", str(gc.ASSETS),
            "--out", str(root / "out"),
            "--budget", repr(gc.MERGE_BUDGET_SEC),
            "--device", "cuda"]
    hard_cap = gc.MERGE_BUDGET_SEC + gc.OVERHEAD_ALLOWANCE_SEC
    t0 = time.monotonic()
    try:
        proc = subprocess.Popen(argv, cwd=str(root / "work"), env=candidate_env(),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, errors="replace", start_new_session=True)
    except BaseException as exc:
        return _invalid(f"candidate_launch_failed:{exc!r}",
                        verifier_cuda_devices=_root_device_count())
    try:
        out, _ = proc.communicate(timeout=hard_cap)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except OSError:
            pass
        try:
            out, _ = proc.communicate(timeout=REAP_GRACE_SEC)
        except (subprocess.TimeoutExpired, OSError):
            proc.kill()
            out = None
        rc, out = -9, (out or "") + "\n[verifier] candidate killed at the hard cap\n"
    wall = time.monotonic() - t0

    tail = (out or "")[-4000:]
    report = {}
    rp = root / "out" / "report.json"
    if rp.is_file():
        try:
            report = json.loads(rp.read_text())
        except Exception:
            report = {"status": "unparseable_report"}
    diag = {"candidate_returncode": rc, "candidate_wall_seconds": round(wall, 2),
            "candidate_report": report, "candidate_stderr": tail,
            "verifier_cuda_devices": _root_device_count()}

    artifact = root / "out" / "merged.safetensors"
    if rc != 0 or not artifact.is_file() or artifact.stat().st_size == 0:
        reason = report.get("reason") if isinstance(report, dict) else None
        if rc == -9:
            reason = "hard_cap_exceeded"
        return _invalid(f"deliverable_failed:{reason or 'no_artifact'}", **diag)

    if not isinstance(report, dict) or report.get("status") != "ok":
        return _invalid("runner_report_missing_or_unparseable", **diag)

    reported = report.get("merge_seconds")
    if isinstance(reported, bool) or not isinstance(reported, (int, float)) or \
            not math.isfinite(float(reported)):
        return _invalid("runner_report_missing_merge_timing", **diag)
    if float(reported) > gc.MERGE_BUDGET_SEC:
        return _invalid("merge_budget_exceeded", **diag)

    from safetensors.torch import load_file

    try:
        candidate = load_file(str(artifact))
    except Exception as exc:
        return _invalid(f"artifact_unreadable:{exc!r}", **diag)

    reference = gc.reference_state_dict()
    bad = gc.validate_state_dict(candidate, reference)
    if bad is not None:
        return _invalid(f"invalid_state_dict:{bad}", **diag)

    scored = gc.evaluate(candidate, PANEL, device="cuda")
    metric = float(scored["metric"])
    reward = float(gc.graded_reward(metric, True))
    payload = {"panel": PANEL, "valid": True, "failure_reason": None,
               "metric": metric, "per_task_accuracy": scored["per_task_accuracy"],
               "panel_rows": scored["panel_rows"], "reward": reward,
               "split_audit": {t: audit[t]["panel_rows"] for t in gc.TASKS}}
    payload.update(diag)
    _write_metric(payload)
    return payload


def _root_device_count() -> int:
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:
        return -1




def test_sealed_labels_are_root_only():
    """The sealed labels must be unreachable from the uid that runs the deliverable."""
    st = os.stat("/tests/hidden_data")
    assert st.st_uid == 0, "hidden_data must be root-owned"
    assert stat.S_IMODE(st.st_mode) == 0o700, oct(stat.S_IMODE(st.st_mode))
    assert not (gc.EVAL_ROOT / PANEL / "sun397" / "labels.npy").exists(), (
        "labels must never sit beside the unlabeled panel images")


def test_grading_inputs_are_outside_the_writable_surface():
    """Everything the scorer reads, apart from the declared `/app/output` surface, is root-owned
    and unwritable to uid 1001: the sealed /tests tree with the trusted runner and grading code
    in it, the checkpoints, and the sealed panel images. The candidate's environment adds nothing
    importable of its own -- no inherited PYTHONPATH, `-P` semantics on -- so only the staged copy
    of the declared surface can be imported, and bytecode caching is off so no import can write a
    `__pycache__` back into that surface."""
    seal = stat.S_IMODE(os.stat("/tests").st_mode)
    assert not (seal & 0o077), f"/tests must stay sealed to root: {oct(seal)}"
    for p in ("/tests", str(RUNNER_SRC), "/tests/grader_core.py", str(gc.ASSETS),
              str(gc.EVAL_ROOT)):
        st = os.stat(p)
        assert st.st_uid == 0, f"{p} must be root-owned"
        assert not (stat.S_IMODE(st.st_mode) & 0o022), f"{p} is group/other writable"

    env = candidate_env()
    assert env["PYTHONPATH"] == ""
    assert env["PYTHONSAFEPATH"] == "1"
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"


def test_split_logic_rejects_duplicate_rows():
    """The shipped split logic: the three parity partitions are disjoint and cover the split, a
    duplicated row that lands in a sealed panel while its twin is in dev is caught by the
    content key, and the 480-bit dhash audit separates a near duplicate from an unrelated row.
    Unique fixtures cannot catch grouped leakage, so this fixture deliberately contains an exact
    duplicate and a near duplicate of rows on the other side of the split."""
    panels = ba.panel_indices(40, 1)
    assert not (set(panels["dev"]) & set(panels["intermediate"]))
    assert not (set(panels["dev"]) & set(panels["final"]))
    assert not (set(panels["intermediate"]) & set(panels["final"]))
    assert sorted(set(panels["dev"]) | set(panels["intermediate"]) | set(panels["final"])) == \
        list(range(40))

    rng = np.random.default_rng(0)
    base = (rng.random((6, 3, 224, 224)) * 255).astype(np.uint8)
    dev_keys = {ba.content_key(base[i]) for i in range(3)}
    dev_h = np.stack([ba.dhash(base[i]) for i in range(3)])
    exact = base[0].copy()
    near = base[1].copy()
    near[:, :2, :2] = 0
    fresh = base[4]
    assert ba.content_key(exact) in dev_keys, "an exact cross-split duplicate must be dropped"
    assert ba.content_key(near) not in dev_keys
    assert ba.content_key(fresh) not in dev_keys
    d = ba.min_hamming(np.stack([ba.dhash(exact), ba.dhash(near), ba.dhash(fresh)]), dev_h)
    assert d[0] == 0
    assert d[1] <= ba.NEAR_DUP_AUDIT_BITS, "the audit must see a near duplicate"
    assert d[2] > ba.NEAR_DUP_AUDIT_BITS, "an unrelated row must not read as a duplicate"


def test_shipped_panels_are_deduped_against_dev():
    """Fail closed on the split audit written when the panel was materialized."""
    audit = gc.audit_panel(PANEL)
    for task in gc.TASKS:
        assert audit[task]["exact_duplicates_remaining"] == 0
        assert audit[task]["panel_rows"] > 0
    print("split audit:", json.dumps({t: {k: audit[t][k] for k in (
        "panel_rows", "dropped_exact_duplicates_of_dev", "min_dhash_bits_to_dev",
        "sealed_rows_within_audit_bits_of_dev")} for t in gc.TASKS}))


def test_metric_json_written():
    """Grading always leaves a metric record, whatever the deliverable did."""
    grade()
    assert (LOGS / "metric.json").is_file()
    payload = json.loads((LOGS / "metric.json").read_text())
    assert set(payload) >= {"valid", "metric", "reward", "panel"}
    assert 0.0 <= float(payload["reward"]) <= 1.0


def test_deliverable_graded():
    """The submission ran, returned a valid CLIPVisionModel state dict, and was scored."""
    r = grade()
    if not r["valid"]:
        pytest.fail(f"deliverable not gradeable: {r['failure_reason']}")
    assert r["metric"] is not None
    assert set(r["per_task_accuracy"]) == set(gc.TASKS)
    for task, acc in r["per_task_accuracy"].items():
        assert 0.0 <= acc <= 100.0, (task, acc)


def test_trusted_runner_was_staged_out_of_tests():
    """uid 1001 is never pointed at a helper under the 0700 /tests seal. The root grader copies
    the runner into a fresh 0755 candidate temp parent, as a root-owned read-only file, and
    invokes that staged path -- so the seal is neither loosened nor bypassed."""
    grade()
    if not STAGE_ROOT.is_dir():
        pytest.skip("submission was rejected before staging")
    parent = os.stat(STAGE_ROOT)
    assert parent.st_uid == 0, "the staging parent must be root-owned"
    assert stat.S_IMODE(parent.st_mode) == 0o755, oct(stat.S_IMODE(parent.st_mode))

    staged = STAGE_ROOT / "run_merge.py"
    st = os.lstat(staged)
    assert stat.S_ISREG(st.st_mode), "the staged runner must be a regular file, not a link"
    assert st.st_uid == 0, "the staged runner must be root-owned"
    assert not (stat.S_IMODE(st.st_mode) & 0o222), oct(stat.S_IMODE(st.st_mode))
    assert staged.read_bytes() == RUNNER_SRC.read_bytes(), (
        "the staged runner must be a byte-identical copy of the trusted one")
    assert not str(staged).startswith("/tests")


def test_staged_surface_is_read_only_and_cache_free():
    """The declared surface, and only it, reaches the candidate -- as a root-owned read-only copy
    the candidate cannot mutate mid-run. PYTHONDONTWRITEBYTECODE keeps grade-time imports from
    depositing a `__pycache__` inside it; deleting caches before grading would not, because the
    grade-time import recreates them."""
    grade()
    solution = STAGE_ROOT / "solution"
    if not solution.is_dir():
        pytest.skip("submission was rejected before staging")
    assert (solution / ENTRY_NAME).is_file(), "the required entry point must have been staged"
    for p in [solution, *solution.rglob("*")]:
        st = os.lstat(p)
        assert st.st_uid == 0, f"{p} must be root-owned in the staged copy"
        assert not (stat.S_IMODE(st.st_mode) & 0o222), f"{p} is writable in the staged copy"
    submitted = {str(p.relative_to(DELIVERABLE)) for p in DELIVERABLE.rglob("__pycache__")}
    created = sorted({str(p.relative_to(solution))
                      for p in solution.rglob("__pycache__")} - submitted)
    assert not created, f"grade-time imports wrote bytecode caches into the surface: {created[:3]}"
