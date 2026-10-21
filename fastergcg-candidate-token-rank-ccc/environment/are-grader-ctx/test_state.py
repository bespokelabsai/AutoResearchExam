from __future__ import annotations

import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grader_core

PANEL = os.environ.get("PANEL", "final")
TESTS_DIR = Path(__file__).resolve().parent
HIDDEN = TESTS_DIR / "hidden_data" / PANEL
RUNNER_SRC = TESTS_DIR / "candidate_runner.py"
SUBMISSION = Path("/app/output")
ENTRY_NAME = "ranker.py"
METRIC_PATH = Path("/logs/verifier/metric.json")

WRITABLE_SURFACE = ("/app/output",)
DEV_ONLY_PATHS = ("/app/gcg_states.py", "/app/data", "/app/work",
                  "/app/example_ranker.py", "/opt/assets/vicuna-7b-v1.5")
GRADE_TIME_ASSETS = ("/opt/assets/embedding_matrix.npy",
                     "/opt/assets/vicuna-tokenizer")

BUDGET_SEC = 300.0
HARD_KILL_SEC = 330.0
CANDIDATE_UID = 1001

SETSID = "/usr/bin/setsid"
TIMEOUT = "/usr/bin/timeout"
RUNUSER = "/usr/sbin/runuser"

CANDIDATE_ENV = {
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "HOME": "/home/agent",
    "LANG": "C.UTF-8",
    "PYTHONHASHSEED": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONPATH": "",
    "PYTHONSAFEPATH": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "CUDA_VISIBLE_DEVICES": "",
}


def _write_metric(payload: dict) -> None:
    METRIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRIC_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"metric.json: reward={payload.get('reward')} "
          f"metric={payload.get('metric')} reason={payload.get('reason')}",
          flush=True)


def _invalid(reason: str, **extra) -> dict:
    payload = {"reward": 0.0, "metric": None, "valid": False, "reason": reason,
               "panel": PANEL}
    payload.update(extra)
    _write_metric(payload)
    return payload


def scan_submission_tree(root: Path):
    """Every entry anywhere under root that is outside the declared surface.

    instruction.md allows exactly regular files and directories under
    `/app/output/`, at any depth. A symlink is rejected because a privileged
    copy that follows one materialises verifier files into the scored tree. A
    non-directory with st_nlink != 1 is rejected because a read-only bind mount
    does not remove hard links that were already there. Real directories
    legitimately have a link count above 1 and are exempt from that rule.
    Anything that is neither a regular file nor a directory -- a FIFO, socket or
    device node -- is rejected too: it is not on the declared surface, and
    copying one would block the grader instead of scoring it.
    """
    offences = []
    stack = [root]
    while stack:
        here = stack.pop()
        try:
            entries = sorted(os.scandir(here), key=lambda e: e.name)
        except OSError as exc:
            offences.append(f"unreadable {here}: {exc}")
            continue
        for entry in entries:
            path = Path(entry.path)
            st = os.lstat(path)
            if stat.S_ISLNK(st.st_mode):
                offences.append(f"symlink {path}")
                continue
            if stat.S_ISDIR(st.st_mode):
                stack.append(path)
                continue
            if not stat.S_ISREG(st.st_mode):
                offences.append(f"not a regular file (mode={oct(st.st_mode)}) "
                                f"{path}")
                continue
            if st.st_nlink != 1:
                offences.append(f"hard link (st_nlink={st.st_nlink}) {path}")
    return offences


def _stage(stage: Path):
    """Lay out the candidate sandbox and return (deliverable, features, out)."""
    stage.chmod(0o755)
    deliverable = stage / "deliverable"
    shutil.copytree(SUBMISSION, deliverable, symlinks=True)
    for path in [deliverable, *deliverable.rglob("*")]:
        os.lchown(path, CANDIDATE_UID, CANDIDATE_UID)
        if path.is_dir() and not path.is_symlink():
            path.chmod(0o755)
        elif not path.is_symlink():
            path.chmod(0o644)

    runner = stage / "runner.py"
    shutil.copy(RUNNER_SRC, runner)
    os.chown(runner, 0, 0)
    runner.chmod(0o444)

    features = stage / "features.npz"
    shutil.copy(HIDDEN / "features.npz", features)
    os.chown(features, 0, 0)
    features.chmod(0o444)

    out = stage / "out"
    out.mkdir()
    os.lchown(out, CANDIDATE_UID, CANDIDATE_UID)
    out.chmod(0o755)
    return deliverable, features, out, runner


def _run_candidate(runner: Path, features: Path, deliverable: Path, out: Path):
    """Run the submission as uid 1001 in its own session, under a wall clock."""
    cmd = [SETSID, TIMEOUT, "-k", "10", str(HARD_KILL_SEC),
           RUNUSER, "-u", "agent", "--",
           sys.executable, str(runner), str(features), str(deliverable),
           str(out)]
    proc = subprocess.Popen(cmd, cwd=str(out), env=CANDIDATE_ENV,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True)
    try:
        stdout, stderr = proc.communicate(timeout=HARD_KILL_SEC + 60.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()
        stdout, stderr = proc.communicate()
        return None, stdout, stderr
    return proc.returncode, stdout, stderr


def _grader_device_report() -> dict:
    """What the ROOT grader itself sees, so an all-zero rollout is legible."""
    report = {}
    try:
        import torch
        report["torch"] = torch.__version__
        report["cuda_available_to_grader"] = bool(torch.cuda.is_available())
        report["device_count_seen_by_grader"] = int(torch.cuda.device_count())
    except Exception as exc:
        report["torch_error"] = repr(exc)
    return report


def test_submission_scores():
    """Execute /app/output/ranker.py on the sealed panel and score its output.

    WHAT: runs the submitted ranker over every sealed snapshot as an
    unprivileged subprocess, then recomputes the concordance metric here from
    the returned scores and the sealed true loss changes.
    WHY: the reward must come from executing the agent's code against inputs it
    never saw, and it must be computed by a process the agent's code cannot
    reach.
    """
    stage = None
    try:
        entry = SUBMISSION / ENTRY_NAME
        if not SUBMISSION.is_dir():
            _invalid("missing_submission_directory")
            return
        offences = scan_submission_tree(SUBMISSION)
        if offences:
            _invalid("link_in_submission", offences=offences[:20])
            return
        if not entry.is_file() or entry.stat().st_size == 0:
            _invalid("missing_or_empty_entry_point",
                     deliverable_bytes=sum(
                         p.stat().st_size for p in SUBMISSION.rglob("*")
                         if p.is_file()))
            return

        stage = Path(tempfile.mkdtemp(prefix="candrun_", dir="/tmp"))
        deliverable, features, out, runner = _stage(stage)
        rc, stdout, stderr = _run_candidate(runner, features, deliverable, out)
        status = {}
        status_path = out / "status.json"
        if status_path.is_file():
            try:
                status = json.loads(status_path.read_text())
            except Exception:
                status = {"ok": False, "reason": "unparseable_status"}
        diag = {
            "returncode": rc,
            "runner_status": status,
            "stdout_tail": (stdout or "")[-2000:],
            "stderr_tail": (stderr or "")[-4000:],
            "grader_device_report": _grader_device_report(),
        }

        if rc is None:
            _invalid("wall_clock_exceeded", **diag)
            return
        if rc != 0:
            reason = {2: "entry_point_unusable", 3: "wall_clock_exceeded",
                      124: "wall_clock_exceeded", 137: "wall_clock_exceeded"}.get(
                          rc, "candidate_process_failed")
            _invalid(reason, **diag)
            return

        pred_path = out / "predictions.npy"
        if not pred_path.is_file():
            _invalid("no_predictions_written", **diag)
            return
        try:
            pred = np.asarray(np.load(pred_path))
        except Exception as exc:
            _invalid("unreadable_predictions", load_error=repr(exc), **diag)
            return
        truth = np.load(HIDDEN / "truth.npz")["true_delta"]
        if pred.dtype.kind not in "fiub" or pred.shape != truth.shape \
                or not np.all(np.isfinite(pred)):
            _invalid("unusable_predictions", pred_shape=list(pred.shape),
                     pred_dtype=str(pred.dtype), **diag)
            return
        pred = pred.astype(np.float64)

        metric = grader_core.pooled_ccc(pred, truth)
        if not np.isfinite(metric):
            _invalid("non_finite_metric", **diag)
            return
        reward = grader_core.graded_reward(metric, PANEL)
        reward = min(1.0, max(0.0, float(reward)))
        _write_metric({
            "reward": reward,
            "metric": metric,
            "metric_name": "pooled_rank_ccc",
            "valid": True,
            "reason": "ok",
            "panel": PANEL,
            "n_snapshots": int(truth.shape[0]),
            "n_pairs": int(truth.size),
            "fallback_calls": status.get("fallback_calls"),
            "candidate_elapsed_sec": status.get("elapsed_sec"),
            "anchors": dict(grader_core.ANCHORS[PANEL],
                            c=grader_core.DEGENERATE_REWARD),
            "diagnostics": diag,
        })
    except Exception:
        try:
            _invalid("grader_exception",
                     traceback=traceback.format_exc()[-4000:])
        except Exception:
            traceback.print_exc()
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)


def test_metric_equals_pearson_on_rank_blocks():
    """WHAT: the concordance metric equals Pearson on pooled rank vectors.
    WHY: every block is a permutation of 1..64, so the means and variances
    cancel; a reviewer can check the implementation against scipy, and this
    pins that equivalence in the shipped code.
    """
    from scipy.stats import pearsonr
    rng = np.random.default_rng(0)
    pred = rng.normal(size=(17, grader_core.N_CANDIDATES))
    truth = rng.normal(size=(17, grader_core.N_CANDIDATES))
    xs = np.concatenate([grader_core.ordinal_ranks(r) for r in pred])
    ys = np.concatenate([grader_core.ordinal_ranks(r) for r in truth])
    assert abs(grader_core.pooled_ccc(pred, truth)
               - float(pearsonr(xs, ys).statistic)) < 1e-12
    assert abs(grader_core.pooled_ccc(truth, truth) - 1.0) < 1e-12
    assert abs(grader_core.pooled_ccc(-truth, truth) + 1.0) < 1e-12


def test_reward_map_landmarks_and_bounds():
    """WHAT: the reward map's anchors, bounds and monotonicity.
    WHY: the clamp in compute_reward.py is the only bound in the stack, so the
    map itself must already be bounded, monotone and pinned at its anchors.
    """
    c = grader_core.DEGENERATE_REWARD
    for panel, a in grader_core.ANCHORS.items():
        gr = lambda m, panel=panel: grader_core.graded_reward(float(m), panel)
        assert a["m_floor"] < a["m0"] < a["baseline_score"], (panel, a)
        assert gr(a["m_floor"]) == 0.0
        assert gr(a["m_floor"] - 5.0) == 0.0
        assert abs(gr(a["m0"]) - c) < 1e-12
        assert abs(gr(a["baseline_score"]) - (c + (1.0 - c) * 0.5)) < 1e-12
        rewards = np.array([gr(m) for m in np.linspace(-1.0, 1.0, 4001)])
        assert rewards.min() >= 0.0 and rewards.max() < 1.0
        assert np.all(np.diff(rewards) >= -1e-15)
        assert gr(1.0) > gr(0.9) > gr(a["baseline_score"]) > gr(a["m0"]) > gr(a["m_floor"])


def test_reward_writer_is_total_and_bounded():
    """WHAT: compute_reward.py maps ANY metric.json payload into [0, 1].
    WHY: reward.txt is the only number the harness reads and nothing clamps it
    server-side, so the writer must be total: a NaN metric, an infinity from an
    overflowed division, a negative reward, a reward above 1, or a payload that
    is not a number at all must each still leave a finite, non-negative value.
    """
    import math

    import compute_reward

    hostile = [float("nan"), float("inf"), float("-inf"), -1e308, 1e308,
               -0.0, -1e-12, 0.0, 1e-300, 0.5, 1.0, 1.0 + 1e-9, 2.0, 10 ** 400,
               None, True, False, "", " ", "nan", "-inf", "0.42", "abc",
               [], {}, {"reward": 0.5}, object()]
    for raw in hostile:
        value = compute_reward.clamp01(raw)
        assert isinstance(value, float) and math.isfinite(value), (raw, value)
        assert 0.0 <= value <= 1.0, (raw, value)
        text = compute_reward.format_reward(value)
        assert not text.startswith("-"), (raw, text)
        back = float(text)
        assert math.isfinite(back) and 0.0 <= back <= 1.0, (raw, text)

    assert compute_reward.clamp01(float("nan")) == 0.0
    assert compute_reward.clamp01(float("-inf")) == 0.0
    assert compute_reward.clamp01(-5.0) == 0.0
    assert compute_reward.clamp01(2.0) == 1.0
    assert compute_reward.clamp01(0.25) == 0.25
    assert compute_reward.format_reward(0.0) == "0"
    for panel in grader_core.ANCHORS:
        for m in np.linspace(-1.0, 1.0, 401):
            r = compute_reward.clamp01(grader_core.graded_reward(float(m), panel))
            assert 0.0 <= float(compute_reward.format_reward(r)) <= 1.0


def test_sealed_panels_are_behaviour_disjoint():
    """WHAT: the two sealed panels share no behaviour, and the checker that
    says so actually rejects a shared key.
    WHY: snapshots of one trajectory are correlated, so a behaviour appearing in
    both panels would leak the privately graded split into the public one. A
    fixture with duplicated keys is included because disjoint fixtures cannot
    demonstrate that the guard fires.
    """
    keys = {}
    for name in ("final", "intermediate"):
        z = np.load(TESTS_DIR / "hidden_data" / name / "truth.npz")
        keys[name] = z["behaviour_row"].tolist()
        assert len(set(keys[name])) == 8
        assert len(keys[name]) == 80
    grader_core.assert_disjoint_behaviours(keys["final"], keys["intermediate"])
    dup_a = keys["final"] + [keys["intermediate"][0]] * 10
    try:
        grader_core.assert_disjoint_behaviours(dup_a, keys["intermediate"])
    except ValueError:
        pass
    else:
        raise AssertionError("the disjointness guard did not fire")


def test_only_the_declared_surface_is_graded():
    """WHAT: the graded surface is exactly the one instruction.md declares --
    `/app/output/`, at any depth, and nothing else in the image.
    WHY: the writable surface has to be enforced, not merely documented. This
    grader reads the submission from `/app/output` alone, so a solution can
    change any regular file under it and no edit anywhere else can reach the
    score. The development-only paths are asserted absent here, so a grader that
    ever started reading one would fail this test instead of silently scoring
    it, and the two runtime assets are asserted present because they sit outside
    the writable surface and are loaded by path.
    """
    assert [str(SUBMISSION)] == list(WRITABLE_SURFACE)
    assert ENTRY_NAME == "ranker.py"
    for path in DEV_ONLY_PATHS:
        assert not os.path.exists(path), f"dev-only path present at grade time: {path}"
    for path in GRADE_TIME_ASSETS:
        assert os.path.exists(path), f"missing grade-time asset: {path}"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "sub").mkdir()
        (root / "sub" / "ranker.py").write_text("x = 1\n")
        assert scan_submission_tree(root) == []
        os.symlink("/tests", root / "link_to_tests")
        os.link(root / "sub" / "ranker.py", root / "hardlink.py")
        os.mkfifo(root / "fifo")
        offences = " ".join(scan_submission_tree(root))
        assert "symlink" in offences and "hard link" in offences \
            and "not a regular file" in offences, offences


def test_hidden_data_is_root_sealed():
    """WHAT: /tests/hidden_data is root-owned and 0700 at grade time.
    WHY: the candidate subprocess runs as uid 1001; the ground truth must not be
    reachable from it even though the features are staged out for it.
    """
    st = os.stat(TESTS_DIR / "hidden_data")
    assert st.st_uid == 0, st.st_uid
    assert stat.S_IMODE(st.st_mode) == 0o700, oct(stat.S_IMODE(st.st_mode))
