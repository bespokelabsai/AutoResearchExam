from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import pretrain_harness as H

BUDGET_TOKENS = 125_000_000
SELECT_TIMEOUT_SEC = 2400
DELIVERABLE_DIR = "/app/output"
ENTRY_POINT = "select.py"
CANDIDATE_UID = 1001
CANDIDATE_GID = 1001

WRITABLE_SURFACE_ROOT = DELIVERABLE_DIR
AGENT_TREES_NEVER_READ = ("/app/harness", "/app/data", "/app/scratch")


def writable_surface_violations(config_path=None) -> list:
    """Every way this grader would read agent-editable state outside the writable surface.

    Empty is the only acceptable result at grade time, and test_state.py asserts exactly that.
    It is reported rather than raised from grade() so that the build-time baseline sweep, which
    grades stub selectors against its own scratch pools and its own config copy, can drive the
    same code path without tripping a check that only describes the graded image.
    """
    sealed = Path(__file__).resolve().parent
    bad = []
    for label, module_file in (("grader_core", __file__), ("pretrain_harness", H.__file__)):
        path = Path(module_file).resolve()
        if sealed not in path.parents:
            bad.append(f"{label} is loaded from {path}, outside the sealed tree {sealed}")
    if config_path is not None:
        cfg = Path(config_path).resolve()
        if sealed not in cfg.parents:
            bad.append(f"the training config is read from {cfg}, outside {sealed}")
    for tree in AGENT_TREES_NEVER_READ:
        if Path(tree).exists():
            bad.append(f"{tree} reached the verifier and could shadow the sealed copy")
    app = Path("/app")
    for entry in sys.path:
        if not entry:
            continue
        path = Path(entry).resolve()
        if path == app or app in path.parents:
            bad.append(f"an agent-writable path is on the grader's sys.path: {entry}")
    return bad


DEGENERATE_REWARD = 0.10
M_FLOOR = -4.01125
M0 = 0.25515
BASELINE_SCORE = 4.4


def graded_reward(metric: float, valid: bool) -> float:
    """rational_squash, higher-is-better, with the linear band below the trivial baseline."""
    if not valid:
        return 0.0
    x = metric - M0
    x_ref = BASELINE_SCORE - M0
    assert x_ref > 0, "malformed task: the anchor does not beat the trivial baseline"
    g = M0 - M_FLOOR
    if g <= 0:
        u = max(0.0, x) / x_ref
        return u / (1.0 + u)
    if x < 0:
        return DEGENERATE_REWARD * (max(0.0, metric - M_FLOOR) / g)
    u = x / x_ref
    return DEGENERATE_REWARD + (1.0 - DEGENERATE_REWARD) * (u / (1.0 + u))


class SubmissionRejected(Exception):
    pass


def scan_submission(src: Path) -> None:
    """Reject symlinks and hard links anywhere in the submitted tree, at any depth.

    A read-only bind mount does not remove hard links already in the submission, and a
    no-dereference copy does not make a directory symlink's contents safe -- so the tree
    is scanned and rejected BEFORE anything is copied.
    """
    if not src.is_dir() or src.is_symlink():
        raise SubmissionRejected(f"{src} is not a real directory")
    stack = [src]
    n_files = 0
    while stack:
        cur = stack.pop()
        with os.scandir(cur) as it:
            for entry in it:
                st = os.lstat(entry.path)
                if os.path.islink(entry.path):
                    raise SubmissionRejected(f"symlink in submission: {entry.path}")
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                    continue
                if st.st_nlink != 1:
                    raise SubmissionRejected(
                        f"hard link in submission (st_nlink={st.st_nlink}): {entry.path}")
                n_files += 1
                if n_files > 200_000:
                    raise SubmissionRejected("submission contains too many files")


def stage_submission(src: Path, dst: Path) -> None:
    """Copy the scanned tree without following links, then hand it to the candidate uid."""
    scan_submission(src)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True)
    for root, dirs, files in os.walk(dst):
        os.lchown(root, CANDIDATE_UID, CANDIDATE_GID)
        for name in dirs + files:
            os.lchown(os.path.join(root, name), CANDIDATE_UID, CANDIDATE_GID)
    os.lchown(dst, CANDIDATE_UID, CANDIDATE_GID)


POOL_FILES_FOR_CANDIDATE = ("docs.jsonl", "tokens.npy", "offsets.npy")


def stage_pool(pool_dir: Path, session: Path) -> Path:
    """Publish the pool to the candidate uid without opening the sealed tree.

    /tests is 0700 root at grade time, so a uid-1001 subprocess cannot traverse it -- and
    it must not, because the evaluation shard and this file live there. The three files
    instruction.md documents are copied into a traversable parent as root-owned 0444, so
    the candidate can read exactly the pool and cannot mutate the bytes the grader scores
    from; the grader keeps reading the originals under /tests. meta.json and guard.json
    are deliberately NOT copied -- guard.json carries the evaluation shard's document
    hashes and URLs.
    """
    dst = session / "pool"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    os.chmod(dst, 0o755)
    for name in POOL_FILES_FOR_CANDIDATE:
        src = Path(pool_dir) / name
        if not src.is_file():
            raise FileNotFoundError(f"the graded pool is missing {name}")
        shutil.copyfile(src, dst / name)
        os.chown(dst / name, 0, 0)
        os.chmod(dst / name, 0o444)
    return dst


def stage_runner(session: Path) -> Path:
    """Copy candidate_runner.py out of the sealed tree into a traversable parent, keeping
    it root-owned and read-only so the candidate can execute but never modify it."""
    session.mkdir(parents=True, exist_ok=True)
    os.chmod(session, 0o755)
    dst = session / "candidate_runner.py"
    shutil.copyfile(Path(__file__).resolve().parent / "candidate_runner.py", dst)
    os.chown(dst, 0, 0)
    os.chmod(dst, 0o444)
    return dst


def candidate_environment() -> dict:
    """The explicit environment the candidate subprocess runs under.

    The NVIDIA visibility variables are deliberately NOT narrowed -- hiding the device
    from the deliverable while the root grader still sees it is the hardest version of
    this bug to read from a rollout log.
    """
    env = {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME": "/home/agent",
        "LANG": "C.UTF-8",
        "TMPDIR": "/tmp",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONSAFEPATH": "1",
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": "/opt/assets/hf-home",
    }
    for key in ("NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES", "LD_LIBRARY_PATH"):
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def run_selector(deliverable_src: Path, pool_dir: Path, session: Path, log=print) -> dict:
    """Stage, launch and collect the submission.  Never raises for a candidate failure."""
    out = {"ok": False, "error": None, "raw": None, "elapsed_sec": None,
           "returncode": None, "stderr_tail": None}
    staged = session / "submission"
    workdir = session / "workdir"
    result_path = workdir / "selection.json"
    try:
        stage_submission(deliverable_src, staged)
    except SubmissionRejected as exc:
        out["error"] = f"submission rejected: {exc}"
        return out
    except Exception as exc:
        out["error"] = f"submission could not be staged: {type(exc).__name__}: {exc}"
        return out

    entry = staged / ENTRY_POINT
    if not entry.is_file():
        out["error"] = f"missing entry point {DELIVERABLE_DIR}/{ENTRY_POINT}"
        return out

    workdir.mkdir(parents=True, exist_ok=True)
    os.chmod(workdir, 0o755)
    os.chown(workdir, CANDIDATE_UID, CANDIDATE_GID)
    if result_path.exists():
        result_path.unlink()
    runner = stage_runner(session)
    try:
        candidate_pool = stage_pool(pool_dir, session)
    except Exception as exc:
        out["error"] = f"the pool could not be staged for the candidate: {exc}"
        return out

    cmd = ["runuser", "-u", "agent", "--", "/usr/bin/python3", str(runner),
           str(staged), str(candidate_pool), str(BUDGET_TOKENS), str(workdir), str(result_path)]
    log(f"[select] launching: {' '.join(cmd)}")
    started = time.monotonic()
    proc = None
    try:
        proc = subprocess.Popen(
            cmd, env=candidate_environment(), cwd=str(workdir),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True,
        )
    except Exception as exc:
        out["elapsed_sec"] = time.monotonic() - started
        out["error"] = f"could not launch the submission: {type(exc).__name__}: {exc}"
        _reap(None)
        return out
    try:
        stdout, _ = proc.communicate(timeout=SELECT_TIMEOUT_SEC)
        out["returncode"] = proc.returncode
        out["stderr_tail"] = (stdout or b"").decode("utf-8", "replace")[-4000:]
    except subprocess.TimeoutExpired:
        _reap(proc)
        try:
            stdout, _ = proc.communicate(timeout=60)
        except Exception:
            stdout = b""
        out["elapsed_sec"] = time.monotonic() - started
        out["error"] = f"select() exceeded its {SELECT_TIMEOUT_SEC}-second budget"
        out["stderr_tail"] = (stdout or b"").decode("utf-8", "replace")[-4000:]
        return out
    except Exception as exc:
        _reap(proc)
        out["elapsed_sec"] = time.monotonic() - started
        out["error"] = f"select() could not be collected: {type(exc).__name__}: {exc}"
        return out
    finally:
        _reap(proc)
    out["elapsed_sec"] = time.monotonic() - started

    if not result_path.is_file():
        out["error"] = (f"select() wrote no result (returncode={out['returncode']}); "
                        f"tail: {out['stderr_tail']}")
        return out
    try:
        blob = json.loads(result_path.read_text())
    except Exception as exc:
        out["error"] = f"select() result is unparseable: {type(exc).__name__}: {exc}"
        return out
    if not blob.get("ok"):
        out["error"] = f"select() raised: {blob.get('error')}"
        return out
    out["raw"] = blob.get("value")
    out["ok"] = True
    return out


def _reap(proc) -> None:
    """Kill the candidate's whole process group, then sweep any double-forked stragglers.

    start_new_session=True made the child a session/process-group leader (setsid(2)), so
    its pgid equals its pid and one killpg reaches every descendant that stayed in the
    group; the uid sweep is the backstop for anything that left it.
    """
    if proc is not None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
    try:
        subprocess.run(["pkill", "-9", "-u", str(CANDIDATE_UID)], timeout=30,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def reference_perplexity(cfg, pool, eval_blocks, cache_dir=None, log=print) -> dict:
    """The full-pool reference run.

    `cache_dir` is None at grade time, where the reference is computed exactly once per
    invocation anyway, so no cache and no cache-shaped attack surface exists there.  The
    build-time baseline sweep passes a root-owned scratch directory so that N selectors
    graded against one pool pay for one reference run instead of N; the key covers the
    pool identity and every training hyperparameter.
    """
    if cache_dir is None:
        return H.run_cycle(cfg, pool, list(range(pool.n_docs)), eval_blocks, log=log)
    key = hashlib.sha256(
        (json.dumps(cfg, sort_keys=True) + "|" + str(Path(pool.dir).resolve()) + "|"
         + str(pool.n_docs) + "|" + str(int(pool.offsets[-1])) + "|"
         + str(int(eval_blocks.shape[0]))).encode("utf-8")).hexdigest()[:24]
    path = Path(cache_dir) / f"reference-{key}.json"
    if path.is_file():
        log(f"[grade] reference run served from {path}")
        return json.loads(path.read_text())
    out = H.run_cycle(cfg, pool, list(range(pool.n_docs)), eval_blocks, log=log)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out


def grade(deliverable_src, pool_dir, eval_dir, config_path, session_dir, log=print,
          reference_cache_dir=None) -> dict:
    """Run the submission, train twice, evaluate twice, return the full record.

    This function NEVER raises.  Every way the cycle can fail -- a hostile submission, a
    crashed or timed-out selector, a wrong return type, a diverged training run, a broken
    sealed slice, a bug in this file -- leaves `valid` False, which finalize() maps to
    exactly 0.0.  There is no path out of here that both fails and scores.
    """
    record = {
        "valid": False, "invalid_reason": None, "metric": None,
        "ppl_selection": None, "ppl_full_pool": None,
        "selection_stats": None, "select_elapsed_sec": None,
        "gpu_visible_to_grader": None, "infra_error": None,
        "writable_surface_violations": None,
        "budget_tokens": BUDGET_TOKENS, "select_timeout_sec": SELECT_TIMEOUT_SEC,
    }
    try:
        _grade_cycle(record, deliverable_src, pool_dir, eval_dir, config_path, session_dir,
                     log=log, reference_cache_dir=reference_cache_dir)
    except BaseException as exc:
        record["valid"] = False
        record["metric"] = None
        record["invalid_reason"] = f"the graded cycle raised: {type(exc).__name__}: {exc}"
        record["grade_traceback"] = traceback.format_exc()[-6000:]
    return record


def _grade_cycle(record, deliverable_src, pool_dir, eval_dir, config_path, session_dir,
                 log=print, reference_cache_dir=None) -> None:
    """The cycle itself.  Fills `record` in place; grade() owns the failure boundary."""
    deliverable_src = Path(deliverable_src)
    pool_dir = Path(pool_dir)
    eval_dir = Path(eval_dir)
    session = Path(session_dir)
    session.mkdir(parents=True, exist_ok=True)

    record["writable_surface_violations"] = writable_surface_violations(config_path)
    for violation in record["writable_surface_violations"]:
        log(f"[grade] writable-surface violation: {violation}")

    try:
        import torch
        record["gpu_visible_to_grader"] = int(torch.cuda.device_count())
    except Exception as exc:
        record["gpu_visible_to_grader"] = -1
        record["infra_error"] = f"torch/CUDA unavailable to the grader: {exc}"

    try:
        cfg = H.load_config(config_path)
        pool = H.Pool(pool_dir)
        block_size = int(cfg["block_size"])
    except Exception as exc:
        record["infra_error"] = (f"the graded pool or config could not be loaded: "
                                 f"{type(exc).__name__}: {exc}")
        record["invalid_reason"] = "graded pool unavailable"
        return
    log(f"[grade] pool docs={pool.n_docs} tokens={int(pool.offsets[-1])}")

    sel = run_selector(deliverable_src, pool_dir, session, log=log)
    record["select_elapsed_sec"] = sel["elapsed_sec"]
    record["select_stderr_tail"] = sel["stderr_tail"]
    if not sel["ok"]:
        record["invalid_reason"] = sel["error"]
        return
    try:
        accepted, used, stats = H.sanitize_selection(sel["raw"], pool.ntokens, BUDGET_TOKENS)
    except ValueError as exc:
        record["invalid_reason"] = str(exc)
        return
    record["selection_stats"] = stats
    log(f"[grade] selection {stats}")
    if len(accepted) == 0 or used < block_size:
        record["invalid_reason"] = "selection buys fewer than one full training block"
        return

    try:
        eval_blocks = H.load_eval_blocks(eval_dir, block_size)
    except Exception as exc:
        record["infra_error"] = (f"the graded evaluation shard could not be loaded: "
                                 f"{type(exc).__name__}: {exc}")
        record["invalid_reason"] = "evaluation shard unavailable"
        return
    log(f"[grade] eval blocks={eval_blocks.shape[0]}")

    try:
        sel_out = H.run_cycle(cfg, pool, accepted, eval_blocks, log=log)
    except Exception as exc:
        record["invalid_reason"] = f"training on the selection failed: {type(exc).__name__}: {exc}"
        return

    try:
        full_out = reference_perplexity(cfg, pool, eval_blocks,
                                        cache_dir=reference_cache_dir, log=log)
    except Exception as exc:
        record["infra_error"] = f"full-pool reference run failed: {type(exc).__name__}: {exc}"
        record["invalid_reason"] = "reference run unavailable"
        return

    ppl_sel = float(sel_out["perplexity"])
    ppl_full = float(full_out["perplexity"])
    record["ppl_selection"] = ppl_sel
    record["ppl_full_pool"] = ppl_full
    if not (np.isfinite(ppl_sel) and np.isfinite(ppl_full)) or ppl_full <= 0.0:
        record["invalid_reason"] = (f"perplexity is unusable (selection={ppl_sel!r}, "
                                    f"full pool={ppl_full!r})")
        return
    metric = 100.0 * (ppl_full - ppl_sel) / ppl_full
    if not np.isfinite(metric):
        record["invalid_reason"] = "metric is not finite"
        return
    record["metric"] = float(metric)
    record["valid"] = True
    log(f"[grade] ppl_sel={ppl_sel:.6f} ppl_full={ppl_full:.6f} metric={metric:.4f}")


def finalize(record: dict) -> dict:
    """Map the record to its reward.  Never raises: anything it cannot score is 0.0.

    An invalid record does not get a small reward, it gets exactly 0.0 -- there is no
    partial credit for a submission that failed, and none for a grader that failed either.
    """
    record = dict(record)
    try:
        valid = bool(record.get("valid")) and record.get("metric") is not None
        metric = float(record["metric"]) if valid else float("nan")
        if valid and not np.isfinite(metric):
            valid = False
            record["invalid_reason"] = "metric is not finite"
        reward = float(graded_reward(metric, valid))
        if not np.isfinite(reward):
            raise ValueError(f"the reward map returned {reward!r}")
    except BaseException as exc:
        valid = False
        reward = 0.0
        record["invalid_reason"] = (f"the reward map could not be evaluated: "
                                    f"{type(exc).__name__}: {exc}")
    record["valid"] = valid
    record["reward"] = reward if valid else 0.0
    record["anchors"] = {"m_floor": M_FLOOR, "m0": M0, "baseline_score": BASELINE_SCORE,
                         "degenerate_reward": DEGENERATE_REWARD}
    return record
