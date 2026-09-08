from __future__ import annotations

import json
import os
import pwd
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch




D_MODEL = 768
N_SEALED = 1_048_576
BATCH_ROWS = 8192
MAX_ATOMS = 8192
MAX_MEAN_L0 = 32.0
ENCODE_BUDGET_S = 900.0
CANDIDATE_UID = 1001



MAX_K = MAX_ATOMS
MAX_TOTAL_NONZERO = int(MAX_MEAN_L0 * N_SEALED)



SUBPROCESS_TIMEOUT_S = ENCODE_BUDGET_S + 600.0







DEGENERATE_REWARD = 0.10
M_FLOOR = 1.000370
M0 = 0.540922
BASELINE_SCORE = 0.19
SIGMA = -1.0


class SubmissionInvalid(Exception):
    """Any contract violation. Always maps to reward exactly 0."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(reason if not detail else f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


@dataclass
class GradeResult:
    valid: bool
    reason: str = "ok"
    detail: str = ""
    metric: float | None = None
    reward: float = 0.0
    mean_l0: float | None = None
    n_atoms: int | None = None
    encode_seconds: float | None = None
    numerator: float | None = None
    denominator: float | None = None
    extra: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "valid": self.valid,
            "reason": self.reason,
            "detail": self.detail,
            "metric": self.metric,
            "reward": self.reward,
            "mean_l0": self.mean_l0,
            "n_atoms": self.n_atoms,
            "encode_seconds": self.encode_seconds,
            "numerator": self.numerator,
            "denominator": self.denominator,
            **self.extra,
        }





def graded_reward(metric: float | None, valid: bool) -> float:
    """rational_squash with a linear degenerate band below the strongest trivial baseline.

    Invalid submissions short-circuit to exactly 0 before any arithmetic runs.
    """
    if not valid or metric is None:
        return 0.0
    if not np.isfinite(metric):
        return 0.0
    x = SIGMA * (metric - M0)
    x_ref = SIGMA * (BASELINE_SCORE - M0)
    assert x_ref > 0, "malformed task: the baseline score does not beat the trivial baseline"
    g = SIGMA * (M0 - M_FLOOR)
    c = DEGENERATE_REWARD
    if g <= 0:
        u = max(0.0, x) / x_ref
        return u / (1.0 + u)
    if x < 0:
        v = max(0.0, SIGMA * (metric - M_FLOOR)) / g
        return c * v
    u = x / x_ref
    return c + (1.0 - c) * (u / (1.0 + u))





def scan_submission_tree(root: Path) -> int:
    """Reject symlinks and hard links anywhere in the submitted tree, at every depth.

    Runs BEFORE anything is copied. A read-only bind mount does not remove hard links that
    are already in the submission, and a no-dereference copy still carries them.
    """
    total_bytes = 0
    if not root.is_dir() or root.is_symlink():
        raise SubmissionInvalid("missing_submission_dir", str(root))
    stack = [root]
    seen = 0
    while stack:
        cur = stack.pop()
        try:
            entries = sorted(os.scandir(cur), key=lambda e: e.name)
        except OSError as exc:
            raise SubmissionInvalid("unreadable_submission_dir", f"{cur}: {exc}") from exc
        for entry in entries:
            seen += 1
            if seen > 20000:
                raise SubmissionInvalid("submission_too_many_entries", str(cur))
            st = os.lstat(entry.path)
            if stat.S_ISLNK(st.st_mode):
                raise SubmissionInvalid("symlink_in_submission", entry.path)
            if stat.S_ISDIR(st.st_mode):
                stack.append(Path(entry.path))
                continue
            if not stat.S_ISREG(st.st_mode):
                raise SubmissionInvalid("non_regular_file_in_submission", entry.path)
            if st.st_nlink != 1:
                raise SubmissionInvalid("hard_link_in_submission",
                                        f"{entry.path} st_nlink={st.st_nlink}")
            total_bytes += st.st_size
    return total_bytes


def stage_submission(src: Path, dst: Path) -> int:
    """Scan, then copy without following links, then hand the copy to the candidate uid."""
    total_bytes = scan_submission_tree(src)
    shutil.copytree(src, dst, symlinks=True)
    gid = pwd.getpwuid(CANDIDATE_UID).pw_gid
    for path in [dst, *dst.rglob("*")]:
        os.lchown(path, CANDIDATE_UID, gid)
    return total_bytes





def load_dictionary(path: Path) -> tuple[torch.Tensor, torch.Tensor, int]:
    if not path.is_file():
        raise SubmissionInvalid("missing_dictionary", str(path))
    try:
        obj = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise SubmissionInvalid("unparseable_dictionary", repr(exc)) from exc
    if not isinstance(obj, dict) or "W_dec" not in obj or "b_pre" not in obj:
        raise SubmissionInvalid("dictionary_missing_keys", str(sorted(map(repr, obj)))[:200]
                                if isinstance(obj, dict) else type(obj).__name__)
    w_dec, b_pre = obj["W_dec"], obj["b_pre"]
    if not isinstance(w_dec, torch.Tensor) or not isinstance(b_pre, torch.Tensor):
        raise SubmissionInvalid("dictionary_not_tensors", "")
    if w_dec.dtype != torch.float32 or b_pre.dtype != torch.float32:
        raise SubmissionInvalid("dictionary_wrong_dtype",
                                f"{w_dec.dtype} {b_pre.dtype}")
    if w_dec.dim() != 2 or w_dec.shape[0] != D_MODEL:
        raise SubmissionInvalid("dictionary_wrong_shape", str(tuple(w_dec.shape)))
    n_atoms = int(w_dec.shape[1])
    if n_atoms < 1 or n_atoms > MAX_ATOMS:
        raise SubmissionInvalid("dictionary_size_out_of_range", str(n_atoms))
    if b_pre.shape != (D_MODEL,):
        raise SubmissionInvalid("offset_wrong_shape", str(tuple(b_pre.shape)))
    if not bool(torch.isfinite(w_dec).all()) or not bool(torch.isfinite(b_pre).all()):
        raise SubmissionInvalid("dictionary_not_finite", "")
    return w_dec.contiguous(), b_pre.contiguous(), n_atoms





CANDIDATE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

_PASSTHROUGH_ENV = (
    "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES",
    "NVIDIA_REQUIRE_CUDA",
    "CUDA_VISIBLE_DEVICES",
    "LD_LIBRARY_PATH",
)


def candidate_env() -> dict:
    """The complete environment the deliverable runs under. Determinism pinned HERE, not
    only in the Dockerfile: a hand-built env dict discards the image's ENV values."""
    env = {
        "PATH": CANDIDATE_PATH,
        "HOME": pwd.getpwuid(CANDIDATE_UID).pw_dir,
        "USER": pwd.getpwuid(CANDIDATE_UID).pw_name,
        "LANG": "C.UTF-8",
        "TMPDIR": "/tmp",

        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",

        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    for key in _PASSTHROUGH_ENV:
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def stage_runner(parent: Path) -> Path:
    """Copy the trusted runner out of the sealed /tests tree as a root-owned 0444 file in a
    0755 parent, so uid 1001 can execute it without /tests ever becoming traversable."""
    src = Path(__file__).resolve().parent / "run_encoder.py"
    dst = parent / "run_encoder.py"
    shutil.copyfile(src, dst)
    os.chown(dst, 0, 0)
    os.chmod(dst, 0o444)
    return dst


def stage_sealed_inputs(acts_path: Path, parent: Path) -> Path:
    """Copy the sealed activation matrix into a candidate-readable staging path.

    The activations ARE the reconstruction target, so the deliverable must see them; the
    grading signal is not a hidden label but how well a <=32-nonzero code over a submitted
    <=8192-atom dictionary can reproduce them. Nothing else from /tests is staged.
    """
    dst = parent / "sealed_acts.npy"
    shutil.copyfile(acts_path, dst)
    os.chown(dst, 0, 0)
    os.chmod(dst, 0o444)
    return dst


def run_candidate(submission_dir: Path, acts_path: Path, workdir: Path) -> tuple[Path, dict]:
    """Execute the deliverable as uid 1001 in its own process group under a wall-clock cap."""
    parent = workdir / "candidate"
    parent.mkdir(parents=True)
    os.chown(parent, 0, 0)
    os.chmod(parent, 0o755)

    staged_sub = parent / "submission"
    stage_submission(submission_dir, staged_sub)
    entry = staged_sub / "encoder.py"
    if not entry.is_file() or entry.stat().st_size == 0:
        raise SubmissionInvalid("missing_or_empty_entry_point", str(entry))

    runner = stage_runner(parent)
    staged_acts = stage_sealed_inputs(acts_path, parent)

    codes_dir = parent / "codes"
    codes_dir.mkdir()
    gid = pwd.getpwuid(CANDIDATE_UID).pw_gid
    os.lchown(codes_dir, CANDIDATE_UID, gid)
    os.chmod(codes_dir, 0o755)

    cmd = [
        "runuser", "-u", "agent", "--",
        sys.executable, str(runner),
        "--submission-dir", str(staged_sub),
        "--acts", str(staged_acts),
        "--out-dir", str(codes_dir),
        "--batch-rows", str(BATCH_ROWS),
        "--n-rows", str(N_SEALED),
        "--budget", str(ENCODE_BUDGET_S),
        "--max-k", str(MAX_K),
        "--max-total-nonzero", str(MAX_TOTAL_NONZERO),
    ]
    started = time.monotonic()


    proc = subprocess.Popen(
        cmd, cwd=str(parent), env=candidate_env(), stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, start_new_session=True)
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=SUBPROCESS_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_group(proc)
        try:
            stdout, stderr = proc.communicate(timeout=60)
        except Exception:
            stdout, stderr = "", ""
    rc = proc.returncode
    elapsed = time.monotonic() - started
    if timed_out:
        raise SubmissionInvalid(
            "harness_timeout",
            f"candidate did not exit within {SUBPROCESS_TIMEOUT_S}s: {stderr[-1500:]}")

    meta = {"returncode": rc, "wall_seconds": elapsed,
            "stdout_tail": stdout[-4000:], "stderr_tail": stderr[-4000:]}
    failure_path = codes_dir / "failure.json"
    if failure_path.is_file():
        try:
            payload = json.loads(failure_path.read_text())
        except Exception:
            payload = {"reason": "candidate_failed", "detail": "unreadable failure record"}
        meta["failure"] = payload
        raise SubmissionInvalid(str(payload.get("reason", "candidate_failed")),
                                f"{payload.get('detail', '')} | stderr={stderr[-1500:]}")
    if rc != 0:
        raise SubmissionInvalid("candidate_nonzero_exit",
                                f"rc={rc} stderr={stderr[-1500:]}")
    run_meta_path = codes_dir / "run_meta.json"
    if not run_meta_path.is_file():
        raise SubmissionInvalid("candidate_wrote_no_run_meta", stderr[-1500:])
    try:
        meta["run_meta"] = json.loads(run_meta_path.read_text())
    except Exception as exc:
        raise SubmissionInvalid("unparseable_run_meta", repr(exc)) from exc
    return codes_dir, meta


def _kill_group(proc: "subprocess.Popen") -> None:
    """Reap the candidate's whole process group. The child called setsid(2) at launch, so
    its pid is the group id and one killpg takes down everything it forked."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        proc.kill()
    except Exception:
        pass





def _load_batch_arrays(codes_dir: Path, b: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        row = np.load(codes_dir / f"b{b:05d}_row.npy")
        col = np.load(codes_dir / f"b{b:05d}_col.npy")
        val = np.load(codes_dir / f"b{b:05d}_val.npy")
    except Exception as exc:
        raise SubmissionInvalid("unreadable_codes", f"batch {b}: {exc!r}") from exc
    return row, col, val


def score_codes(codes_dir: Path, acts_path: Path, w_dec: torch.Tensor,
                b_pre: torch.Tensor, n_atoms: int, device: torch.device) -> dict:
    acts = np.load(acts_path, mmap_mode="r")
    if acts.shape != (N_SEALED, D_MODEL):
        raise SubmissionInvalid("sealed_split_wrong_shape", str(acts.shape))

    w_dec_d = w_dec.to(device)
    b_pre_d = b_pre.to(device)


    mean_acc = torch.zeros(D_MODEL, dtype=torch.float64, device=device)
    n_batches = N_SEALED // BATCH_ROWS
    for b in range(n_batches):
        x = torch.from_numpy(np.array(
            acts[b * BATCH_ROWS:(b + 1) * BATCH_ROWS])).to(device).float()
        mean_acc += x.double().sum(dim=0)
    x_bar = (mean_acc / float(N_SEALED)).float()

    numerator = 0.0
    denominator = 0.0
    total_nonzero = 0
    for b in range(n_batches):
        row, col, val = _load_batch_arrays(codes_dir, b)
        if row.ndim != 1 or col.ndim != 1 or val.ndim != 1:
            raise SubmissionInvalid("codes_wrong_rank", f"batch {b}")
        if not (row.shape == col.shape == val.shape):
            raise SubmissionInvalid("codes_length_mismatch", f"batch {b}")
        if row.dtype != np.int32 or col.dtype != np.int32 or val.dtype != np.float32:
            raise SubmissionInvalid("codes_wrong_dtype", f"batch {b}")
        nnz = int(row.shape[0])
        total_nonzero += nnz
        if total_nonzero > MAX_TOTAL_NONZERO:
            raise SubmissionInvalid("nonzero_budget_exceeded", str(total_nonzero))
        if nnz:
            if int(row.min()) < 0 or int(row.max()) >= BATCH_ROWS:
                raise SubmissionInvalid("code_row_out_of_range", f"batch {b}")
            if int(col.min()) < 0 or int(col.max()) >= n_atoms:
                raise SubmissionInvalid("code_index_out_of_range",
                                        f"batch {b}: index outside [0, {n_atoms})")
            if not np.isfinite(val).all():
                raise SubmissionInvalid("code_value_not_finite", f"batch {b}")
            if np.any(val == 0.0):
                raise SubmissionInvalid("zero_valued_code_entry", f"batch {b}")
            keys = row.astype(np.int64) * np.int64(n_atoms) + col.astype(np.int64)
            if np.unique(keys).shape[0] != keys.shape[0]:
                raise SubmissionInvalid("duplicate_index_in_row", f"batch {b}")

        x = torch.from_numpy(np.array(
            acts[b * BATCH_ROWS:(b + 1) * BATCH_ROWS])).to(device).float()
        code = torch.zeros((BATCH_ROWS, n_atoms), dtype=torch.float32, device=device)
        if nnz:
            code[torch.from_numpy(row.astype(np.int64)).to(device),
                 torch.from_numpy(col.astype(np.int64)).to(device)] = \
                torch.from_numpy(val).to(device)
        x_hat = torch.addmm(b_pre_d, code, w_dec_d.t())
        numerator += float(((x - x_hat).double() ** 2).sum().item())
        denominator += float(((x - x_bar).double() ** 2).sum().item())
        del code, x, x_hat

    if denominator <= 0.0:
        raise SubmissionInvalid("degenerate_sealed_split", "zero variance")
    mean_l0 = total_nonzero / float(N_SEALED)
    if mean_l0 > MAX_MEAN_L0:
        raise SubmissionInvalid("mean_l0_exceeded", f"{mean_l0:.6f} > {MAX_MEAN_L0}")
    return {"metric": numerator / denominator, "numerator": numerator,
            "denominator": denominator, "mean_l0": mean_l0,
            "total_nonzero": total_nonzero}





SEALED_CACHE = Path("/tmp/sealed_acts")


def materialize_sealed(split_dir: Path) -> Path:
    """Derive the sealed activations for one half, at grade time, on the GPU.

    `docker build` has no accelerator, so this cannot happen in the image. It runs here, in
    the trusted root process, from the sealed shard the verifier image holds, using the same
    pinned extractor the agent had -- byte-identical bytes, a different shard argument. The
    result is cached for the life of the container so the shape assertions and the grade
    share one extraction.
    """
    spec = load_split_spec(split_dir)
    SEALED_CACHE.mkdir(parents=True, exist_ok=True)
    os.chmod(SEALED_CACHE, 0o700)
    out = SEALED_CACHE / f"{split_dir.name}.npy"
    if out.is_file():
        return out
    shard = split_dir.parent / "sealed_shard.parquet"
    if not shard.is_file():
        raise SubmissionInvalid("missing_sealed_shard", str(shard))
    excludes = split_dir.parent / "excluded_doc_indices.json"
    extractor = Path(__file__).resolve().parent / "extract_acts.py"
    cmd = [sys.executable, str(extractor),
           "--parquet", str(shard),
           "--doc-start", str(int(spec["doc_start"])),
           "--doc-end", str(int(spec["doc_end"])),
           "--max-tokens", str(int(spec["max_tokens"])),
           "--ctx", str(int(spec["ctx"])),
           "--device", "cuda" if torch.cuda.is_available() else "cpu",
           "--exclude-doc-indices", str(excludes),
           "--out", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=3600, check=False)
    if proc.returncode != 0 or not out.is_file():
        raise SubmissionInvalid(
            "sealed_extraction_failed",
            f"rc={proc.returncode} stderr={proc.stderr[-2000:]}")
    return out


def load_split_spec(split_dir: Path) -> dict:
    spec_path = split_dir / "spec.json"
    if not spec_path.is_file():
        raise SubmissionInvalid("missing_split_spec", str(spec_path))
    return json.loads(spec_path.read_text())


def assert_split_disjoint(specs: list[dict], agent_shards: list[str]) -> dict:
    """Group-level disjointness on the canonical key, which here is (shard, document index).

    Rows are 64-token windows of a document, so a row-level split would put windows of the
    same document on both sides. The assertion is made while both partitions exist, and it
    is re-run at grade time rather than trusted from build time.
    """
    seen: dict[str, list[tuple[int, int]]] = {}
    for spec in specs:
        shard = spec["shard"]
        if shard in agent_shards:
            raise SubmissionInvalid("sealed_split_uses_agent_shard", shard)
        lo, hi = int(spec["doc_start"]), int(spec["doc_end"])
        for other_lo, other_hi in seen.get(shard, []):
            if lo < other_hi and other_lo < hi:
                raise SubmissionInvalid(
                    "sealed_splits_overlap", f"{shard} [{lo},{hi}) vs [{other_lo},{other_hi})")
        seen.setdefault(shard, []).append((lo, hi))
    return {"shards": sorted(seen), "ranges": {k: v for k, v in seen.items()}}





def grade(split_dir: Path, submission_dir: Path, all_split_dirs: list[Path],
          agent_shards: list[str]) -> GradeResult:
    workdir = Path(tempfile.mkdtemp(prefix="grade_", dir="/tmp"))
    os.chmod(workdir, 0o755)
    try:
        specs = [load_split_spec(d) for d in all_split_dirs]
        integrity = assert_split_disjoint(specs, agent_shards)

        acts_path = materialize_sealed(split_dir)

        dict_path = submission_dir / "dictionary.pt"
        w_dec, b_pre, n_atoms = load_dictionary(dict_path)

        codes_dir, meta = run_candidate(submission_dir, acts_path, workdir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scored = score_codes(codes_dir, acts_path, w_dec, b_pre, n_atoms, device)

        try:
            encode_seconds = float(meta["run_meta"]["encode_seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SubmissionInvalid("unparseable_run_meta",
                                    f"encode_seconds: {exc!r}") from exc
        if encode_seconds > ENCODE_BUDGET_S:
            raise SubmissionInvalid("encode_budget_exceeded",
                                    f"{encode_seconds:.1f}s > {ENCODE_BUDGET_S}s")
        metric = scored["metric"]
        return GradeResult(
            valid=True, reason="ok", metric=metric,
            reward=graded_reward(metric, True), mean_l0=scored["mean_l0"],
            n_atoms=n_atoms, encode_seconds=encode_seconds,
            numerator=scored["numerator"], denominator=scored["denominator"],
            extra={"split": split_dir.name, "split_integrity": integrity,
                   "torch_device": str(device),
                   "cuda_device_count": torch.cuda.device_count(),
                   "candidate_wall_seconds": meta.get("wall_seconds"),
                   "candidate_stderr_tail": meta.get("stderr_tail", "")[-1500:],
                   "candidate_stdout_tail": meta.get("stdout_tail", "")[-1500:]})
    except SubmissionInvalid as exc:
        return GradeResult(valid=False, reason=exc.reason, detail=exc.detail[:4000],
                           reward=0.0, extra={"split": split_dir.name})
    except Exception as exc:
        return GradeResult(valid=False, reason="grader_internal_error",
                           detail=f"{type(exc).__name__}: {exc}"[:4000], reward=0.0,
                           extra={"split": split_dir.name})
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
