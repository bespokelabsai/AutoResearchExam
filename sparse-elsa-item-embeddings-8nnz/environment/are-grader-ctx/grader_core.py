import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp





N_ITEMS = 10000
MAX_NNZ_PER_ITEM = 8
MAX_DIMS = 65536
CANDIDATE_SEED = 815
CANDIDATE_TIMEOUT_SEC = 2400.0
K = 100
CANDIDATE_THREADS = "8"



M0 = 0.354019
BASELINE_SCORE = 0.469





def load_train_matrix(path) -> sp.csr_matrix:
    X = sp.load_npz(str(path)).tocsr()
    X.sort_indices()
    return X.astype(np.float32)


def load_eval_slice(path):
    """Return (src, target) csr matrices of the sealed evaluation rows."""
    with np.load(str(path), allow_pickle=False) as z:
        shape = tuple(int(v) for v in z["shape"])
        src = sp.csr_matrix(
            (z["src_data"].astype(np.float32), z["src_indices"], z["src_indptr"]), shape=shape
        )
        tgt_indptr = z["tgt_indptr"]
        tgt_indices = z["tgt_indices"]
    tgt = sp.csr_matrix(
        (np.ones(len(tgt_indices), dtype=np.float32), tgt_indices, tgt_indptr), shape=shape
    )
    src.sort_indices()
    tgt.sort_indices()
    return src, tgt





def candidate_command(python_exe, runner, code_copy, train_copy, emb_path, run_as, timeout):
    """The exact command line the graded run is launched with.

    Three properties are load-bearing and are asserted by tests/test_state.py: `runuser` drops
    to the unprivileged uid, `setsid` detaches the run into its own session, and `timeout
    --signal=KILL` bounds the whole process group in wall-clock seconds.
    """
    argv = [
        str(python_exe),
        "-I",
        str(runner),
        str(code_copy),
        str(train_copy),
        str(emb_path),
        str(N_ITEMS),
        str(MAX_NNZ_PER_ITEM),
        str(MAX_DIMS),
        str(CANDIDATE_SEED),
    ]
    wrapped = ["setsid", "timeout", "--signal=KILL", str(int(timeout))] + argv
    if run_as:
        runuser = shutil.which("runuser") or "/usr/sbin/runuser"
        wrapped = [runuser, "-u", run_as, "--"] + wrapped
    return wrapped


def run_candidate(
    code_dir,
    harness_dir,
    work_dir,
    train_path,
    python_exe=None,
    run_as="agent",
    timeout_sec=CANDIDATE_TIMEOUT_SEC,
):
    """Copy the deliverable somewhere the candidate cannot rewrite, run it, return status.

    Returns (ok, detail, embedding_path, elapsed_sec).
    """
    python_exe = python_exe or sys.executable
    harness_dir = Path(harness_dir)
    work_dir = Path(work_dir)
    code_dir = Path(code_dir)

    for d in (harness_dir, work_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    runner_src = Path(__file__).resolve().parent / "run_candidate.py"
    runner = harness_dir / "run_candidate.py"
    shutil.copy2(runner_src, runner)

    code_copy = harness_dir / "candidate_code"
    if not code_dir.is_dir():
        return False, f"deliverable directory {code_dir} is missing", None, 0.0



    shutil.copytree(code_dir, code_copy, symlinks=True, ignore_dangling_symlinks=True)

    train_copy = harness_dir / "train.npz"
    shutil.copy2(train_path, train_copy)

    emb_path = work_dir / "embedding.npz"


    for path in harness_dir.rglob("*"):
        os.chmod(path, 0o555 if path.is_dir() else 0o444)
    os.chmod(harness_dir, 0o555)
    if run_as:
        shutil.chown(work_dir, user=run_as)
    os.chmod(work_dir, 0o700)

    wrapped = candidate_command(
        python_exe, runner, code_copy, train_copy, emb_path, run_as=run_as, timeout=timeout_sec
    )

    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": str(work_dir),
        "TMPDIR": str(work_dir),
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": CANDIDATE_THREADS,
        "MKL_NUM_THREADS": CANDIDATE_THREADS,
        "OPENBLAS_NUM_THREADS": CANDIDATE_THREADS,
        "NUMEXPR_NUM_THREADS": CANDIDATE_THREADS,
        "TORCH_NUM_THREADS": CANDIDATE_THREADS,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }

    start = time.monotonic()
    try:
        proc = subprocess.run(
            wrapped,
            cwd=str(work_dir),
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout_sec + 120.0,
        )
    except subprocess.TimeoutExpired:
        return False, "candidate exceeded its wall-clock budget", None, time.monotonic() - start
    elapsed = time.monotonic() - start

    tail = (proc.stdout or "")[-2000:] + (proc.stderr or "")[-2000:]
    if proc.returncode == -9 or proc.returncode == 137:
        return False, f"candidate killed at the wall-clock budget: {tail}", None, elapsed
    if proc.returncode != 0:
        return False, f"candidate exited {proc.returncode}: {tail}", None, elapsed
    if not emb_path.is_file():
        return False, f"candidate wrote no embedding: {tail}", None, elapsed
    return True, tail, emb_path, elapsed





class InvalidSubmission(Exception):
    """The submitted embedding violates the declared contract."""


def load_embedding(path) -> sp.csr_matrix:
    """Build A from the candidate's serialized output, enforcing the declared contract.

    The serialization holds at most MAX_NNZ_PER_ITEM slots per row, so the sparsity budget
    is structurally unforgeable: a candidate that lies about its own row counts can only
    ever produce a matrix that already satisfies the cap.
    """
    with np.load(str(path), allow_pickle=False) as z:
        missing = {"idx", "val", "nnz_per_row", "n_rows", "n_dims"} - set(z.files)
        if missing:
            raise InvalidSubmission(f"serialized embedding is missing {sorted(missing)}")
        idx = np.asarray(z["idx"])
        val = np.asarray(z["val"])
        nnz_per_row = np.asarray(z["nnz_per_row"])
        n_rows = int(np.asarray(z["n_rows"]).reshape(-1)[0])
        n_dims = int(np.asarray(z["n_dims"]).reshape(-1)[0])

    if n_rows != N_ITEMS:
        raise InvalidSubmission(f"A has {n_rows} rows, expected {N_ITEMS}")
    if not (1 <= n_dims <= MAX_DIMS):
        raise InvalidSubmission(f"A has {n_dims} columns, expected 1..{MAX_DIMS}")
    if idx.shape != (N_ITEMS, MAX_NNZ_PER_ITEM) or val.shape != (N_ITEMS, MAX_NNZ_PER_ITEM):
        raise InvalidSubmission(f"serialized shapes {idx.shape}/{val.shape} are malformed")
    if nnz_per_row.shape != (N_ITEMS,):
        raise InvalidSubmission("serialized nnz_per_row is malformed")
    if int(nnz_per_row.max(initial=0)) > MAX_NNZ_PER_ITEM:
        worst = int(nnz_per_row.max(initial=0))
        raise InvalidSubmission(
            f"row {int(np.argmax(nnz_per_row))} of A has {worst} nonzeros, "
            f"the budget is {MAX_NNZ_PER_ITEM}"
        )
    if not np.issubdtype(val.dtype, np.floating) and not np.issubdtype(val.dtype, np.integer):
        raise InvalidSubmission(f"A has non-real dtype {val.dtype}")
    val = val.astype(np.float64)
    idx = idx.astype(np.int64)
    if not np.isfinite(val).all():
        raise InvalidSubmission("A contains a non-finite value")
    live = (idx >= 0) & (val != 0.0)
    if np.any(idx[live] >= n_dims):
        raise InvalidSubmission("A references a column outside its declared width")

    rows = np.repeat(np.arange(N_ITEMS, dtype=np.int64), MAX_NNZ_PER_ITEM)[live.ravel()]
    cols = idx.ravel()[live.ravel()]
    data = val.ravel()[live.ravel()]
    A = sp.csr_matrix((data, (rows, cols)), shape=(N_ITEMS, n_dims), dtype=np.float64)
    A.sum_duplicates()
    A.eliminate_zeros()
    over = np.flatnonzero(np.diff(A.indptr) > MAX_NNZ_PER_ITEM)
    if over.size:
        raise InvalidSubmission(f"row {int(over[0])} of A exceeds the nonzero budget")
    return A


def row_normalize(A: sp.csr_matrix) -> sp.csr_matrix:
    """L2-normalize every row of A; all-zero rows stay all-zero."""
    A = A.tocsr(copy=True).astype(np.float64)
    norms = np.sqrt(np.asarray(A.multiply(A).sum(axis=1)).ravel())
    scale = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
    return sp.diags(scale).dot(A).tocsr()


def gram(A_norm: sp.csr_matrix, block: int = 2000) -> np.ndarray:
    """Dense A A^T.  Blocked, because A A^T can be dense even when A is very sparse."""
    G = np.empty((A_norm.shape[0], A_norm.shape[0]), dtype=np.float64)
    AT = A_norm.T.tocsc()
    for i in range(0, A_norm.shape[0], block):
        j = min(i + block, A_norm.shape[0])
        G[i:j] = (A_norm[i:j] @ AT).toarray()
    return G


def mean_ndcg_at_k(A_norm, src, tgt, k=K, block=1000):
    """Mean nDCG@k over the evaluation rows, exactly as instruction.md defines it.

    r = x_src A_norm A_norm^T - x_src, the row's own src items forced to -inf, ranked by
    descending score with ties broken by ascending item index; DCG sums 1/log2(rank+1)
    over target items landing in the top k; IDCG = sum_{i=1..min(k, #targets)} 1/log2(i+1).
    """
    G = gram(A_norm)
    n = src.shape[0]
    disc = 1.0 / np.log2(np.arange(1, k + 1) + 1.0)
    idcg_table = np.concatenate([[0.0], np.cumsum(disc)])
    scores = np.empty(n, dtype=np.float64)

    for i in range(0, n, block):
        j = min(i + block, n)
        xs = src[i:j]
        R = xs @ G
        R -= xs.toarray()
        rows, cols = xs.nonzero()
        R[rows, cols] = -np.inf
        top = np.argsort(-R, axis=1, kind="stable")[:, :k]
        for b in range(j - i):
            targets = tgt.indices[tgt.indptr[i + b] : tgt.indptr[i + b + 1]]
            n_tgt = len(targets)
            if n_tgt == 0:
                scores[i + b] = 0.0
                continue
            hit = np.isin(top[b], targets, assume_unique=False)
            dcg = float(disc[hit].sum())
            idcg = float(idcg_table[min(k, n_tgt)])
            scores[i + b] = dcg / idcg
    return float(scores.mean()), scores





def graded_reward(metric, valid=True, m0=M0, baseline_score=BASELINE_SCORE):
    if not valid or metric is None or not math.isfinite(metric):
        return 0.0
    x_ref = baseline_score - m0
    assert x_ref > 0, "malformed task: the baseline score does not beat the trivial baseline"
    u = max(0.0, metric - m0) / x_ref
    return u / (1.0 + u)





def grade(
    code_dir,
    train_path,
    eval_path,
    harness_dir,
    work_dir,
    metric_path=None,
    python_exe=None,
    run_as="agent",
    timeout_sec=CANDIDATE_TIMEOUT_SEC,
):
    result = {
        "valid": False,
        "metric": None,
        "reward": 0.0,
        "stage": "start",
        "detail": "",
        "elapsed_sec": None,
        "n_dims": None,
        "max_nnz_per_item": None,
        "n_eval_rows": None,
    }
    try:
        ok, detail, emb_path, elapsed = run_candidate(
            code_dir,
            harness_dir,
            work_dir,
            train_path,
            python_exe=python_exe,
            run_as=run_as,
            timeout_sec=timeout_sec,
        )
        result["elapsed_sec"] = round(elapsed, 2)
        result["detail"] = detail[-1500:] if detail else ""
        if not ok:
            result["stage"] = "execution"
            return result

        result["stage"] = "validation"
        A = load_embedding(emb_path)
        result["n_dims"] = int(A.shape[1])
        result["max_nnz_per_item"] = int(np.diff(A.indptr).max(initial=0))

        result["stage"] = "scoring"
        src, tgt = load_eval_slice(eval_path)
        result["n_eval_rows"] = int(src.shape[0])
        metric, per_row = mean_ndcg_at_k(row_normalize(A), src, tgt)
        result["metric"] = metric
        result["per_row_sem"] = float(per_row.std(ddof=1) / math.sqrt(len(per_row)))
        result["valid"] = True
        result["reward"] = graded_reward(metric, True)
        result["stage"] = "done"
        return result
    except InvalidSubmission as exc:
        result["detail"] = f"invalid submission: {exc}"
        return result
    except Exception as exc:
        result["detail"] = f"{type(exc).__name__}: {exc}"
        return result
    finally:
        if metric_path is not None:
            Path(metric_path).parent.mkdir(parents=True, exist_ok=True)
            Path(metric_path).write_text(json.dumps(result, indent=2, default=str) + "\n")
