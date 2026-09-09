import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oracle

TESTS_DIR = Path(__file__).resolve().parent

ENTRY_POINT = "estimator.py"
REPLICATIONS = int(os.environ.get("GRADER_REPLICATIONS", "20"))
REPLICATION_TIMEOUT_S = 90.0
TRIVIAL_ERROR = 1.0
RUN_AS_UID = int(os.environ.get("GRADER_RUN_AS_UID", "1001"))
WORKERS = int(os.environ.get("GRADER_WORKERS", "8"))
GLOBAL_DEADLINE_S = float(os.environ.get("GRADER_DEADLINE_S", "1200"))

M0 = 1.0
BASELINE_SCORE = 0.013


def reward_from_metric(metric):
    """log-ratio squash: 0 at the trivial baseline, 0.5 at the reference, -> 1, no cutoff."""
    if metric is None:
        return 0.0
    m = float(metric)
    if not math.isfinite(m) or m < 0.0:
        return 0.0
    if m == 0.0:
        return 1.0
    u = math.log(M0 / m) / math.log(M0 / BASELINE_SCORE)
    if u <= 0.0:
        return 0.0
    return u / (1.0 + u)


def load_instances(split_dir):
    paths = sorted(Path(split_dir).glob("inst_*.json"))
    if not paths:
        raise RuntimeError(f"no instances under {split_dir}")
    return [json.loads(p.read_text()) for p in paths]


def instance_covariance(inst):
    d = int(inst["d"])
    Sigma = np.full((d, d), float(inst["equi_r"]))
    np.fill_diagonal(Sigma, 1.0)
    return Sigma


def true_functional(inst):
    """w' Xi* w for one instance, from the closed form. Never leaves this process."""
    d = int(inst["d"])
    Xi = oracle.xi_star(instance_covariance(inst), float(inst["sigma"]) ** 2, int(inst["tau"]))
    w = np.full(d, 1.0 / d)
    return float(w @ Xi @ w)


def draw_samples(inst, seed):
    """The realized samples of one replication. The only thing the child is given."""
    d, T = int(inst["d"]), int(inst["T"])
    L = np.linalg.cholesky(instance_covariance(inst))
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((T, d)) @ L.T
    b = A @ np.asarray(inst["x_star"], dtype=np.float64) + rng.standard_normal(T) * float(inst["sigma"])
    return A, b


def _launch(workdir, timeout_s):
    """Run the deliverable in its own session as an unprivileged uid, hard-killed on time.

    Output goes to a file rather than a pipe, so an agent that prints without bound cannot
    grow the trusted process's memory; only the last 2 kB is ever read back.
    """
    py = sys.executable
    script = str(workdir / "run_one.py")
    argv = [py, script, "--workdir", str(workdir)]
    kwargs = dict(cwd=str(workdir / "out"), start_new_session=True,
                  user=RUN_AS_UID, group=RUN_AS_UID, env={
                      "PATH": "/usr/local/bin:/usr/bin:/bin",
                      "HOME": str(workdir / "out"), "TMPDIR": str(workdir / "out"),
                      "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                      "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
                      "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"})

    log = workdir / "child.log"
    with open(log, "wb") as fh:
        proc = subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT, **kwargs)
        try:
            proc.communicate(timeout=timeout_s)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                proc.kill()
            proc.communicate()
            rc = "timeout"
    tail = ""
    try:
        with open(log, "rb") as fh:
            fh.seek(max(0, log.stat().st_size - 2000))
            tail = fh.read().decode("utf-8", "replace")
    except Exception:
        pass
    return rc, tail


def run_replication(inst, rep, solution_dir, workroot, timeout_s):
    """Prepare a scratch dir, run one replication, return the raw functional or a failure."""
    out = {"instance": inst["name"], "rep": rep, "status": "error", "v_hat": None, "detail": ""}
    if timeout_s <= 1.0:
        out["status"] = "deadline"
        return out
    work = Path(tempfile.mkdtemp(dir=workroot, prefix=f"{inst['name']}_r{rep}_"))
    try:
        shutil.copy2(TESTS_DIR / "run_one.py", work / "run_one.py")
        shutil.copytree(TESTS_DIR / "harness", work / "harness")
        shutil.copytree(solution_dir, work / "solution", symlinks=True)
        seed = int(inst["seed_base"]) + 2 * rep
        A, b = draw_samples(inst, seed)
        np.save(work / "samples_a.npy", A, allow_pickle=False)
        np.save(work / "samples_b.npy", b, allow_pickle=False)
        (work / "job.json").write_text(json.dumps(
            {"d": int(inst["d"]), "T": int(inst["T"]), "tau": int(inst["tau"]),
             "alg_seed": seed + 1}))
        del A, b
        for p in [work] + list(work.rglob("*")):
            os.chmod(p, 0o755 if p.is_dir() else 0o644)
        (work / "out").mkdir()
        os.chown(work / "out", RUN_AS_UID, RUN_AS_UID)
        os.chmod(work / "out", 0o700)

        rc, err = _launch(work, timeout_s)
        if rc == "timeout":
            out["status"] = "timeout"
            out["detail"] = f"exceeded {timeout_s:.0f}s"
            return out
        if rc != 0:
            out["status"] = "crashed"
            out["detail"] = err.strip()[-600:]
            return out

        res = work / "out" / "result.npy"
        if not res.exists():
            out["status"] = "no_output"
            return out
        try:
            M = np.load(res, allow_pickle=False)
        except Exception as exc:
            out["status"] = "unparseable"
            out["detail"] = str(exc)[:200]
            return out
        d = int(inst["d"])
        if M.shape != (d, d) or M.dtype != np.float64:
            out["status"] = "bad_shape"
            out["detail"] = f"got shape {M.shape} dtype {M.dtype}"
            return out
        if not np.all(np.isfinite(M)):
            out["status"] = "non_finite"
            return out
        w = np.full(d, 1.0 / d)
        out["v_hat"] = float(w @ M @ w)
        out["status"] = "ok"
        return out
    except Exception as exc:
        out["status"] = "infra_error"
        out["detail"] = f"{type(exc).__name__}: {exc}"[:300]
        return out
    finally:
        shutil.rmtree(work, ignore_errors=True)


def grade(split_dir, deliverable_dir, workroot=None, replications=REPLICATIONS):
    t0 = time.time()
    instances = load_instances(split_dir)
    truth = {i["name"]: true_functional(i) for i in instances}

    entry = Path(deliverable_dir) / ENTRY_POINT
    have_entry = entry.is_file() and entry.stat().st_size > 0
    results = []

    if have_entry:
        workroot = workroot or tempfile.mkdtemp(prefix="grade_")
        os.chmod(workroot, 0o711)
        staged = Path(workroot) / "solution"
        shutil.rmtree(staged, ignore_errors=True)
        src = Path(deliverable_dir)
        for p in [src] + list(src.rglob("*")):
            if p.is_symlink():
                raise RuntimeError(f"symlink in deliverable is not allowed: {p}")
        shutil.copytree(deliverable_dir, staged, symlinks=True)
        for p in [staged] + list(staged.rglob("*")):
            os.chmod(p, 0o755 if p.is_dir() else 0o644)

        jobs = [(i, r) for i in instances for r in range(replications)]

        def work(job):
            inst, rep = job
            left = GLOBAL_DEADLINE_S - (time.time() - t0)
            return run_replication(inst, rep, staged, workroot,
                                   min(REPLICATION_TIMEOUT_S, left))

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            results = list(pool.map(work, jobs))
    else:
        for inst in instances:
            for rep in range(replications):
                results.append({"instance": inst["name"], "rep": rep,
                                "status": "missing_entry_point", "v_hat": None, "detail": ""})

    per_instance = {}
    for inst in instances:
        name = inst["name"]
        rows = [r for r in results if r["instance"] == name]
        bad = [r for r in rows if r["status"] != "ok"]
        if bad or len(rows) != replications:
            per_instance[name] = {"error": TRIVIAL_ERROR, "failed": len(bad) + replications - len(rows)}
            continue
        v = truth[name]
        rel = np.array([r["v_hat"] for r in rows], dtype=np.float64) / v - 1.0
        sd = float(rel.std(ddof=1)) if len(rel) > 1 else float("nan")
        per_instance[name] = {"error": float(abs(rel.mean())), "failed": 0,
                              "signed_mean": float(rel.mean()),
                              "sd_per_replication": sd,
                              "mc_error": sd / math.sqrt(len(rel))}

    metric = float(np.mean([per_instance[i["name"]]["error"] for i in instances]))
    n_failed = sum(1 for r in results if r["status"] != "ok")
    statuses = {}
    for r in results:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    first_detail = next((r["detail"] for r in results if r["status"] != "ok" and r["detail"]), "")

    return {
        "metric": metric,
        "reward": reward_from_metric(metric),
        "replications_per_instance": replications,
        "replications_failed": n_failed,
        "replication_status_counts": statuses,
        "first_failure_detail": first_detail,
        "per_instance": per_instance,
        "wall_clock_s": round(time.time() - t0, 1),
    }


def summary_lines(report):
    """The only thing printed to stdout. Aggregates only -- no per-instance detail."""
    return [
        f"replications per instance : {report['replications_per_instance']}",
        f"replications failed       : {report['replications_failed']}"
        + (f" {report['replication_status_counts']}" if report["replications_failed"] else ""),
        f"first failure             : {report['first_failure_detail'][:600]}"
        if report["first_failure_detail"] else "first failure             : -",
        f"metric (lower is better)  : {report['metric']:.6f}",
        f"reward                    : {report['reward']:.6f}",
        f"grading wall clock (s)    : {report['wall_clock_s']}",
    ]
