from __future__ import annotations

import json
import os
import pwd
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import al_loop
from worker_client import CALL_BUDGET_SEC, IMPORT_BUDGET_SEC, PruneWorker, PrunerFailure

MAX_CONCURRENT_RUNS = 6


def stage_pool_texts(texts, path):
    """Write the pool text list where a worker can read it (texts only: no labels)."""
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(list(texts), fh, ensure_ascii=False)
    os.replace(tmp, path)
    os.chmod(path, 0o644)
    return str(path)


def make_readonly(tree):
    """Strip every write bit from a staged tree.  The scoring harness hands the pruner a
    read-only copy of the deliverable so no seeded run can leave anything on disk for the next
    one; the development runner does the same, so a solution that needs to write at run time
    fails locally rather than at grade time."""
    os.chmod(tree, 0o555)
    for base, dirs, files in os.walk(tree):
        for name in dirs:
            os.chmod(os.path.join(base, name), 0o555)
        for name in files:
            os.chmod(os.path.join(base, name), 0o444)
    return tree


def _make_scratch(root, name, as_user):
    path = os.path.join(root, name)
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)
    if as_user:
        entry = pwd.getpwnam(as_user)
        os.lchown(path, entry.pw_uid, entry.pw_gid)
        os.chmod(path, 0o700)
    return path


def execute_runs(specs, *, python_exe, worker_script, module_dir, entry_module, scratch_root,
                 as_user=None, launch_prefix=None, new_session=os.setsid,
                 max_workers=MAX_CONCURRENT_RUNS,
                 call_budget=CALL_BUDGET_SEC, import_budget=IMPORT_BUDGET_SEC,
                 stderr_root=None):
    """specs: iterable of dicts with keys draw, seed, pool_labels, pool_X, eval_y, eval_X,
    pool_texts_path.  Returns one result dict per spec, in the input order."""
    os.makedirs(scratch_root, exist_ok=True)

    def one(index_spec):
        index, spec = index_spec
        tag = f"run_{spec['draw']}_{spec['seed']}"
        scratch = _make_scratch(scratch_root, tag, as_user)
        stderr_path = os.path.join(stderr_root, tag + ".err") if stderr_root else None
        worker = PruneWorker(python_exe, worker_script, module_dir, entry_module,
                             spec["pool_texts_path"], scratch, launch_prefix=launch_prefix,
                             new_session=new_session, stderr_path=stderr_path,
                             call_budget=call_budget,
                             import_budget=import_budget)
        result = {"index": index, "draw": spec["draw"], "seed": spec["seed"],
                  "curve": None, "final": None, "timings": [], "error": None}
        try:
            worker.start()
            curve = al_loop.run_al(spec["pool_labels"], spec["pool_X"], spec["eval_y"],
                                   spec["eval_X"], spec["seed"], worker.call)
            result["curve"] = [float(v) for v in curve]
            result["final"] = float(curve[-1])
        except (PrunerFailure, al_loop.InvalidSelection) as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            result["error"] = f"harness error: {type(exc).__name__}: {exc}"
        finally:
            result["timings"] = [round(t, 3) for t in worker.timings]
            worker.close()
            shutil.rmtree(scratch, ignore_errors=True)
        return result

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(one, list(enumerate(specs))))
    results.sort(key=lambda r: r["index"])
    return results
