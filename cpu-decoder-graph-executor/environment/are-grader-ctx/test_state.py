import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import grader_core

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
DELIVERABLE = "/app/output"
ENTRY_POINT = "executor.py"
LOG_DIR = "/logs/verifier"
CORE_MODULES = ("worker.py", "graph_spec.py", "reference_executor.py")
SPLIT = os.environ.get("GRADE_SPLIT", "final")



PRIVATE = SPLIT == "final"
PREFLIGHT_TIMEOUT_SEC = 120.0

_RESULT = {}


def _sealed_seeds():
    path = os.path.join(TESTS_DIR, "hidden_data", SPLIT, "seeds.json")
    with open(path) as fh:
        return [int(s) for s in json.load(fh)["seeds"]]


def _privilege_drop():
    """Run candidate code as uid 1001, never as root.

    Returns (argv prefix, extra Popen kwargs).  `runuser` is the normal path; the
    Popen user=/group= form is a binary-free fallback so that a base image without
    util-linux cannot silently turn every rollout into a zero.
    """
    runuser = shutil.which("runuser")
    if runuser:
        return [runuser, "-u", "agent", "--"], {}
    setpriv = shutil.which("setpriv")
    if setpriv:
        return [setpriv, "--reuid=1001", "--regid=1001", "--clear-groups", "--"], {}
    return [], {"user": 1001, "group": 1001}


def _reject_links(root):
    """Refuse a submission tree containing anything but plain files and directories.

    A symlink or a hard link inside the deliverable would let a privileged copy
    materialise a file from outside the deliverable into the directory the score is
    computed from.
    """
    if os.path.islink(root):
        raise RuntimeError("%s is a symlink" % root)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in dirnames + filenames:
            path = os.path.join(dirpath, name)
            st = os.lstat(path)
            if stat.S_ISLNK(st.st_mode):
                raise RuntimeError("symlink in the deliverable: %s" % path)
            if stat.S_ISREG(st.st_mode) and st.st_nlink > 1:
                raise RuntimeError("hard link in the deliverable: %s" % path)
            if not (stat.S_ISREG(st.st_mode) or stat.S_ISDIR(st.st_mode)):
                raise RuntimeError("not a regular file or directory: %s" % path)


def _stage():
    """Copy the verifier's own modules and the submission into a root-owned run dir."""
    run = tempfile.mkdtemp(prefix="grade_", dir="/tmp")
    for name in CORE_MODULES:
        shutil.copy2(os.path.join(TESTS_DIR, name), os.path.join(run, name))
    dst = os.path.join(run, "submission")
    _reject_links(DELIVERABLE)


    shutil.copytree(DELIVERABLE, dst, symlinks=True)
    for dirpath, dirnames, filenames in os.walk(run):
        os.chown(dirpath, 0, 0)
        os.chmod(dirpath, 0o755)
        for name in filenames:
            path = os.path.join(dirpath, name)
            os.chown(path, 0, 0)
            os.chmod(path, 0o644)
    os.chmod(run, 0o755)
    return run


def _preflight(run, prefix, popen_kwargs):
    """Import the entry point in a throwaway unprivileged process, for a clear message."""


    code = ("import sys; sys.path.insert(0, %r); sys.path.insert(0, %r); "
            "import executor; assert callable(executor.build_executor)"
            % (run, os.path.join(run, "submission")))
    proc = subprocess.run(prefix + [sys.executable, "-I", "-B", "-c", code],
                          capture_output=True, text=True, cwd=run,
                          timeout=PREFLIGHT_TIMEOUT_SEC, check=False,
                          start_new_session=True, **popen_kwargs)
    return proc.returncode, (proc.stderr or "")[-2000:]


def _grade():
    if _RESULT:
        return _RESULT
    record = {
        "metric_name": "geometric_mean_speedup",
        "metric": 0.0,
        "valid": False,
        "failure": None,
        "split": SPLIT,
        "per_instance": [],
        "n_instances": 0,
        "n_equivalent": 0,
        "preflight_rc": None,
        "preflight_stderr": "",
    }
    run = None
    try:
        entry = os.path.join(DELIVERABLE, ENTRY_POINT)
        if not os.path.isfile(entry):
            raise RuntimeError("no entry point at %s" % entry)
        prefix, popen_kwargs = _privilege_drop()
        run = _stage()
        rc, err = _preflight(run, prefix, popen_kwargs)
        record["preflight_rc"] = rc
        record["preflight_stderr"] = err
        if rc != 0:
            raise RuntimeError("executor.py could not be imported (rc=%s)" % rc)
        measured = grader_core.measure(_sealed_seeds(), run, python=sys.executable,
                                       launch_prefix=prefix,
                                       popen_kwargs=popen_kwargs)
        record.update({k: measured[k] for k in
                       ("metric", "valid", "failure", "per_instance",
                        "n_instances", "n_equivalent")})
        record["stderr_tail"] = measured.get("stderr_tail", "")
    except Exception as exc:
        record["metric"] = 0.0
        record["valid"] = False
        record["failure"] = "%s: %s" % (type(exc).__name__, exc)
    finally:
        if run:
            shutil.rmtree(run, ignore_errors=True)

    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, "metric.json"), "w") as fh:
        json.dump(record, fh, indent=2, sort_keys=True)
    _RESULT.update(record)
    return record


def test_deliverable_is_present_and_importable():
    """The graded deliverable exists at the declared path and imports cleanly."""
    record = _grade()
    assert record["preflight_rc"] == 0, (
        "executor.py missing or not importable: %s\n%s"
        % (record["failure"], record.get("preflight_stderr", "")))


def test_output_matches_the_reference_on_every_sealed_instance():
    """Every sealed instance is executed and its output matches the reference."""
    record = _grade()
    assert record["n_instances"] > 0, "no instance completed: %s" % record["failure"]
    bad = [i for i, r in enumerate(record["per_instance"]) if not r["equivalent"]]
    assert not bad, (
        "outputs differ from the reference on instance(s) %s of %d (tolerances: "
        "max abs <= %g, relative Frobenius <= %g)"
        % (bad, record["n_instances"], grader_core.ABS_TOL, grader_core.REL_TOL))
    assert record["valid"], "measurement did not complete: %s" % record["failure"]


def test_speedup_metric_is_recorded():
    """The geometric-mean speedup is measured and written to metric.json."""
    record = _grade()
    assert os.path.isfile(os.path.join(LOG_DIR, "metric.json"))
    print("instances=%d  equivalent=%d  geometric-mean speedup=%.4f"
          % (record["n_instances"], record["n_equivalent"], record["metric"]))
    ratios = sorted(r["ratio"] for r in record["per_instance"])
    if ratios:
        print("per-instance speedup: min %.4f  median %.4f  max %.4f"
              % (ratios[0], ratios[len(ratios) // 2], ratios[-1]))
    if PRIVATE:

        for r in record["per_instance"]:
            print("  seed %6d T=%-4d d=%-5d L=%-3d ref %8.2f ms  sub %8.2f ms  ratio %.4f"
                  % (r["seed"], r["shape"]["T"], r["shape"]["d_model"],
                     r["shape"]["n_layers"], r["ref_best_s"] * 1e3,
                     r["sub_best_s"] * 1e3, r["ratio"]))
    assert record["metric"] >= 0.0
