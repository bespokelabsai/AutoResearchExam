import json
import math
import os
import shutil
import stat
import subprocess
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
HIDDEN_ROOT = os.path.join(TESTS_DIR, "hidden_data")
HARNESS_SRC = os.path.join(TESTS_DIR, "harness")

SUBMISSION_SRC = "/app/output"
ENTRY_NAME = "advantage.py"
GRADE_ROOT = "/grade"
HARNESS_RUN = os.path.join(GRADE_ROOT, "harness")
SUBMISSION_RUN = os.path.join(GRADE_ROOT, "submission")
CKPT_DIR = os.path.join(GRADE_ROOT, "ckpt")
WORK_ROOT = os.path.join(GRADE_ROOT, "work")

AGENT_UID = 1001
AGENT_GID = 1001

GRADE_SPLIT = os.environ.get("GRADE_SPLIT", "final")
PUBLIC_STDOUT = GRADE_SPLIT == "intermediate"

RUN_WALL_CLOCK_CAP = 1500.0
RUN_KILL_GRACE = 120.0
EVAL_TIMEOUT = 900.0
CONCURRENT_RUNS = 3

M0 = 589.8015290641684
BASELINE_SCORE = 1334.5186897062024


def load_seeds():
    """The sealed seeds of the selected split. The two splits share no seed with each other."""
    hidden = os.path.join(HIDDEN_ROOT, GRADE_SPLIT, "seeds.json")
    with open(hidden, "r", encoding="utf-8") as fh:
        slice_ = json.load(fh)
    return [int(s) for s in slice_["train_seeds"]], [int(s) for s in slice_["eval_seeds"]]


def run_label(seed, index):
    """How one training run is named wherever the name can reach stdout.

    On the public split the seed value itself is withheld: an agent that learned the public seeds
    could train against those exact runs and tune the score it can see rather than the estimator.
    """
    return f"run{index}" if PUBLIC_STDOUT else str(seed)


def redact_seeds(text, seeds):
    """Strip seed values out of a diagnostic tail that the public split would otherwise print."""
    if not PUBLIC_STDOUT:
        return text
    for seed in seeds:
        text = text.replace(str(seed), "<seed>")
    return text


def graded_reward(metric, valid):
    """rational_squash: 0 at the trivial reference, 0.5 at the reference configuration, no cap."""
    if not valid or metric is None or not math.isfinite(metric):
        return 0.0
    x_ref = BASELINE_SCORE - M0
    if x_ref <= 0:
        raise AssertionError("malformed task: the reference does not beat the trivial baseline")
    u = max(0.0, metric - M0) / x_ref
    return u / (1.0 + u)


def submission_entry_present():
    entry = os.path.join(SUBMISSION_SRC, ENTRY_NAME)
    return os.path.isfile(entry) and os.path.getsize(entry) > 0


def reject_symlinks(base):
    """Refuse an agent-writable tree containing symlinks.

    The deliverable is written by the agent while the staging copy below runs as root: a symlink
    pointing at a sealed file would otherwise be dereferenced and its contents copied into the
    world-readable submission tree.
    """
    if os.path.islink(base):
        raise RuntimeError(f"the submission path {base} is a symlink; refusing to grade")
    for root, dirs, files in os.walk(base, followlinks=False):
        for name in dirs + files:
            path = os.path.join(root, name)
            if os.path.islink(path):
                raise RuntimeError(f"symlinks are not permitted in the submission tree: {path}")


def stage():
    """Lay out the grade-time tree with the ownership the two-process boundary depends on."""
    os.makedirs(GRADE_ROOT, mode=0o755, exist_ok=True)
    for entry in os.listdir(GRADE_ROOT):
        path = os.path.join(GRADE_ROOT, entry)
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            os.remove(path)
    os.chown(GRADE_ROOT, 0, 0)
    os.chmod(GRADE_ROOT, 0o755)
    shutil.copytree(HARNESS_SRC, HARNESS_RUN)
    reject_symlinks(SUBMISSION_SRC)
    shutil.copytree(SUBMISSION_SRC, SUBMISSION_RUN, symlinks=True)
    for base in (HARNESS_RUN, SUBMISSION_RUN):
        for root, dirs, files in os.walk(base):
            os.chown(root, 0, 0)
            os.chmod(root, 0o755)
            for name in dirs + files:
                path = os.path.join(root, name)
                if os.path.islink(path):
                    continue
                os.chown(path, 0, 0)
                os.chmod(path, 0o644 if os.path.isfile(path) else 0o755)
    os.makedirs(CKPT_DIR, mode=0o700, exist_ok=True)
    os.chown(CKPT_DIR, 0, 0)
    os.chmod(CKPT_DIR, 0o700)
    os.makedirs(WORK_ROOT, mode=0o755, exist_ok=True)


def candidate_work_dir(seed):
    work = os.path.join(WORK_ROOT, f"seed{seed}")
    os.makedirs(work, mode=0o700, exist_ok=True)
    os.chown(work, AGENT_UID, AGENT_GID)
    return work


def checkpoint_path(seed):
    return os.path.join(CKPT_DIR, f"policy_{seed}.pt")


def run_log_path(seed):
    """Diagnostics for one run. Lives in the root-only directory, so the run cannot unlink it."""
    return os.path.join(CKPT_DIR, f"run_{seed}.log")


def read_log_tail(path, limit=1200):
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            fh.seek(max(0, size - limit))
            return fh.read().decode("utf-8", "replace")
    except OSError:
        return ""


def train_command(seed, ckpt_fd):
    """The command line the candidate process runs. The seed is all it is handed."""
    return [
        sys.executable,
        os.path.join(HARNESS_RUN, "train.py"),
        "--seed", str(seed),
        "--submission", SUBMISSION_RUN,
        "--ckpt-fd", str(ckpt_fd),
    ]


def candidate_env(work):
    return {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": work,
        "TMPDIR": work,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }


def evaluate(ckpt, eval_seeds):
    """Roll the checkpoint out on the sealed evaluation seeds. Root only; no agent code runs."""
    proc = subprocess.run(
        [sys.executable, os.path.join(HARNESS_RUN, "eval_policy.py"),
         "--ckpt", ckpt, "--seeds", ",".join(str(s) for s in eval_seeds)],
        cwd=GRADE_ROOT,
        capture_output=True,
        text=True,
        timeout=EVAL_TIMEOUT,
        env={"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": GRADE_ROOT, "PYTHONHASHSEED": "0"},
    )
    if proc.returncode != 0:
        raise RuntimeError(f"evaluation failed: {proc.stderr[-500:]}")
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    returns = [float(x) for x in payload["returns"]]
    if len(returns) != len(eval_seeds) or not all(math.isfinite(r) for r in returns):
        raise RuntimeError("evaluation produced a malformed set of episode returns")
    return returns


def cleanup():
    shutil.rmtree(WORK_ROOT, ignore_errors=True)


def seal_check():
    """The sealed slice must be unreadable by the uid that runs the submitted estimator."""
    info = os.stat(os.path.join(HIDDEN_ROOT, GRADE_SPLIT))
    mode = stat.S_IMODE(info.st_mode)
    return {"hidden_mode": oct(mode), "hidden_uid": info.st_uid,
            "sealed": info.st_uid == 0 and not (mode & 0o077)}
