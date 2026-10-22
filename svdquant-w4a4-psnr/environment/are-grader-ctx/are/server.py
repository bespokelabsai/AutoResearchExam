"""ARE 1.1 grader sidecar.

One image, two roles:

- ARE_ROLE=val  ("grader" service). Reachable from the agent. Serves MCP tools and a
  REST fallback. On submit it snapshots the shared deliverable volume, grades the
  snapshot with the task's public grader (intermediate.sh), and returns only that
  score. Every snapshot is also sent to the test grader; the hidden score goes to a
  root-only ledger and never back to the agent.
- ARE_ROLE=test ("test-grader" service). On a private network that only the val
  grader can reach. Grades snapshots with the task's private grader (test.sh).
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import io
import secrets
import json
import os
import queue
import shutil
import subprocess
import tarfile
import threading
import time
import urllib.request
from pathlib import Path

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

ROLE = os.environ.get("ARE_ROLE", "val")
PORT = int(os.environ.get("ARE_PORT", "8000"))
STATE = Path(os.environ.get("ARE_STATE_DIR", "/var/lib/are"))
ARTIFACT_PATHS = [p for p in os.environ.get("ARE_ARTIFACT_PATHS", "/app/output").split(":") if p]
# The shared deliverable volume. Its parent is root-only, so submitted code (which every
# task grader runs as the unprivileged agent user) cannot reach it to write back to the agent.
SUBMISSION_ROOT = Path(os.environ.get("ARE_SUBMISSION_ROOT", "/run/are-deliverable"))
AGENT_UID = int(os.environ.get("ARE_AGENT_UID", "1001"))
BUDGET_SEC = float(os.environ.get("ARE_BUDGET_SEC", "14400"))
GRADE_TIMEOUT_SEC = float(os.environ.get("ARE_GRADE_TIMEOUT_SEC", "1800"))
TEST_GRADER_URL = os.environ.get("ARE_TEST_GRADER_URL", "http://test-grader:8001")
FEEDBACK_TAIL = int(os.environ.get("ARE_FEEDBACK_TAIL_CHARS", "8192"))
SCRIPT = "/tests/intermediate.sh" if ROLE == "val" else "/tests/test.sh"
# Shared only by the two graders (a volume the agent's container never mounts). The test
# grader writes a fresh token at startup; every internal request must carry it, so the
# agent cannot reach the test grader or the internal endpoints even on a shared network.
TOKEN_FILE = Path(os.environ.get("ARE_TOKEN_FILE", "/run/are-secret/token"))
TOKEN = ""


def _authorized(request: Request) -> bool:
    """Internal endpoints need the graders' token; an unset token never matches."""
    return bool(TOKEN) and secrets.compare_digest(request.headers.get("x-are-token", ""), TOKEN)


def _net_isolation_available() -> bool:
    try:
        return subprocess.run(["unshare", "-n", "true"], capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False


NET_ISOLATED = False  # set at startup: grader scripts run in an empty network namespace

T0 = time.time()
_lock = threading.Lock()


def _elapsed() -> float:
    return round(time.time() - T0, 1)


def _ledger(event: dict) -> None:
    event = {"t": _elapsed(), "wall": time.time(), **event}
    with _lock, open(STATE / "ledger.jsonl", "a") as fh:
        fh.write(json.dumps(event) + "\n")


def _kill_grading_processes(proc: subprocess.Popen) -> None:
    """Kill everything a timed-out grade started, including processes that left its group."""
    try:
        os.killpg(proc.pid, 9)
    except ProcessLookupError:
        pass
    _kill_other_processes()


def _kill_other_processes() -> None:
    """Kill every process run by the agent user. Every task grader runs submitted code as
    that user, so this stops all graded code (detached or not) and nothing else: the grader
    server, the task's own grader and Harbor's hooks run as root."""
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid == AGENT_UID:
                os.kill(int(entry.name), 9)
        except (ProcessLookupError, PermissionError, FileNotFoundError):
            pass


# Places the agent user (who runs submitted code inside every task grader) can write.
_BASELINE: set[str] = set()  # every path that existed when the grader started (its image)


def _walk_one_fs(base: Path):
    """Every path under base on base's own filesystem (skips /proc, /sys and other mounts)."""
    dev = base.stat().st_dev
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if os.lstat(os.path.join(root, d)).st_dev == dev]
        for name in dirs + files:
            yield Path(root) / name


def _all_paths() -> set[str]:
    out = set()
    for base in (Path("/"), Path("/dev/shm")):
        if base.is_dir():
            out.update(str(q) for q in _walk_one_fs(base))
    return out


def _protected(path: str) -> bool:
    keep = [str(STATE), "/opt/are", str(TOKEN_FILE.parent)]
    return any(path == k or path.startswith(k + "/") for k in keep)


def _fresh_state() -> None:
    """Start every grade from the grader image's own files, as ARE 1.0's fresh verifier
    container did. The grader container lives for the whole run, so before each grade
    stop anything an earlier grade left running and delete every file it created,
    whoever wrote it. The deliverable volume and the token are separate mounts and are
    never walked; the grader's own state is kept."""
    _kill_other_processes()
    # System V shared memory, semaphores and queues outlive processes; this server uses none.
    subprocess.run(["ipcrm", "--all"], capture_output=True)
    for path in sorted(_all_paths() - _BASELINE, key=lambda q: q.count("/"), reverse=True):
        if _protected(path):
            continue
        try:
            q = Path(path)
            if q.is_dir() and not q.is_symlink():
                shutil.rmtree(q, ignore_errors=True)
            else:
                q.unlink()
        except OSError:
            pass


def grade_snapshot(tar_bytes: bytes, tag: str) -> dict:
    """Extract a snapshot over the artifact paths and run this role's grader script."""
    _fresh_state()
    for path in ARTIFACT_PATHS:
        p = Path(path)
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tf:
        tf.extractall("/", filter="tar")
    logs = Path("/logs/verifier")
    if logs.exists():
        shutil.rmtree(logs)
    logs.mkdir(parents=True)
    started = time.time()
    # Submitted code runs with no network, as in ARE 1.0's verifier.
    cmd = (["unshare", "-n"] if NET_ISOLATED else []) + ["bash", SCRIPT]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            cwd="/tests", start_new_session=True)
    try:
        out, err = proc.communicate(timeout=GRADE_TIMEOUT_SEC)
        error = None
    except subprocess.TimeoutExpired:
        _kill_grading_processes(proc)
        out, err = proc.communicate()
        error = f"grader timed out after {GRADE_TIMEOUT_SEC:.0f}s"
    output = (out or "") + (err or "")
    # Nothing a grade started may keep running after it, not even detached processes.
    _kill_other_processes()
    score = 0.0
    reward_file = logs / "reward.txt"
    if reward_file.exists():
        try:
            score = float(reward_file.read_text().strip() or 0)
        except ValueError:
            error = error or "unparseable reward"
    metric = None
    if (logs / "metric.json").exists():
        try:
            metric = json.loads((logs / "metric.json").read_text())
        except ValueError:
            pass
    (STATE / "logs").mkdir(exist_ok=True)
    (STATE / "logs" / f"{tag}.log").write_text(output)
    return {
        "score": score,
        "metric": metric,
        "error": error,
        "net_isolated": NET_ISOLATED,
        "seconds": round(time.time() - started, 1),
        # What the agent sees, as in ARE 1.0: the grader's stdout tail (not stderr).
        "output_tail": (out or "")[-FEEDBACK_TAIL:],
    }


mcp = FastMCP("are")


# ---------------------------------------------------------------- test role


_grade_lock = threading.Lock()


def _grade_one_at_a_time(tar_bytes: bytes, tag: str) -> dict:
    # A request that timed out on the caller's side may still be grading; the next one
    # waits for it rather than sharing /app/output and /logs/verifier with it.
    with _grade_lock:
        return grade_snapshot(tar_bytes, tag)


if ROLE == "test":

    @mcp.custom_route("/grade", methods=["POST"])
    async def http_grade(request: Request) -> Response:
        if not _authorized(request):
            return JSONResponse({"error": "forbidden"}, status_code=403)
        tag = request.query_params.get("tag", "x")
        body = await request.body()
        result = await asyncio.to_thread(_grade_one_at_a_time, body, f"{tag}-test")
        return JSONResponse(result)


# ----------------------------------------------------------------- val role


class Board:
    """Submissions and their grading state. Hidden scores are never stored here."""

    def __init__(self) -> None:
        self.subs: dict[int, dict] = {}
        self.val_q: queue.Queue[int] = queue.Queue()
        self.next_id = 1
        self.test_pending = 0
        self.last_digest: str | None = None

    def public(self, sid: int) -> dict:
        s = self.subs[sid]
        out = {k: s[k] for k in ("id", "note", "submitted_at_s", "status")}
        if s["status"] == "graded":
            v = s["val"]
            out.update(
                validation_score=v["score"],
                error=v["error"],
                grading_seconds=v["seconds"],
                grader_output_tail=v["output_tail"],
            )
        return out

    def best(self) -> dict | None:
        graded = [s for s in self.subs.values() if s["status"] == "graded"]
        if not graded:
            return None
        b = max(graded, key=lambda s: (s["val"]["score"], -s["id"]))
        return {"id": b["id"], "validation_score": b["val"]["score"]}


board = Board()


def snapshot() -> bytes:
    """Deterministic tar.gz of the deliverable: same content, same bytes (gzip mtime 0)."""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tf:
        for path in ARTIFACT_PATHS:
            src = SUBMISSION_ROOT / path.lstrip("/")
            tf.add(str(src), arcname=path.lstrip("/"))
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="wb", mtime=0) as gz:
        gz.write(raw.getvalue())
    return out.getvalue()


def do_submit(note: str, final: bool = False) -> dict:
    with _lock:
        pending = [s["id"] for s in board.subs.values() if s["status"] != "graded"]
        if pending and not final:
            return {
                "error": f"Submission {pending[0]} is still being graded. One submission is "
                         f"graded at a time: check result({pending[0]}) and submit again "
                         "once it has its score.",
                "pending_id": pending[0],
            }
        sid = board.next_id
        board.next_id += 1
        board.subs[sid] = {"id": sid, "note": note[:500], "submitted_at_s": _elapsed(),
                           "status": "queued"}
    try:
        data = snapshot()
        digest = hashlib.sha256(data).hexdigest()
        (STATE / "snapshots" / f"{sid:04d}.tar.gz").write_bytes(data)
        _ledger({"event": "submitted", "id": sid, "note": note[:500], "bytes": len(data),
                 "sha256": digest, "final": final})
    except Exception as exc:
        with _lock:
            board.subs.pop(sid, None)  # release the slot; the id is not reused
        return {"error": f"Could not snapshot the deliverable ({exc}). Nothing was submitted. "
                         "Make sure nothing is writing to it, then submit again."}
    board.last_digest = digest
    with _lock:
        board.test_pending += 1
    board.val_q.put(sid)
    return {"id": sid, "status": "queued",
            "hint": "Grading has started. Check result(id) for the validation score."}


def final_submit() -> dict:
    """After the agent stops: grade the final workspace unless it was already submitted."""
    try:
        data = snapshot()
    except Exception as exc:
        return {"error": f"final snapshot failed: {exc}"}
    if board.subs and hashlib.sha256(data).hexdigest() == board.last_digest:
        return {"skipped": "final workspace equals the last submission"}
    return do_submit("final workspace (automatic)", final=True)


def grade_worker() -> None:
    """Grade each submission on val, then on test, one submission at a time.

    The val score is published as soon as it is ready. The next submission's val grade
    waits until this test grade is done, so the two graders never run at the same time.
    """
    while True:
        sid = board.val_q.get()
        board.subs[sid]["status"] = "grading"
        data = b""
        try:
            data = (STATE / "snapshots" / f"{sid:04d}.tar.gz").read_bytes()
            res = grade_snapshot(data, f"{sid:04d}-val")
        except Exception as exc:
            res = {"score": 0.0, "metric": None, "error": f"grader failed: {exc}",
                   "net_isolated": NET_ISOLATED, "seconds": 0.0, "output_tail": ""}
        board.subs[sid]["val"] = res
        board.subs[sid]["status"] = "graded"
        _ledger({"event": "val", "id": sid, **{k: res[k] for k in ("score", "metric", "error", "seconds", "net_isolated")}})
        test_grade(sid, data)


def test_grade(sid: int, data: bytes) -> None:
    try:
        _test_grade(sid, data)
    finally:
        with _lock:
            board.test_pending -= 1


def _test_grade(sid: int, data: bytes) -> None:
    started = time.time()
    try:
        req = urllib.request.Request(
            f"{TEST_GRADER_URL}/grade?tag={sid:04d}", data=data, method="POST",
            headers={"x-are-token": TOKEN},
        )
        with urllib.request.urlopen(req, timeout=2 * GRADE_TIMEOUT_SEC + 300) as resp:
            res = json.loads(resp.read())
    except Exception as exc:  # recorded, never shown to the agent
        res = {"score": None, "metric": None, "error": f"test grader: {exc}",
               "seconds": round(time.time() - started, 1)}
    _ledger({"event": "test", "id": sid, **{k: res.get(k) for k in ("score", "metric", "error", "seconds", "net_isolated")}})


def with_time(out: dict) -> dict:
    """Wall-clock accounting on every reply, as the v1 harness gave each phase."""
    out["total_budget_seconds"] = BUDGET_SEC
    out["time_remaining_seconds"] = round(max(0.0, BUDGET_SEC - _elapsed()), 1)
    return out


def status_dict() -> dict:
    return {
        "submissions": len(board.subs),
        "queued_or_grading": sum(s["status"] != "graded" for s in board.subs.values()),
        "best": board.best(),
    }


def current_result(sid: int) -> dict:
    return board.public(sid)


if ROLE == "val":

    @mcp.tool
    async def submit(note: str = "") -> dict:
        """Submit the current deliverable for grading.

        Snapshots the deliverable directory right now (later edits do not affect this
        submission) and starts grading it on the public validation split. Returns at
        once with a submission id. One submission is graded at a time: check `result`
        and submit again once the previous submission has its score. `note` is a short
        description of what changed.
        """
        return with_time(await asyncio.to_thread(do_submit, note))

    @mcp.tool
    async def result(id: int) -> dict:
        """The current state of a submission: queued, grading, or graded with its
        validation score and grader output. Returns immediately."""
        if id not in board.subs:
            return {"error": f"unknown submission id {id}"}
        return with_time(current_result(id))

    @mcp.tool
    async def status() -> dict:
        """Submission count, whether one is being graded, the best validation score,
        and the time budget and time remaining."""
        return with_time(status_dict())

    @mcp.custom_route("/submit", methods=["POST"])
    async def http_submit(request: Request) -> Response:
        note = request.query_params.get("note", "")
        return JSONResponse(with_time(await asyncio.to_thread(do_submit, note)))

    @mcp.custom_route("/result/{sid}", methods=["GET"])
    async def http_result(request: Request) -> Response:
        sid = int(request.path_params["sid"])
        if sid not in board.subs:
            return JSONResponse({"error": f"unknown submission id {sid}"}, status_code=404)
        return JSONResponse(with_time(current_result(sid)))

    @mcp.custom_route("/status", methods=["GET"])
    async def http_status(request: Request) -> Response:
        return JSONResponse(with_time(status_dict()))

    @mcp.custom_route("/final", methods=["POST"])
    async def http_final(request: Request) -> Response:
        if not _authorized(request):
            return JSONResponse({"error": "forbidden"}, status_code=403)
        # Collect hook, after the agent has stopped.
        return JSONResponse(await asyncio.to_thread(final_submit))

    @mcp.custom_route("/drain", methods=["GET"])
    async def http_drain(request: Request) -> Response:
        if not _authorized(request):
            return JSONResponse({"error": "forbidden"}, status_code=403)
        # Used by the verifier collect hook after the agent has stopped.
        pending = sum(s["status"] != "graded" for s in board.subs.values())
        return JSONResponse({"val_pending": pending, "test_pending": board.test_pending})


def _gpus() -> list[str]:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20)
        return [l for l in out.stdout.splitlines() if l.strip()]
    except Exception:
        return []


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> Response:
    return JSONResponse({"ok": True, "role": ROLE, "gpus": await asyncio.to_thread(_gpus)})


def _setup_token() -> None:
    global TOKEN
    if ROLE == "test":
        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(TOKEN_FILE.parent, 0o700)
        TOKEN = secrets.token_hex(32)
        TOKEN_FILE.write_text(TOKEN)
        os.chmod(TOKEN_FILE, 0o600)
    else:
        for _ in range(300):  # the test grader is healthy before this one starts
            if TOKEN_FILE.exists() and TOKEN_FILE.read_text().strip():
                break
            time.sleep(1)
        TOKEN = TOKEN_FILE.read_text().strip()
    if not TOKEN:
        raise SystemExit("ARE: empty internal token; refusing to start")


def main() -> None:
    global NET_ISOLATED
    NET_ISOLATED = _net_isolation_available()
    if not NET_ISOLATED:
        # Graded code must never have a network path (for example back through /submit).
        raise SystemExit("ARE grader: cannot run grades without network access "
                         "(needs `unshare -n`, i.e. CAP_SYS_ADMIN); refusing to start")
    STATE.mkdir(parents=True, exist_ok=True)
    _BASELINE.update(_all_paths())
    _setup_token()
    STATE.mkdir(parents=True, exist_ok=True)
    os.chmod(STATE, 0o700)
    (STATE / "snapshots").mkdir(exist_ok=True)
    if ROLE == "val":
        # The deliverable volume starts empty and root-owned; hand it to the agent user.
        SUBMISSION_ROOT.mkdir(parents=True, exist_ok=True)
        os.chmod(SUBMISSION_ROOT, 0o700)
        for path in ARTIFACT_PATHS:
            src = SUBMISSION_ROOT / path.lstrip("/")
            src.mkdir(parents=True, exist_ok=True)
            os.chown(src, AGENT_UID, AGENT_UID)
        (STATE / "meta.json").write_text(json.dumps(
            {"t0_wall": T0, "budget_seconds": BUDGET_SEC, "artifact_paths": ARTIFACT_PATHS}
        ))
        threading.Thread(target=grade_worker, daemon=True).start()
    mcp.run(transport="http", host="0.0.0.0", port=PORT, show_banner=False,
            uvicorn_config={"timeout_keep_alive": 86400})


if __name__ == "__main__":
    main()
