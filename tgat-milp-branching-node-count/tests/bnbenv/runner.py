from __future__ import annotations

import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

from pyscipopt import SCIP_RESULT, Branchrule

from . import features, protocol
from .setcover import NODE_LIMIT, TIME_LIMIT, build_model

ENTRY_POINT = "policy.py"
STARTUP_LIMIT = 30.0
DECISION_READ_CHUNK = 4096


class PolicyChannel:
    """Owns the policy subprocess and the framed request/response cycle."""

    def __init__(self, submission_dir, host_pkg_root, python_exe=None,
                 launch_prefix=(), stderr_to=None, entry_point=ENTRY_POINT,
                 drop_to_uid=None):
        self.submission_dir = str(Path(submission_dir).resolve())
        self.entry_point = entry_point
        env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "PYTHONPATH": str(Path(host_pkg_root).resolve()),
            "PYTHONSAFEPATH": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HOME": "/home/agent",
            "TMPDIR": "/tmp",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
        read_fd, write_fd = os.pipe()
        os.set_inheritable(write_fd, True)
        env["BNB_REPLY_FD"] = str(write_fd)
        cmd = list(launch_prefix) + [
            python_exe or sys.executable, "-c",
            "from bnbenv.host import main; raise SystemExit(main())",
        ]

        def _drop():
            """Drop to the unprivileged uid before exec, in the child."""
            os.setgroups([])
            os.setgid(int(drop_to_uid))
            os.setuid(int(drop_to_uid))

        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=stderr_to if stderr_to is not None else subprocess.DEVNULL,
            env=env,
            pass_fds=(write_fd,),
            start_new_session=True,
            preexec_fn=(_drop if drop_to_uid and not launch_prefix else None),
            cwd="/tmp",
        )
        os.close(write_fd)
        self.reply = os.fdopen(read_fd, "rb")
        self.error = None

    def _send(self, payload):
        protocol.write_obs(self.proc.stdin, payload)

    def _recv(self, budget):
        """Read one newline-terminated reply, or None on timeout/death."""
        deadline = time.monotonic() + budget
        buf = bytearray()
        fd = self.reply.fileno()
        while b"\n" not in buf:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.error = "policy process exceeded its wall-clock budget"
                return None
            ready, _, _ = select.select([fd], [], [], min(remaining, 1.0))
            if not ready:
                continue
            chunk = os.read(fd, DECISION_READ_CHUNK)
            if not chunk:
                self.error = "policy process exited without answering"
                return None
            buf.extend(chunk)
            if len(buf) > 1 << 16:
                self.error = "policy process wrote an oversized reply"
                return None
        return bytes(buf).split(b"\n", 1)[0].decode("ascii", "replace")

    def start(self, budget=STARTUP_LIMIT):
        self._send({"cmd": "load", "submission_dir": self.submission_dir,
                    "entry_point": self.entry_point})
        reply = self._recv(budget)
        if reply != "ready":
            self.error = self.error or f"load failed ({reply})"
            return False
        self._send({"cmd": "new"})
        reply = self._recv(budget)
        if reply != "ok":
            self.error = self.error or f"Policy() failed ({reply})"
            return False
        return True

    def decide(self, obs, budget, n_cands):
        try:
            self._send({"cmd": "select", "obs": obs})
        except (BrokenPipeError, OSError):
            self.error = "policy process closed the channel"
            return None
        reply = self._recv(budget)
        if reply is None:
            return None
        if not reply.startswith("i "):
            self.error = f"policy error ({reply[:200]})"
            return None
        try:
            index = int(reply[2:])
        except ValueError:
            self.error = f"unparseable decision ({reply[:80]})"
            return None
        if not 0 <= index < n_cands:
            self.error = f"decision {index} outside [0, {n_cands})"
            return None
        return index

    def extra_processes(self):
        """Live processes in the policy session other than the policy itself."""
        try:
            sid = os.getsid(self.proc.pid)
        except (ProcessLookupError, PermissionError, OSError):
            return 0
        extra = 0
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            if pid == self.proc.pid:
                continue
            try:
                if os.getsid(pid) == sid:
                    extra += 1
            except (ProcessLookupError, PermissionError, OSError):
                continue
        return extra

    def close(self):
        try:
            self._send({"cmd": "quit"})
        except Exception:
            pass
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except Exception:
                pass
            try:
                self.proc.wait(timeout=5)
            except Exception:
                pass
        try:
            self.reply.close()
        except Exception:
            pass


class HostedBranchrule(Branchrule):
    """Hands every branching decision of this solve to the policy process."""

    def __init__(self, channel, deadline):
        super().__init__()
        self.channel = channel
        self.deadline = deadline
        self.decisions = 0
        self.failure = None

    def branchexeclp(self, allowaddcons):
        if self.failure is not None:
            return {"result": SCIP_RESULT.DIDNOTRUN}
        cands, cands_sol, _frac, ncands, _nprio, _nimpl = \
            self.model.getLPBranchCands()
        if ncands == 0:
            return {"result": SCIP_RESULT.DIDNOTRUN}
        cands = list(cands)[:ncands]
        obs = features.extract(self.model, cands, cands_sol)
        budget = self.deadline - time.monotonic()
        if budget <= 0:
            self.failure = "run exceeded its wall-clock budget"
            self.model.interruptSolve()
            return {"result": SCIP_RESULT.DIDNOTRUN}
        index = self.channel.decide(obs, budget, len(cands))
        if index is None:
            self.failure = self.channel.error or "policy failed"
            self.model.interruptSolve()
            return {"result": SCIP_RESULT.DIDNOTRUN}
        self.decisions += 1
        if self.decisions % 20 == 1 and self.channel.extra_processes():
            self.failure = "policy process spawned extra processes"
            self.model.interruptSolve()
            return {"result": SCIP_RESULT.DIDNOTRUN}
        self.model.branchVar(cands[index])
        return {"result": SCIP_RESULT.BRANCHED}


def run_one(seed, shift, submission_dir, host_pkg_root, python_exe=None,
            launch_prefix=(), stderr_to=None, node_limit=NODE_LIMIT,
            time_limit=TIME_LIMIT, startup_timeout=STARTUP_LIMIT,
            entry_point=ENTRY_POINT, drop_to_uid=None):
    """Solve one instance with the submission in charge of variable selection.

    Returns a record with the node count charged for this run: the solver's own
    total node count when it proved optimality, and `node_limit` otherwise.
    """
    record = {
        "seed": int(seed),
        "shift": int(shift),
        "nodes": int(node_limit),
        "raw_nodes": None,
        "status": "unstarted",
        "proved_optimal": False,
        "decisions": 0,
        "solve_seconds": 0.0,
        "error": None,
        "extra_processes": 0,
    }
    if not (Path(submission_dir) / entry_point).is_file():
        record["status"] = "missing_entry_point"
        record["error"] = f"no {entry_point} in {submission_dir}"
        return record

    channel = PolicyChannel(submission_dir, host_pkg_root,
                            python_exe=python_exe, launch_prefix=launch_prefix,
                            stderr_to=stderr_to, entry_point=entry_point,
                            drop_to_uid=drop_to_uid)
    try:
        if not channel.start(budget=startup_timeout):
            record["status"] = "startup_failed"
            record["error"] = channel.error
            return record

        model, _xs = build_model(seed, shift, node_limit=node_limit,
                                 time_limit=time_limit)
        started = time.monotonic()
        rule = HostedBranchrule(channel, started + time_limit)
        model.includeBranchrule(rule, "hosted", "policy-process branching",
                                priority=10_000_000, maxdepth=-1,
                                maxbounddist=1.0)
        model.optimize()
        record["solve_seconds"] = round(time.monotonic() - started, 3)
        record["status"] = model.getStatus()
        record["decisions"] = rule.decisions
        raw_nodes = int(model.getNTotalNodes())
        record["raw_nodes"] = raw_nodes
        record["extra_processes"] = channel.extra_processes()
        proved = (rule.failure is None and model.getStatus() == "optimal"
                  and raw_nodes <= node_limit
                  and record["extra_processes"] == 0)
        if rule.failure is not None:
            record["error"] = rule.failure
        elif record["extra_processes"] > 0:
            record["error"] = "policy process spawned extra processes"
        record["proved_optimal"] = bool(proved)
        record["nodes"] = raw_nodes if proved else int(node_limit)
        del model
        return record
    finally:
        channel.close()
