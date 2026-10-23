import json
import math
import os
import select
import signal
import statistics
import struct
import subprocess
import sys
import time

import numpy as np


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import graph_spec

_HDR = struct.Struct("<4sQ")

N_WORKERS = 8
N_CPUS = 8
WARMUPS = 3
REPS = 15
BUILD_TIMEOUT_SEC = 20.0
CALL_TIMEOUT_SEC = 120.0
ABS_TOL = 1e-3
REL_TOL = 1e-3
AGENT_UID = 1001


class WorkerError(RuntimeError):
    """Any failure of a child: crash, protocol violation, timeout, bad output."""


def _uid_of(pid):
    try:
        with open("/proc/%d/status" % pid, "rb") as fh:
            for line in fh:
                if line.startswith(b"Uid:"):
                    return int(line.split()[1])
    except (OSError, IndexError, ValueError):
        pass
    return None


def _tasks_stopped(pids):
    for pid in pids:
        try:
            tids = os.listdir("/proc/%d/task" % pid)
        except OSError:
            continue
        for tid in tids:
            try:
                with open("/proc/%d/task/%s/stat" % (pid, tid), "rb") as fh:
                    state = fh.read().rsplit(b")", 1)[1].split()[0]
            except (OSError, IndexError):
                continue
            if state not in (b"T", b"Z", b"X"):
                return False
    return True


class Child:
    """A worker subprocess, SIGSTOPped whenever it is not the one being timed."""

    def __init__(self, kind, rundir, python, launch_prefix=(), env=None,
                 popen_kwargs=None):
        self.kind = kind
        argv = list(launch_prefix) + [python, "-I", "-B",
                                      os.path.join(rundir, "worker.py"), kind]
        self.proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, cwd=rundir, env=env, close_fds=True,
            start_new_session=True, **(popen_kwargs or {}))
        self.rfd = self.proc.stdout.fileno()
        self.wfd = self.proc.stdin.fileno()
        self.stopped = False
        self._pids = None
        self._expect(b"HI", BUILD_TIMEOUT_SEC)


    def _send(self, cmd, payload=b""):
        buf = _HDR.pack(cmd.ljust(4, b"\0"), len(payload)) + payload
        off = 0
        deadline = time.monotonic() + CALL_TIMEOUT_SEC
        while off < len(buf):
            if time.monotonic() > deadline:
                raise WorkerError("%s: timed out sending a request" % self.kind)
            _, w, _ = select.select([], [self.wfd], [], 1.0)
            if not w:
                continue
            try:
                off += os.write(self.wfd, buf[off:off + (1 << 20)])
            except OSError as exc:
                raise WorkerError("%s: pipe closed while sending (%s)" % (self.kind, exc))

    def _read_exactly(self, n, deadline):
        out = bytearray()
        while len(out) < n:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WorkerError("%s: timed out waiting for a response" % self.kind)
            r, _, _ = select.select([self.rfd], [], [], min(1.0, remaining))
            if not r:
                continue
            chunk = os.read(self.rfd, n - len(out))
            if not chunk:
                raise WorkerError("%s: exited before responding (rc=%s)"
                                  % (self.kind, self.proc.poll()))
            out += chunk
        return bytes(out)

    def _expect(self, cmd, timeout):
        deadline = time.monotonic() + timeout
        got, ln = _HDR.unpack(self._read_exactly(_HDR.size, deadline))
        payload = self._read_exactly(ln, deadline) if ln else b""
        got = got.rstrip(b"\0")
        if got != cmd:
            raise WorkerError("%s: expected %r, got %r" % (self.kind, cmd, got))
        return payload


    def _signal(self, sig):
        try:
            os.killpg(self.proc.pid, sig)
        except OSError:
            pass

    def tree_pids(self, refresh=False):
        """The child's own process tree, including anything it started itself."""
        if self._pids is not None and not refresh:
            return self._pids
        seen, stack = [], [self.proc.pid]
        while stack:
            pid = stack.pop()
            seen.append(pid)
            try:
                for tid in os.listdir("/proc/%d/task" % pid):
                    with open("/proc/%d/task/%s/children" % (pid, tid)) as fh:
                        stack.extend(int(v) for v in fh.read().split())
            except (OSError, ValueError):
                pass
        self._pids = seen
        return seen

    def freeze(self):
        """SIGSTOP, then wait until every thread has really stopped.

        Returning early would let this child's threads -- BLAS worker threads busy-spin
        for a while after the last call returns -- drain into the other child's timed
        region, which would bias whichever executor is measured second.
        """
        if not self.stopped:
            self._signal(signal.SIGSTOP)
            self.stopped = True
            deadline = time.monotonic() + 2.0
            while not _tasks_stopped(self.tree_pids()):
                if time.monotonic() > deadline:
                    break
                time.sleep(0.0002)

    def thaw(self):
        if self.stopped:
            self._signal(signal.SIGCONT)
            self.stopped = False
        self._send(b"PING")
        self._expect(b"PONG", CALL_TIMEOUT_SEC)


    def build(self, seed):
        t0 = time.perf_counter()
        self._send(b"BLD", str(int(seed)).encode())
        self._expect(b"RDY", BUILD_TIMEOUT_SEC)
        return time.perf_counter() - t0

    def call(self, xb):
        """Round-trip one input.  Returns (seconds, header, raw output bytes)."""
        t0 = time.perf_counter()
        self._send(b"CALL", xb)
        payload = self._expect(b"OUT", CALL_TIMEOUT_SEC)
        dt = time.perf_counter() - t0
        head, _, body = payload.partition(b"|")
        return dt, head.decode("utf-8", "replace"), body

    def stderr_tail(self, limit=2000):
        self._signal(signal.SIGCONT)
        self._signal(signal.SIGKILL)
        try:
            return self.proc.stderr.read().decode("utf-8", "replace")[-limit:]
        except Exception:
            return ""

    def close(self):
        self._signal(signal.SIGCONT)
        self.stopped = False
        try:
            self._send(b"EXIT")
            self.proc.wait(timeout=10)
        except Exception:
            self._signal(signal.SIGKILL)
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            try:
                stream.close()
            except Exception:
                pass


def _quiesce(active):
    """Stop every unprivileged process that is not part of `active`'s tree.

    An executor that detaches a helper into its own session would otherwise keep burning
    cycles while the other executor is on the clock.  Only the privileged harness can do
    this, so the developer-side harness (which runs unprivileged) skips it; the minimum
    statistic is what protects a measurement there.
    """
    if os.geteuid() != 0:
        return 0
    keep_pids = set(active.tree_pids(refresh=True))
    strays = 0
    try:
        pids = [int(p) for p in os.listdir("/proc") if p.isdigit()]
    except OSError:
        return 0
    for pid in pids:
        if pid in keep_pids or _uid_of(pid) != AGENT_UID:
            continue
        try:
            os.kill(pid, signal.SIGSTOP)
            strays += 1
        except OSError:
            pass
    return strays


def _check_output(head, body, spec):
    """Validate a returned tensor.  Returns the array or raises WorkerError."""
    T, d = spec["T"], spec["d_model"]
    try:
        ndim, dtype, dims = head.split(",")
        dims = tuple(int(v) for v in dims.split("x")) if dims else ()
    except ValueError:
        raise WorkerError("unparseable output header %r" % head)
    if int(ndim) != 2 or dims != (T, d):
        raise WorkerError("output shape %r, expected (%d, %d)" % (dims, T, d))
    if dtype != "float32":
        raise WorkerError("output dtype %r, expected float32" % dtype)
    if len(body) != T * d * 4:
        raise WorkerError("output payload is %d bytes, expected %d"
                          % (len(body), T * d * 4))
    return np.frombuffer(body, dtype=np.float32).reshape(T, d)


def _errors(y, y_ref):
    a = y.astype(np.float64)
    b = y_ref.astype(np.float64)
    if not np.all(np.isfinite(a)):
        return float("inf"), float("inf")
    diff = np.abs(a - b)
    max_abs = float(diff.max()) if diff.size else 0.0
    denom = float(np.linalg.norm(b))
    rel = float(np.linalg.norm(diff) / denom) if denom > 0 else max_abs
    return max_abs, rel


def measure_instance(seed, ref, sub, warmups=WARMUPS, reps=REPS):
    """Time one instance.  Returns a record; raises WorkerError on any child failure."""
    spec = graph_spec.sample_spec(seed)
    xbs = [np.ascontiguousarray(x).tobytes()
           for x in graph_spec.build_inputs(spec, warmups + reps)]
    ref.freeze()
    sub.thaw()
    build_sub = sub.build(seed)
    sub.freeze()
    ref.thaw()
    build_ref = ref.build(seed)
    ref.freeze()

    times = {id(ref): [], id(sub): []}
    max_abs = 0.0
    max_rel = 0.0
    strays = 0
    for i, xb in enumerate(xbs):

        first, second = (ref, sub) if i % 2 == 0 else (sub, ref)
        got = {}
        for child in (first, second):
            other = sub if child is ref else ref
            other.freeze()
            strays += _quiesce(child)
            child.thaw()
            dt, head, body = child.call(xb)
            child.freeze()
            got[id(child)] = (dt, head, body)
        y_ref = _check_output(got[id(ref)][1], got[id(ref)][2], spec)
        y_sub = _check_output(got[id(sub)][1], got[id(sub)][2], spec)
        a, r = _errors(y_sub, y_ref)
        max_abs = max(max_abs, a)
        max_rel = max(max_rel, r)
        if i >= warmups:
            times[id(ref)].append(got[id(ref)][0])
            times[id(sub)].append(got[id(sub)][0])

    ref_t, sub_t = times[id(ref)], times[id(sub)]
    ref_best, sub_best = min(ref_t), min(sub_t)
    equivalent = bool(max_abs <= ABS_TOL and max_rel <= REL_TOL)
    return {
        "seed": int(seed),
        "shape": {k: spec[k] for k in
                  ("d_model", "d_ff", "n_layers", "n_heads", "n_kv_heads", "T")},
        "ref_best_s": ref_best,
        "sub_best_s": sub_best,
        "ref_median_s": statistics.median(ref_t),
        "sub_median_s": statistics.median(sub_t),
        "ref_jitter": statistics.median(ref_t) / ref_best - 1.0,
        "sub_jitter": statistics.median(sub_t) / sub_best - 1.0,
        "build_ref_s": build_ref,
        "build_sub_s": build_sub,
        "strays_stopped": strays,
        "max_abs_err": max_abs,
        "rel_fro_err": max_rel,
        "equivalent": equivalent,
        "ratio": (ref_best / sub_best) if (equivalent and sub_best > 0) else 0.0,
    }


def child_env():
    """The environment both children get.  BLAS gets the full worker budget."""
    return {
        "PATH": "/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME": "/tmp",
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "OPENBLAS_NUM_THREADS": str(N_WORKERS),
        "OMP_NUM_THREADS": str(N_WORKERS),
        "MKL_NUM_THREADS": str(N_WORKERS),
        "NUMEXPR_NUM_THREADS": str(N_WORKERS),
        "EXEC_N_WORKERS": str(N_WORKERS),
        "EXEC_N_CPUS": str(N_CPUS),
    }


def measure(seeds, rundir, python=sys.executable, launch_prefix=(),
            popen_kwargs=None):
    """Measure every instance and aggregate.  Returns the metric record."""
    env = child_env()
    ref = sub = None
    per_instance = []
    failure = None
    stderr_tail = ""
    try:





        ref = Child("ref", rundir, python, (), env, None)
        sub = Child("sub", rundir, python, launch_prefix, env, popen_kwargs)
        for seed in seeds:
            per_instance.append(measure_instance(int(seed), ref, sub))
    except WorkerError as exc:
        failure = str(exc)
    except Exception as exc:
        failure = "%s: %s" % (type(exc).__name__, exc)
    finally:
        if failure is not None and sub is not None:
            stderr_tail = sub.stderr_tail()
        for child in (ref, sub):
            if child is not None:
                try:
                    child.close()
                except Exception:
                    pass

    n_ok = sum(1 for r in per_instance if r["equivalent"])
    valid = bool(failure is None and len(per_instance) == len(seeds) == n_ok)
    if valid:
        metric = math.exp(
            sum(math.log(r["ratio"]) for r in per_instance) / len(per_instance))
    else:
        metric = 0.0
    return {
        "metric_name": "geometric_mean_speedup",
        "metric": metric,
        "valid": valid,
        "failure": failure,
        "stderr_tail": stderr_tail,
        "n_instances": len(seeds),
        "n_equivalent": n_ok,
        "per_instance": per_instance,
    }


def dump(record, path):
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2, sort_keys=True)
