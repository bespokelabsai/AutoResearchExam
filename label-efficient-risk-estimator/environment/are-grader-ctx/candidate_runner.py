import argparse
import os
import signal
import socket
import struct
import sys
import time
import types

import numpy as np

_HDR = struct.Struct("<c")
_I32 = struct.Struct("<i")
_F64 = struct.Struct("<d")


def _sendall(sock, data):
    sock.sendall(data)


def _recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("scorer closed the label channel")
        buf += chunk
    return buf


def _request_label(sock, index):
    _sendall(sock, _HDR.pack(b"L") + _I32.pack(int(index)))
    return _I32.unpack(_recv_exact(sock, 4))[0]


def _run_one_seed(sock, code, module_path, submission_dir, pool, budget, seed):
    """Executed in the grandchild only. Never returns."""
    try:
        os.setpgid(0, 0)
    except OSError:
        pass
    try:
        sys.path.insert(0, submission_dir)
        mod = types.ModuleType("candidate_estimator")
        mod.__file__ = module_path
        exec(code, mod.__dict__)
        Estimator = mod.Estimator
        rng = np.random.default_rng([pool["_cseed"], seed])
        est = Estimator(pool["_view"], budget, rng)
        n_pool = pool["_view"]["n_pool"]
        for _ in range(budget):
            idx = est.next_index()
            idx = int(idx)
            if not (0 <= idx < n_pool):
                raise ValueError(f"next_index returned {idx}, outside [0, {n_pool})")
            label = _request_label(sock, idx)
            if label < 0:
                raise RuntimeError("label budget exhausted")
            est.observe(idx, label)
        value = float(est.estimate())
        if not np.isfinite(value):
            raise ValueError(f"estimate() returned {value}")
        _sendall(sock, _HDR.pack(b"R") + _I32.pack(seed) + _F64.pack(value))
        _recv_exact(sock, 1)
        os._exit(0)
    except BaseException:
        try:
            sys.stderr.write(f"[seed {seed}] candidate failed\n")
        except Exception:
            pass
        os._exit(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--submission-dir", required=True)
    ap.add_argument("--entry", required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--seeds", type=int, required=True)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--cell-seed", type=int, required=True)
    ap.add_argument("--per-seed-timeout", type=float, required=True)
    ap.add_argument("--sock-fd", type=int, required=True)
    args = ap.parse_args()

    sock = socket.socket(fileno=args.sock_fd)

    with open(args.entry, "rb") as fh:
        src = fh.read()
    code = compile(src, args.entry, "exec")

    npz = np.load(args.pool)
    target_probs = np.ascontiguousarray(npz["target_probs"], dtype=np.float64)
    surrogate_probs = np.ascontiguousarray(npz["surrogate_probs"], dtype=np.float64)
    n_pool, n_classes = target_probs.shape

    for seed in range(args.seed_start, args.seed_start + args.seeds):
        _sendall(sock, _HDR.pack(b"S") + _I32.pack(seed))
        if _recv_exact(sock, 1) != b"\x01":
            sys.stderr.write("scorer refused the seed handshake\n")
            return 1
        perm = np.random.default_rng([args.cell_seed, seed, 0x5EED]).permutation(n_pool)
        view = {
            "target_probs": np.ascontiguousarray(target_probs[perm]),
            "surrogate_probs": np.ascontiguousarray(surrogate_probs[perm]),
            "n_classes": int(n_classes),
            "n_pool": int(n_pool),
        }
        pool = {"_view": view, "_cseed": args.cell_seed}
        pid = os.fork()
        if pid == 0:
            _run_one_seed(sock, code, args.entry, args.submission_dir, pool,
                          args.budget, seed)
        deadline = time.monotonic() + args.per_seed_timeout
        status = None
        while True:
            done, status = os.waitpid(pid, os.WNOHANG)
            if done:
                break
            if time.monotonic() > deadline:
                for sig in (signal.SIGKILL,):
                    try:
                        os.killpg(pid, sig)
                    except OSError:
                        try:
                            os.kill(pid, sig)
                        except OSError:
                            pass
                os.waitpid(pid, 0)
                status = -1
                break
            time.sleep(0.0005)
        if status != 0:
            _sendall(sock, _HDR.pack(b"F") + _I32.pack(seed))
            _recv_exact(sock, 1)
    _sendall(sock, _HDR.pack(b"Q") + _I32.pack(0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
