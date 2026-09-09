import os
import struct
import sys

_HDR = struct.Struct("<4sQ")


def _pin():
    try:
        avail = sorted(os.sched_getaffinity(0))
        n = int(os.environ.get("EXEC_N_CPUS", "8"))
        if len(avail) > n:
            os.sched_setaffinity(0, set(avail[:n]))
    except (AttributeError, OSError):
        pass


def _read_exactly(fd, n):
    chunks = []
    got = 0
    while got < n:
        b = os.read(fd, min(1 << 20, n - got))
        if not b:
            raise EOFError("parent closed the pipe")
        chunks.append(b)
        got += len(b)
    return b"".join(chunks)


def _recv(fd):
    cmd, ln = _HDR.unpack(_read_exactly(fd, _HDR.size))
    return cmd.rstrip(b"\0"), _read_exactly(fd, ln)


def _send(fd, cmd, payload=b""):
    os.write(fd, _HDR.pack(cmd.ljust(4, b"\0"), len(payload)))
    off = 0
    while off < len(payload):
        off += os.write(fd, payload[off:off + (1 << 20)])


def main():
    _pin()
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    import numpy as np
    import graph_spec

    which = sys.argv[1]
    if which == "sub":
        sys.path.insert(0, os.path.join(here, "submission"))
        import executor as impl
    else:
        import reference_executor as impl

    n_workers = int(os.environ.get("EXEC_N_WORKERS", "8"))
    rfd, wfd = 0, 1
    spec = None
    fn = None

    _send(wfd, b"HI")
    while True:
        cmd, payload = _recv(rfd)
        if cmd == b"EXIT":
            return 0
        if cmd == b"PING":
            _send(wfd, b"PONG")
        elif cmd == b"BLD":
            fn = None
            spec = graph_spec.sample_spec(int(payload))
            weights = graph_spec.build_weights(spec)
            fn = impl.build_executor(spec, weights, n_workers)
            if not callable(fn):
                raise TypeError("build_executor did not return a callable")
            _send(wfd, b"RDY")
        elif cmd == b"CALL":
            x = np.frombuffer(payload, dtype=np.float32).reshape(spec["T"], spec["d_model"]).copy()
            y = np.asarray(fn(x))
            head = ("%d,%s,%s" % (y.ndim, str(y.dtype), "x".join(str(v) for v in y.shape)))
            _send(wfd, b"OUT", head.encode() + b"|" + np.ascontiguousarray(y).tobytes())
        else:
            raise ValueError("unknown command %r" % cmd)


if __name__ == "__main__":
    sys.exit(main())
