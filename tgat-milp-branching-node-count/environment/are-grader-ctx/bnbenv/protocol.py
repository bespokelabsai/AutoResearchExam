from __future__ import annotations

import pickle
import struct

_HEADER = struct.Struct("<I")
MAX_FRAME = 1 << 28


def write_obs(stream, payload):
    blob = pickle.dumps(payload, protocol=5)
    stream.write(_HEADER.pack(len(blob)))
    stream.write(blob)
    stream.flush()


def read_obs(stream):
    head = _read_exactly(stream, _HEADER.size)
    if head is None:
        return None
    (size,) = _HEADER.unpack(head)
    if size > MAX_FRAME:
        raise ValueError("frame too large")
    body = _read_exactly(stream, size)
    if body is None:
        return None
    return pickle.loads(body)


def write_decision(stream, text):
    data = (text + "\n").encode("ascii", "replace")
    stream.write(data)
    stream.flush()


def _read_exactly(stream, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
