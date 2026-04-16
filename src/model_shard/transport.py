"""Wire framing for node-to-node messages.

Frame format:
    [uint32 BE msg_len] [msg_len bytes] [uint32 BE tensor_len] [tensor_len bytes]

Both length prefixes are always present. `tensor_len` may be zero. A truncated
stream at any read position raises EOFError; callers treat that as connection
closure.
"""

import struct
from typing import BinaryIO

_LEN_FMT = "!I"
_LEN_SIZE = 4


def _read_exact(stream: BinaryIO, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise EOFError(f"expected {n} bytes, got {len(buf)} before EOF")
        buf.extend(chunk)
    return bytes(buf)


def write_frame(stream: BinaryIO, msg_bytes: bytes, tensor_bytes: bytes = b"") -> None:
    stream.write(struct.pack(_LEN_FMT, len(msg_bytes)))
    if msg_bytes:
        stream.write(msg_bytes)
    stream.write(struct.pack(_LEN_FMT, len(tensor_bytes)))
    if tensor_bytes:
        stream.write(tensor_bytes)


def read_frame(stream: BinaryIO) -> tuple[bytes, bytes]:
    (msg_len,) = struct.unpack(_LEN_FMT, _read_exact(stream, _LEN_SIZE))
    msg = _read_exact(stream, msg_len) if msg_len else b""
    (tensor_len,) = struct.unpack(_LEN_FMT, _read_exact(stream, _LEN_SIZE))
    tensor = _read_exact(stream, tensor_len) if tensor_len else b""
    return msg, tensor
