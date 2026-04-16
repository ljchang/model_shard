"""Wire framing round-trip tests.

The framing format is a core Phase 1 contract:

    [uint32 BE msg_len] [msg_len bytes] [uint32 BE tensor_len] [tensor_len bytes]

tensor_len may be 0 (no tensor payload). Both lengths are always present.
"""

import io

import pytest

from model_shard.transport import read_frame, write_frame


def test_roundtrip_no_tensor() -> None:
    buf = io.BytesIO()
    write_frame(buf, b"hello world")
    buf.seek(0)
    msg, tensor = read_frame(buf)
    assert msg == b"hello world"
    assert tensor == b""


def test_roundtrip_with_tensor() -> None:
    buf = io.BytesIO()
    write_frame(buf, b"header", tensor_bytes=b"\x00\x01\x02\x03")
    buf.seek(0)
    msg, tensor = read_frame(buf)
    assert msg == b"header"
    assert tensor == b"\x00\x01\x02\x03"


def test_empty_message_allowed() -> None:
    """A frame with zero-length msg and a non-empty tensor is legal wire."""
    buf = io.BytesIO()
    write_frame(buf, b"", tensor_bytes=b"tensor-only")
    buf.seek(0)
    msg, tensor = read_frame(buf)
    assert msg == b""
    assert tensor == b"tensor-only"


def test_multiple_frames_in_one_stream() -> None:
    """Consecutive frames on the same stream should decode independently."""
    buf = io.BytesIO()
    write_frame(buf, b"first", tensor_bytes=b"AAA")
    write_frame(buf, b"second")
    write_frame(buf, b"third", tensor_bytes=b"C")
    buf.seek(0)

    assert read_frame(buf) == (b"first", b"AAA")
    assert read_frame(buf) == (b"second", b"")
    assert read_frame(buf) == (b"third", b"C")


def test_large_tensor_payload() -> None:
    """Framing must survive multi-MB tensor payloads (realistic activation)."""
    big = b"\xab" * (4 * 1024 * 1024)  # 4 MB
    buf = io.BytesIO()
    write_frame(buf, b"desc", tensor_bytes=big)
    buf.seek(0)
    msg, tensor = read_frame(buf)
    assert msg == b"desc"
    assert tensor == big


def test_eof_on_empty_stream_raises() -> None:
    """Reading from a closed/empty stream must raise, not return garbage."""
    buf = io.BytesIO(b"")
    with pytest.raises(EOFError):
        read_frame(buf)


def test_truncated_length_prefix_raises() -> None:
    """Partial length prefix (fewer than 4 bytes) must raise EOFError."""
    buf = io.BytesIO(b"\x00\x00")
    with pytest.raises(EOFError):
        read_frame(buf)


def test_truncated_message_payload_raises() -> None:
    """Length prefix says N bytes but fewer are present: EOFError."""
    # Claims a 5-byte message, provides only 3.
    buf = io.BytesIO(b"\x00\x00\x00\x05hel")
    with pytest.raises(EOFError):
        read_frame(buf)


def test_truncated_tensor_payload_raises() -> None:
    """Tensor length says N but fewer bytes are present: EOFError."""
    # Valid msg_len=3, msg="abc", tensor_len=10, only 2 tensor bytes given.
    buf = io.BytesIO(b"\x00\x00\x00\x03abc\x00\x00\x00\x0aXY")
    with pytest.raises(EOFError):
        read_frame(buf)


def test_write_preserves_byte_order_big_endian() -> None:
    """The on-wire length prefix must be big-endian regardless of host endianness."""
    buf = io.BytesIO()
    # 260 in BE = 0x00 0x00 0x01 0x04
    payload = b"x" * 260
    write_frame(buf, payload)
    raw = buf.getvalue()
    assert raw[:4] == b"\x00\x00\x01\x04", f"got {raw[:4]!r}"
