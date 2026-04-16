"""Tests for envelope helpers (protobuf serialization + framing over a stream)."""

import io

import pytest

from model_shard._pb import wire_pb2
from model_shard.envelope import recv_envelope, send_envelope


def _begin(request_id: str, tokens: list[int]) -> wire_pb2.Envelope:
    env = wire_pb2.Envelope()
    env.begin.protocol_version = 1
    env.begin.request_id = request_id
    env.begin.sequence_id = "seq-0"
    env.begin.prompt_token_ids.extend(tokens)
    env.begin.sampling.greedy = True
    env.begin.start_layer = 0
    return env


def _activation(request_id: str, next_layer: int, nbytes: int) -> wire_pb2.Envelope:
    env = wire_pb2.Envelope()
    env.activation.protocol_version = 1
    env.activation.request_id = request_id
    env.activation.next_layer_idx = next_layer
    env.activation.tensor.shape.extend([1, 5, 2816])
    env.activation.tensor.dtype = wire_pb2.DTYPE_BFLOAT16
    env.activation.tensor.quant = wire_pb2.QUANT_NONE
    env.activation.tensor.byte_count = nbytes
    return env


def test_send_recv_begin_request_roundtrip() -> None:
    buf = io.BytesIO()
    send_envelope(buf, _begin("r1", [7, 8, 9]))
    buf.seek(0)
    env, tensor = recv_envelope(buf)
    assert env.WhichOneof("payload") == "begin"
    assert env.begin.request_id == "r1"
    assert list(env.begin.prompt_token_ids) == [7, 8, 9]
    assert tensor == b""


def test_send_recv_activation_carries_tensor_bytes() -> None:
    payload = b"\x01\x02\x03\x04" * 64  # 256 bytes
    buf = io.BytesIO()
    send_envelope(buf, _activation("r1", next_layer=10, nbytes=len(payload)), tensor_bytes=payload)
    buf.seek(0)
    env, tensor = recv_envelope(buf)
    assert env.WhichOneof("payload") == "activation"
    assert env.activation.next_layer_idx == 10
    assert env.activation.tensor.byte_count == len(payload)
    assert tensor == payload


def test_multiple_envelopes_in_stream() -> None:
    buf = io.BytesIO()
    send_envelope(buf, _begin("r1", [1]))
    send_envelope(buf, _activation("r1", 10, 8), tensor_bytes=b"\x00" * 8)
    buf.seek(0)
    env1, _t1 = recv_envelope(buf)
    env2, t2 = recv_envelope(buf)
    assert env1.WhichOneof("payload") == "begin"
    assert env2.WhichOneof("payload") == "activation"
    assert t2 == b"\x00" * 8


def test_recv_empty_stream_raises_eof() -> None:
    buf = io.BytesIO(b"")
    with pytest.raises(EOFError):
        recv_envelope(buf)
