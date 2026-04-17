"""Roundtrip tests for Phase 3 ExpertRequest / ExpertResponse envelopes."""

from __future__ import annotations

import io

import numpy as np

from model_shard._pb import wire_pb2
from model_shard.envelope import recv_envelope, send_envelope


def _roundtrip(env: wire_pb2.Envelope, tensor: bytes) -> tuple[wire_pb2.Envelope, bytes]:
    buf = io.BytesIO()
    send_envelope(buf, env, tensor)
    buf.seek(0)
    out_env, out_tensor = recv_envelope(buf)
    return out_env, out_tensor


def test_expert_request_roundtrip() -> None:
    env = wire_pb2.Envelope()
    env.expert_request.protocol_version = 1
    env.expert_request.request_id = "req-abc"
    env.expert_request.layer_idx = 15
    env.expert_request.expert_ids.extend([3, 6, 126])
    env.expert_request.h_spec.shape.extend([1, 7, 2816])
    env.expert_request.h_spec.dtype = wire_pb2.DTYPE_BFLOAT16
    env.expert_request.h_spec.byte_count = 1 * 7 * 2816 * 2

    tensor = np.zeros((1, 7, 2816), dtype=np.uint16).tobytes()
    got_env, got_tensor = _roundtrip(env, tensor)

    assert got_env.WhichOneof("payload") == "expert_request"
    assert got_env.expert_request.layer_idx == 15
    assert list(got_env.expert_request.expert_ids) == [3, 6, 126]
    assert got_tensor == tensor


def test_expert_response_roundtrip() -> None:
    env = wire_pb2.Envelope()
    env.expert_response.protocol_version = 1
    env.expert_response.request_id = "req-abc"
    env.expert_response.layer_idx = 15
    env.expert_response.expert_ids.extend([3, 6, 126])
    env.expert_response.outputs_spec.shape.extend([1, 7, 3, 2816])
    env.expert_response.outputs_spec.dtype = wire_pb2.DTYPE_BFLOAT16
    env.expert_response.outputs_spec.byte_count = 1 * 7 * 3 * 2816 * 2

    tensor = np.zeros((1, 7, 3, 2816), dtype=np.uint16).tobytes()
    got_env, got_tensor = _roundtrip(env, tensor)

    assert got_env.WhichOneof("payload") == "expert_response"
    assert got_env.expert_response.layer_idx == 15
    assert list(got_env.expert_response.expert_ids) == [3, 6, 126]
    assert got_tensor == tensor
