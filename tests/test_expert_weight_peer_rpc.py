"""Unit test for ExpertWeightPeerRPC against an in-process TCP server."""
from __future__ import annotations

import socket
import threading

import mlx.core as mx
import pytest

from model_shard._pb import wire_pb2
from model_shard.envelope import recv_envelope, send_envelope
from model_shard.migration import ExpertWeightPeerRPC
from model_shard.mlx_engine import _mx_to_wire_dtype, tensor_to_bytes


def _fake_server(host: str, port: int, tensors: list[mx.array]) -> threading.Thread:
    def run():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen(1)
        conn, _ = s.accept()
        stream = conn.makefile("rwb", buffering=0)
        env, _ = recv_envelope(stream)
        assert env.WhichOneof("payload") == "expert_weight_request"
        req = env.expert_weight_request
        resp = wire_pb2.Envelope()
        resp.expert_weight_transfer.protocol_version = 1
        resp.expert_weight_transfer.request_id = req.request_id
        resp.expert_weight_transfer.layer_idx = req.layer_idx
        resp.expert_weight_transfer.expert_id = req.expert_id
        resp.expert_weight_transfer.tensor_count = len(tensors)
        blobs: list[bytes] = []
        for t in tensors:
            d = resp.expert_weight_transfer.tensors.add()
            d.shape.extend(list(t.shape))
            d.dtype = _mx_to_wire_dtype(t.dtype)
            d.quant = wire_pb2.QUANT_NONE
            raw = tensor_to_bytes(t)
            d.byte_count = len(raw)
            blobs.append(raw)
        send_envelope(stream, resp, b"".join(blobs))
        conn.close()
        s.close()
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_pull_deserialises_nine_tensors():
    tensors = [mx.full((4, 4), float(i), dtype=mx.bfloat16) for i in range(9)]
    port = _free_port()
    server = _fake_server("127.0.0.1", port, tensors)
    rpc = ExpertWeightPeerRPC(
        addresses={"peer": ("127.0.0.1", port)}, timeout_s=5.0
    )
    received = rpc.pull(source_shard_id="peer", layer_idx=15, expert_id=7)
    server.join(timeout=2.0)
    assert len(received) == 9
    for got, want in zip(received, tensors, strict=True):
        assert mx.array_equal(got, want).item()


def test_pull_raises_on_error_envelope():
    port = _free_port()
    def run():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", port))
        s.listen(1)
        conn, _ = s.accept()
        stream = conn.makefile("rwb", buffering=0)
        recv_envelope(stream)
        err = wire_pb2.Envelope()
        err.error.protocol_version = 1
        err.error.request_id = "r"
        err.error.code = wire_pb2.ERR_SHARD_UNAVAILABLE
        err.error.detail = "gone"
        send_envelope(stream, err)
        conn.close()
        s.close()
    t = threading.Thread(target=run, daemon=True)
    t.start()
    rpc = ExpertWeightPeerRPC(
        addresses={"peer": ("127.0.0.1", port)}, timeout_s=5.0
    )
    with pytest.raises(RuntimeError, match="gone"):
        rpc.pull(source_shard_id="peer", layer_idx=15, expert_id=7)
    t.join(timeout=2.0)
