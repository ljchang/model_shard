"""Integration test: TcpPeerRPC against a fake peer that echoes computed outputs.

Proves the RPC wire mechanism: ExpertRequest envelope out, out-of-band tensor
decoded by the fake server, per-expert outputs stacked on axis 2 and sent back
in an ExpertResponse. The orchestrator unstacks into {expert_id: tensor}.

No model weights are loaded — this test exercises only serialization + TCP
framing, so it runs in the fast suite.
"""

from __future__ import annotations

import socket
import threading
from typing import BinaryIO, cast

import mlx.core as mx

from model_shard._pb import wire_pb2
from model_shard.envelope import recv_envelope, send_envelope
from model_shard.expert_orchestrator import TcpPeerRPC
from model_shard.mlx_engine import _mx_to_wire_dtype, bytes_to_tensor, tensor_to_bytes


def _start_fake_peer(expert_ids: list[int]) -> int:
    """Spin up a one-shot TCP server that echoes h+eid for each requested expert.

    Returns the bound port. The server thread handles exactly one connection
    then exits.
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = int(server.getsockname()[1])

    def _run() -> None:
        try:
            conn, _ = server.accept()
            try:
                stream = conn.makefile("rwb")
                env, tensor = recv_envelope(cast(BinaryIO, stream))
                assert env.WhichOneof("payload") == "expert_request"
                req = env.expert_request
                h = bytes_to_tensor(
                    tensor,
                    shape=list(req.h_spec.shape),
                    dtype=req.h_spec.dtype,
                )
                # Deterministic echo: expert eid produces h + float(eid).
                stacked = mx.stack([h + float(eid) for eid in expert_ids], axis=2)
                mx.eval(stacked)

                resp = wire_pb2.Envelope()
                resp.expert_response.protocol_version = 1
                resp.expert_response.request_id = req.request_id
                resp.expert_response.layer_idx = req.layer_idx
                resp.expert_response.expert_ids.extend(expert_ids)
                raw = tensor_to_bytes(stacked)
                resp.expert_response.outputs_spec.shape.extend(list(stacked.shape))
                resp.expert_response.outputs_spec.dtype = _mx_to_wire_dtype(stacked.dtype)
                resp.expert_response.outputs_spec.quant = wire_pb2.QUANT_NONE
                resp.expert_response.outputs_spec.byte_count = len(raw)
                send_envelope(cast(BinaryIO, stream), resp, raw)
                stream.flush()
            finally:
                conn.close()
        finally:
            server.close()

    threading.Thread(target=_run, daemon=True).start()
    return port


def test_tcp_peer_rpc_roundtrip() -> None:
    ids = [3, 6]
    port = _start_fake_peer(ids)
    rpc = TcpPeerRPC(
        addresses={"peer": ("127.0.0.1", port)},
        timeout_s=5.0,
    )
    # Use mx.ones for determinism. float32 (default) is fine — the dtype
    # travels in h_spec and round-trips losslessly.
    h = mx.ones((1, 2, 4))
    out = rpc.call("peer", "r1", 15, ids, h)
    mx.eval(*out.values())

    assert set(out.keys()) == set(ids)
    for eid in ids:
        expected = h + float(eid)
        assert mx.allclose(out[eid], expected).item(), f"mismatch for eid={eid}"
        # Shape must be the post-attention hidden, not the stacked shape.
        assert out[eid].shape == h.shape


def test_orchestrator_close_is_idempotent() -> None:
    from unittest.mock import MagicMock

    from model_shard.backends import Backend
    from model_shard.expert_orchestrator import ExpertOrchestrator, PeerRPC

    class _Rpc(PeerRPC):
        def call(self, *a, **kw):  # type: ignore[no-untyped-def]
            raise AssertionError

    orch = ExpertOrchestrator(
        self_shard_id="s",
        owners={"s": set()},
        peer_rpc=_Rpc(),
        rpc_timeout_s=1.0,
        backend=MagicMock(spec=Backend),
    )
    orch.close()
    orch.close()  # must not raise


def test_tcp_peer_rpc_roundtrip_bf16_is_bit_exact() -> None:
    """bf16 is the production dtype for activations; round-trip must be exact."""
    ids = [1, 2, 5]
    port = _start_fake_peer(ids)
    rpc = TcpPeerRPC(
        addresses={"peer": ("127.0.0.1", port)},
        timeout_s=5.0,
    )
    h = mx.ones((1, 3, 8), dtype=mx.bfloat16)
    out = rpc.call("peer", "r2", 15, ids, h)
    mx.eval(*out.values())

    assert set(out.keys()) == set(ids)
    for eid in ids:
        expected = h + float(eid)
        assert out[eid].dtype == mx.bfloat16
        # bit-exact: float(eid) is a whole number and h is all ones, so
        # the add is exact in bf16.
        assert mx.array_equal(out[eid], expected).item(), f"bf16 mismatch for eid={eid}"
