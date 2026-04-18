"""Phase 6-C: _handle_expert_request must consult _live_experts (runtime),
not self._shard.moe_experts (bootstrap). This also fixes a latent 5b bug
where migration-attached experts would be rejected as "not hosted"."""
from __future__ import annotations

import io
import types
from unittest.mock import MagicMock

import mlx.core as mx
import pytest

from model_shard._pb import wire_pb2
from model_shard.mlx_engine import tensor_to_bytes
from model_shard.node import Node
from model_shard.shard_map import NodeAddress, ShardMap, ShardSpec


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("ENABLE_GOSSIP", "false")
    monkeypatch.setenv("ENABLE_PARTIAL_LOAD", "false")
    monkeypatch.setenv("ENABLE_DYNAMIC_MIGRATION", "false")
    monkeypatch.setenv("ENABLE_EXPERT_RETRY", "false")
    yield


def _mk_spec(sid: str, port: int, moe: dict[int, tuple[int, ...]]) -> ShardSpec:
    return ShardSpec(
        shard_id=sid,
        address=NodeAddress(host="127.0.0.1", port=port),
        start_layer=0, end_layer=30,
        moe_experts=moe,
    )


def _make_expert_request(layer_idx: int, expert_ids: list[int], h: mx.array):
    env = wire_pb2.Envelope()
    env.expert_request.protocol_version = 1
    env.expert_request.request_id = "r-test"
    env.expert_request.layer_idx = layer_idx
    env.expert_request.expert_ids.extend(expert_ids)
    env.expert_request.h_spec.shape.extend(list(h.shape))
    env.expert_request.h_spec.dtype = wire_pb2.DTYPE_BFLOAT16
    env.expert_request.h_spec.quant = wire_pb2.QUANT_NONE
    raw = tensor_to_bytes(h)
    env.expert_request.h_spec.byte_count = len(raw)
    return env, raw


def test_handle_expert_request_accepts_migrated_in_expert(monkeypatch):
    """A node that migration-attached expert 42 (not in bootstrap YAML)
    should accept inbound ExpertRequest for expert 42."""
    spec = _mk_spec("self", 31200, {15: (0, 3)})
    spec_peer = _mk_spec("peer", 31201, {15: (1, 4)})
    sm = ShardMap({"self": spec, "peer": spec_peer})
    lm = MagicMock()
    lm.held_ids_per_layer = {15: (0, 3, 42)}
    lm.text_model = types.SimpleNamespace(layers=[None] * 15 + [MagicMock()])
    n = Node(shard=spec, shard_map=sm, loaded_model=lm, total_layers=30)
    # Simulate migration attach (skip the actual attach_expert, just update registry).
    n._live_experts[15].add(42)
    # Mock moe.run_selected_experts so the handler doesn't need real MLX compute.
    from model_shard import moe as moe_mod
    def _fake_run_selected_experts(lm_, h_, layer_idx_, expert_ids_):
        return {eid: mx.zeros((1, 1, 8), dtype=mx.bfloat16) for eid in expert_ids_}
    monkeypatch.setattr(moe_mod, "run_selected_experts", _fake_run_selected_experts)
    # Build a BytesIO stream to capture any error.
    stream = io.BytesIO()
    env, raw = _make_expert_request(15, [42], mx.zeros((1, 1, 8), dtype=mx.bfloat16))
    n._handle_expert_request(env.expert_request, raw, stream)
    # Parse outbound: must be an ExpertResponse, NOT Error{ERR_WRONG_SHARD}.
    stream.seek(0)
    from model_shard.envelope import recv_envelope
    env_out, _ = recv_envelope(stream)
    assert env_out.WhichOneof("payload") == "expert_response", (
        f"expected expert_response, got {env_out.WhichOneof('payload')}"
    )


def test_handle_expert_request_rejects_evicted_expert():
    """A node that evicted expert E (not in _live_experts anymore) should
    return ERR_WRONG_SHARD for inbound ExpertRequest for E."""
    spec = _mk_spec("self", 31202, {15: (0, 3)})
    spec_peer = _mk_spec("peer", 31203, {15: (1, 4)})
    sm = ShardMap({"self": spec, "peer": spec_peer})
    lm = MagicMock()
    lm.held_ids_per_layer = {15: (0, 3)}
    lm.text_model = types.SimpleNamespace(layers=[None] * 15 + [MagicMock()])
    n = Node(shard=spec, shard_map=sm, loaded_model=lm, total_layers=30)
    # _live_experts does NOT include expert 42 → simulates post-eviction state.
    assert 42 not in n._live_experts.get(15, set())
    stream = io.BytesIO()
    env, raw = _make_expert_request(15, [42], mx.zeros((1, 1, 8), dtype=mx.bfloat16))
    n._handle_expert_request(env.expert_request, raw, stream)
    stream.seek(0)
    from model_shard.envelope import recv_envelope
    env_out, _ = recv_envelope(stream)
    assert env_out.WhichOneof("payload") == "error"
    assert env_out.error.code == wire_pb2.ERR_WRONG_SHARD
