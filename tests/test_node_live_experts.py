"""Tests for Node._live_experts runtime ownership registry."""
from __future__ import annotations

import types
from unittest.mock import MagicMock

import mlx.core as mx
import pytest

from model_shard.node import Node
from model_shard.shard_map import NodeAddress, ShardMap, ShardSpec


@pytest.fixture(autouse=True)
def _no_gossip_env(monkeypatch):
    monkeypatch.setenv("ENABLE_GOSSIP", "false")
    monkeypatch.setenv("ENABLE_PARTIAL_LOAD", "false")
    monkeypatch.setenv("ENABLE_DYNAMIC_MIGRATION", "false")
    yield


def _mk_spec(sid: str, port: int, moe: dict[int, tuple[int, ...]]) -> ShardSpec:
    return ShardSpec(
        shard_id=sid,
        address=NodeAddress(host="127.0.0.1", port=port),
        start_layer=0,
        end_layer=30,
        moe_experts=moe,
    )


def test_live_experts_seeded_from_shard_spec():
    spec_a = _mk_spec("A", 30100, {15: (0, 3, 6, 9)})
    spec_b = _mk_spec("B", 30199, {15: (1, 4, 7, 10)})
    sm = ShardMap({"A": spec_a, "B": spec_b})
    n = Node(shard=spec_a, shard_map=sm, loaded_model=MagicMock(), total_layers=30)
    assert n._live_experts == {15: {0, 3, 6, 9}}


def test_ownership_seen_seeded_with_every_bootstrap_shard():
    spec_a = _mk_spec("A", 30101, {15: (0, 3, 6, 9)})
    spec_b = _mk_spec("B", 30102, {15: (1, 4, 7, 10)})
    sm = ShardMap({"A": spec_a, "B": spec_b})
    n = Node(shard=spec_a, shard_map=sm, loaded_model=MagicMock(), total_layers=30)
    assert ("A", 15, 0) in n._ownership_seen
    assert ("B", 15, 1) in n._ownership_seen
    assert ("B", 15, 10) in n._ownership_seen


def test_owners_of_resolves_union():
    spec_a = _mk_spec("A", 30103, {15: (0, 3)})
    spec_b = _mk_spec("B", 30104, {15: (3, 7)})  # B also owns 3 (overlap)
    sm = ShardMap({"A": spec_a, "B": spec_b})
    n = Node(shard=spec_a, shard_map=sm, loaded_model=MagicMock(), total_layers=30)
    assert n.owners_of(15, 3) == {"A", "B"}
    assert n.owners_of(15, 7) == {"B"}
    assert n.owners_of(15, 99) == set()


def test_attach_path_updates_live_experts_and_announces(monkeypatch):
    monkeypatch.setenv("ENABLE_GOSSIP", "false")
    monkeypatch.setenv("ENABLE_PARTIAL_LOAD", "false")
    monkeypatch.setenv("ENABLE_DYNAMIC_MIGRATION", "false")
    spec = _mk_spec("self", 30150, {15: (0, 3)})
    spec_peer = _mk_spec("peer", 30151, {15: (1, 4)})
    sm = ShardMap({"self": spec, "peer": spec_peer})
    lm = MagicMock()
    lm.held_ids_per_layer = {15: (0, 3)}
    # Synthesize a mutable switch_glu so attach_expert succeeds.
    def _stack(n: int, cols: int) -> mx.array:
        return mx.zeros((n, 4, cols))
    projs = {
        name: types.SimpleNamespace(
            weight=_stack(2, 4), scales=_stack(2, 8), biases=_stack(2, 8),
        )
        for name in ("gate_proj", "up_proj", "down_proj")
    }
    layer = types.SimpleNamespace(
        experts=types.SimpleNamespace(switch_glu=types.SimpleNamespace(**projs))
    )
    lm.text_model = types.SimpleNamespace(layers=[None] * 15 + [layer])
    n = Node(shard=spec, shard_map=sm, loaded_model=lm, total_layers=30)

    # 9 tensors in _PROJ_ATTR_ORDER: (gate_proj, up_proj, down_proj) x
    # (weight=(4,4), scales=(4,8), biases=(4,8)) — shapes must match the
    # current stacked attr (n, 4, cols) with the leading expert-count dim removed.
    new_tensors = [
        mx.zeros((4, 4)), mx.zeros((4, 8)), mx.zeros((4, 8)),  # gate_proj
        mx.zeros((4, 4)), mx.zeros((4, 8)), mx.zeros((4, 8)),  # up_proj
        mx.zeros((4, 4)), mx.zeros((4, 8)), mx.zeros((4, 8)),  # down_proj
    ]
    n.migration_attach(layer_idx=15, expert_id=7, tensors=new_tensors)
    assert 7 in n._live_experts[15]
    assert ("self", 15, 7) in n._ownership_seen


def test_node_ownership_view_supports_remove():
    """Phase 6-C: Node._ownership_view is a versioned dict; REMOVE supersedes ADD
    by ts_unix_ms."""
    import os
    monkeypatch_env = {"ENABLE_GOSSIP": "false", "ENABLE_PARTIAL_LOAD": "false",
                        "ENABLE_DYNAMIC_MIGRATION": "false"}
    for k, v in monkeypatch_env.items():
        os.environ[k] = v
    try:
        spec_a = _mk_spec("A", 31000, {15: (0, 3)})
        spec_b = _mk_spec("B", 31001, {15: (3, 7)})
        sm = ShardMap({"A": spec_a, "B": spec_b})
        from unittest.mock import MagicMock
        n = Node(shard=spec_a, shard_map=sm, loaded_model=MagicMock(), total_layers=30)
        # Initial: B owns 3 (bootstrap).
        assert "B" in n.owners_of(15, 3)
        # Simulate B's REMOVE at a later timestamp.
        n._ownership_view_put("B", 15, 3, action=1, ts_unix_ms=9_999_999_999_999)
        assert "B" not in n.owners_of(15, 3)
    finally:
        for k in monkeypatch_env:
            os.environ.pop(k, None)


def test_node_ownership_view_put_last_writer_wins():
    """Older ts_unix_ms must not supersede a newer entry."""
    import os
    monkeypatch_env = {"ENABLE_GOSSIP": "false", "ENABLE_PARTIAL_LOAD": "false",
                        "ENABLE_DYNAMIC_MIGRATION": "false"}
    for k, v in monkeypatch_env.items():
        os.environ[k] = v
    try:
        spec_a = _mk_spec("A", 31002, {15: (0, 3)})
        spec_b = _mk_spec("B", 31003, {15: (1, 4)})
        sm = ShardMap({"A": spec_a, "B": spec_b})
        from unittest.mock import MagicMock
        n = Node(shard=spec_a, shard_map=sm, loaded_model=MagicMock(), total_layers=30)
        # Set REMOVE at t=2000.
        n._ownership_view_put("A", 15, 99, action=1, ts_unix_ms=2000)
        assert "A" not in n.owners_of(15, 99)
        # Stale ADD at t=1000 must be dropped.
        n._ownership_view_put("A", 15, 99, action=0, ts_unix_ms=1000)
        assert "A" not in n.owners_of(15, 99)
        # Newer ADD at t=3000 supersedes.
        n._ownership_view_put("A", 15, 99, action=0, ts_unix_ms=3000)
        assert "A" in n.owners_of(15, 99)
    finally:
        for k in monkeypatch_env:
            os.environ.pop(k, None)
