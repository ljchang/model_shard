"""Gate tests for ENABLE_DYNAMIC_MIGRATION + partial-load dependency."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from model_shard.node import Node
from model_shard.shard_map import NodeAddress, ShardMap, ShardSpec


def _mk_spec() -> ShardSpec:
    return ShardSpec(
        shard_id="self",
        address=NodeAddress(host="127.0.0.1", port=30300),
        start_layer=0, end_layer=30,
        moe_experts={15: (0, 3, 6, 9)},
    )


def _mk_peer() -> ShardSpec:
    return ShardSpec(
        shard_id="peer",
        address=NodeAddress(host="127.0.0.1", port=30301),
        start_layer=0, end_layer=30,
        moe_experts={15: (1, 4, 7, 10)},
    )


def test_migration_on_partial_off_raises(monkeypatch):
    monkeypatch.setenv("ENABLE_GOSSIP", "false")
    monkeypatch.setenv("ENABLE_PARTIAL_LOAD", "false")
    monkeypatch.setenv("ENABLE_DYNAMIC_MIGRATION", "true")
    spec = _mk_spec()
    peer = _mk_peer()
    sm = ShardMap({"self": spec, "peer": peer})
    with pytest.raises(ValueError, match="ENABLE_PARTIAL_LOAD"):
        Node(shard=spec, shard_map=sm, loaded_model=MagicMock(), total_layers=30)


def test_migration_off_partial_off_ok(monkeypatch):
    monkeypatch.setenv("ENABLE_GOSSIP", "false")
    monkeypatch.setenv("ENABLE_PARTIAL_LOAD", "false")
    monkeypatch.setenv("ENABLE_DYNAMIC_MIGRATION", "false")
    spec = _mk_spec()
    peer = _mk_peer()
    sm = ShardMap({"self": spec, "peer": peer})
    n = Node(shard=spec, shard_map=sm, loaded_model=MagicMock(), total_layers=30)
    assert n._scanner is None
