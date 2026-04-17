"""Tests for Node._live_experts runtime ownership registry."""
from __future__ import annotations

from unittest.mock import MagicMock

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
