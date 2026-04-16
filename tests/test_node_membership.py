"""Unit tests for the node.py / membership integration. Do NOT load the model
— these tests use a stub LoadedModel to keep the suite fast."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from model_shard.shard_map import NodeAddress, ShardMap, ShardSpec


def _make_shardmap() -> ShardMap:
    return ShardMap(
        {
            "head": ShardSpec("head", NodeAddress("127.0.0.1", 19001), 0, 10),
            "mid": ShardSpec("mid", NodeAddress("127.0.0.1", 19002), 10, 20),
            "tail": ShardSpec("tail", NodeAddress("127.0.0.1", 19003), 20, 30),
        }
    )


def test_node_constructs_membership_runner_when_gossip_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_GOSSIP", "true")
    from model_shard.node import Node

    sm = _make_shardmap()
    n = Node(
        shard=sm.lookup("head"),
        shard_map=sm,
        loaded_model=MagicMock(),
        total_layers=30,
    )
    assert n.membership is not None
    n.shutdown()


def test_node_does_not_construct_runner_when_gossip_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_GOSSIP", "false")
    from model_shard.node import Node

    sm = _make_shardmap()
    n = Node(
        shard=sm.lookup("head"),
        shard_map=sm,
        loaded_model=MagicMock(),
        total_layers=30,
    )
    assert n.membership is None
    n.shutdown()
