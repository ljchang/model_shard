"""Unit tests for the node.py / membership integration. Do NOT load the model
— these tests use a stub LoadedModel to keep the suite fast."""

from __future__ import annotations

import io
from unittest.mock import MagicMock

import pytest

from model_shard._pb import wire_pb2
from model_shard.envelope import recv_envelope
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


def test_admission_rejects_when_a_peer_is_dead(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_GOSSIP", "true")
    from model_shard.membership.records import MemberRecord, MemberState
    from model_shard.node import Node

    sm = _make_shardmap()
    n = Node(
        shard=sm.lookup("head"),
        shard_map=sm,
        loaded_model=MagicMock(),
        total_layers=30,
    )
    # Force mid into DEAD state in the runner's view.
    assert n.membership is not None
    members = n.membership.state._members
    members["mid"] = MemberRecord(
        "mid", "127.0.0.1", 20002, MemberState.DEAD, 1, 0.0, None
    )

    # Build an in-memory client stream and a BeginRequest.
    buf = io.BytesIO()
    req = wire_pb2.BeginRequest(
        protocol_version=1,
        request_id="req-1",
        sequence_id="seq-1",
        prompt_token_ids=[1, 2, 3],
        sampling=wire_pb2.SamplingParams(greedy=True),
        start_layer=0,
        max_new_tokens=4,
    )
    n._handle_begin(req, buf)
    buf.seek(0)
    env, _ = recv_envelope(buf)
    assert env.WhichOneof("payload") == "error"
    assert env.error.code == wire_pb2.ERR_SHARD_UNAVAILABLE
    assert "mid" in env.error.detail
    n.shutdown()


def test_admission_passes_when_all_peers_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_GOSSIP", "true")
    from model_shard.node import Node

    sm = _make_shardmap()
    n = Node(
        shard=sm.lookup("head"),
        shard_map=sm,
        loaded_model=MagicMock(),
        total_layers=30,
    )
    # All peers are alive in the initial view, so admission passes.
    # We bail out before MLX work runs by raising in the mock.
    n._lm.language_model.make_cache = MagicMock(side_effect=RuntimeError("mlx not real"))
    buf = io.BytesIO()
    req = wire_pb2.BeginRequest(
        protocol_version=1,
        request_id="req-2",
        sequence_id="seq-2",
        prompt_token_ids=[1, 2, 3],
        sampling=wire_pb2.SamplingParams(greedy=True),
        start_layer=0,
        max_new_tokens=4,
    )
    with pytest.raises(RuntimeError, match="mlx not real"):
        n._handle_begin(req, buf)
    n.shutdown()
