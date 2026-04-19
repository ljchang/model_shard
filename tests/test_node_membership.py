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
    n._backend.make_cache = MagicMock(side_effect=RuntimeError("mlx not real"))
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


def test_observer_closes_outbound_on_peer_going_suspect(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_GOSSIP", "true")
    from model_shard.membership.records import (
        MemberRecord,
        MemberState,
        StateTransition,
    )
    from model_shard.node import Node

    sm = _make_shardmap()
    n = Node(
        shard=sm.lookup("head"),
        shard_map=sm,
        loaded_model=MagicMock(),
        total_layers=30,
    )
    # Inject a fake outbound stream so we can assert it gets closed.
    closed = MagicMock()
    n._out_stream = MagicMock(close=closed)
    n._out_sock = MagicMock(close=MagicMock())

    new_rec = MemberRecord("mid", "127.0.0.1", 20002, MemberState.SUSPECT, 0, 0.0, 4.0)
    n._on_membership_change(
        StateTransition(shard_id="mid", old_state=MemberState.ALIVE, new_record=new_rec)
    )

    assert closed.called
    assert n._out_stream is None
    n.shutdown()


def test_decode_loop_emits_error_to_client_on_broken_pipe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_GOSSIP", "true")
    from model_shard.node import Node, _HeadRequestState

    sm = _make_shardmap()
    n = Node(
        shard=sm.lookup("head"),
        shard_map=sm,
        loaded_model=MagicMock(),
        total_layers=30,
    )

    # Set up a fake _drive_decode_loop scenario: a head state pointing at an
    # in-memory client stream. Simulate a broken pipe on _forward_activation.
    buf = io.BytesIO()
    state = _HeadRequestState(client_stream=buf, max_new_tokens=4)
    state.token_queue.put(123)  # one token to process

    monkeypatch.setattr(
        n,
        "_forward_activation",
        MagicMock(side_effect=BrokenPipeError("peer closed")),
    )
    monkeypatch.setattr(n, "_run_my_layers", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(
        n._backend, "embed", lambda *_a, **_k: MagicMock()
    )

    with n._state_lock:
        n._kv_caches["req-1"] = []
        n._head_states["req-1"] = state

    n._drive_decode_loop("req-1", state)

    buf.seek(0)
    # Skip the SampledToken envelope (token 123); read the next envelope (the error).
    _env1, _ = recv_envelope(buf)
    env2, _ = recv_envelope(buf)
    assert env2.WhichOneof("payload") == "error"
    assert env2.error.code == wire_pb2.ERR_SHARD_UNAVAILABLE
    n.shutdown()
