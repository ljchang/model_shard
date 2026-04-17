"""Observer-triggered queue poison unblocks _drive_decode_loop."""
from __future__ import annotations

import io
from unittest.mock import MagicMock

import pytest

from model_shard.membership.records import MemberRecord, MemberState, StateTransition
from model_shard.node import _POISON_TOKEN, Node, _HeadRequestState
from model_shard.shard_map import NodeAddress, ShardMap, ShardSpec


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("ENABLE_GOSSIP", "false")
    monkeypatch.setenv("ENABLE_PARTIAL_LOAD", "false")
    monkeypatch.setenv("ENABLE_DYNAMIC_MIGRATION", "false")
    yield


def _make_node() -> Node:
    spec_head = ShardSpec(
        shard_id="head",
        address=NodeAddress(host="127.0.0.1", port=30200),
        start_layer=0, end_layer=10, moe_experts={},
    )
    spec_tail = ShardSpec(
        shard_id="tail",
        address=NodeAddress(host="127.0.0.1", port=30201),
        start_layer=10, end_layer=30, moe_experts={},
    )
    sm = ShardMap({"head": spec_head, "tail": spec_tail})
    return Node(
        shard=spec_head, shard_map=sm, loaded_model=MagicMock(), total_layers=30
    )


def test_observer_poisons_active_head_states():
    n = _make_node()
    state = _HeadRequestState(
        client_stream=io.BytesIO(), max_new_tokens=10,
    )
    n._head_states["r1"] = state
    # Simulate peer-left-ALIVE transition.
    rec = MemberRecord(
        shard_id="tail", host="127.0.0.1", udp_port=31201,
        state=MemberState.SUSPECT, incarnation=1,
        last_state_change=0.0, suspect_deadline=None,
    )
    transition = StateTransition(
        shard_id="tail", old_state=MemberState.ALIVE, new_record=rec
    )
    n._on_membership_change(transition)
    assert state.token_queue.get_nowait() == _POISON_TOKEN


def test_drive_decode_loop_raises_on_poison():
    n = _make_node()
    state = _HeadRequestState(
        client_stream=io.BytesIO(), max_new_tokens=10,
    )
    state.token_queue.put(_POISON_TOKEN)
    n._head_states["r1"] = state
    n._kv_caches["r1"] = []
    n._drive_decode_loop("r1", state)
    # After poison handling, the request should be cleaned up.
    assert "r1" not in n._head_states
