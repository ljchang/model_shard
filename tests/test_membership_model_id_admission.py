"""Phase 7-C-3b Task 4: cluster admission contract — model_id validation
in MembershipState.

Reject peers with mismatched model_id; accept matching; reject empty
peer model_id when local has set one. Coverage spans all three admission
sites: _handle_join, _handle_delta (bulk install), _maybe_apply_peer_delta.
"""
from __future__ import annotations

import random

from model_shard.membership.config import SwimConfig
from model_shard.membership.records import (
    JoinMsg,
    MemberRecord,
    MembershipDeltaMsg,
    MemberState,
)
from model_shard.membership.state import MembershipState, PeerSpec


def _make_state(local_model_id: str = "") -> MembershipState:
    self_spec = PeerSpec(shard_id="self", host="127.0.0.1", udp_port=10001)
    peer_specs = [PeerSpec(shard_id="peer", host="127.0.0.1", udp_port=10002)]
    return MembershipState(
        self_spec=self_spec,
        peer_specs=peer_specs,
        rng=random.Random(0),
        config=SwimConfig(),
        local_model_id=local_model_id,
    )


def _peer_record(model_id: str = "", incarnation: int = 1) -> MemberRecord:
    return MemberRecord(
        shard_id="peer", host="127.0.0.1", udp_port=10002,
        state=MemberState.ALIVE, incarnation=incarnation,
        model_id=model_id,
        last_state_change=0.0, suspect_deadline=None,
    )


def _new_peer_record(
    shard_id: str, model_id: str = "", incarnation: int = 1,
) -> MemberRecord:
    """A record for a peer NOT in the initial membership view."""
    return MemberRecord(
        shard_id=shard_id, host="127.0.0.1", udp_port=10999,
        state=MemberState.ALIVE, incarnation=incarnation,
        model_id=model_id,
        last_state_change=0.0, suspect_deadline=None,
    )


def test_admission_accepts_matching_model_id_via_peer_delta():
    """A gossip update from peer with matching model_id is applied."""
    state = _make_state(local_model_id="google/gemma-4-26B-A4B-it")
    rec = _peer_record(model_id="google/gemma-4-26B-A4B-it", incarnation=2)
    state._maybe_apply_peer_delta(rec, now=1.0)
    view = state.view()
    assert view["peer"].incarnation == 2


def test_admission_rejects_mismatched_model_id_via_peer_delta():
    """A gossip update from a peer with a different model_id is silently dropped."""
    state = _make_state(local_model_id="google/gemma-4-26B-A4B-it")
    rec = _peer_record(
        model_id="mlx-community/gemma-4-26b-a4b-it-4bit", incarnation=2,
    )
    state._maybe_apply_peer_delta(rec, now=1.0)
    view = state.view()
    # Original incarnation 0 unchanged.
    assert view["peer"].incarnation == 0


def test_admission_rejects_empty_peer_when_local_set():
    """A new node with model_id='X' rejects a peer reporting model_id=''.
    This is intentional: once the cluster is on the new contract, legacy
    peers can't silently join."""
    state = _make_state(local_model_id="google/gemma-4-26B-A4B-it")
    rec = _peer_record(model_id="", incarnation=2)
    state._maybe_apply_peer_delta(rec, now=1.0)
    view = state.view()
    assert view["peer"].incarnation == 0


def test_admission_accepts_when_local_empty():
    """Backwards compat: if local has no model_id set, admission is permissive."""
    state = _make_state(local_model_id="")
    rec = _peer_record(model_id="any-model", incarnation=2)
    state._maybe_apply_peer_delta(rec, now=1.0)
    view = state.view()
    assert view["peer"].incarnation == 2


def test_admission_rejects_join_with_mismatched_model_id():
    """A JoinMsg from a newcomer with mismatched model_id is dropped:
    no response sent, no record installed."""
    state = _make_state(local_model_id="google/gemma-4-26B-A4B-it")
    newcomer = _new_peer_record(
        "newcomer", model_id="other-model", incarnation=0,
    )
    out = state.recv(JoinMsg(self_record=newcomer), now=1.0)
    assert out == []
    assert "newcomer" not in state.view()


def test_admission_accepts_join_with_matching_model_id():
    """A JoinMsg from a newcomer with matching model_id installs the
    newcomer and echoes back a MembershipDeltaMsg."""
    state = _make_state(local_model_id="google/gemma-4-26B-A4B-it")
    newcomer = _new_peer_record(
        "newcomer", model_id="google/gemma-4-26B-A4B-it", incarnation=0,
    )
    out = state.recv(JoinMsg(self_record=newcomer), now=1.0)
    assert len(out) == 1
    assert "newcomer" in state.view()


def test_admission_rejects_delta_bulk_install_with_mismatched_model_id():
    """A MembershipDeltaMsg containing a new peer with mismatched
    model_id is dropped at the bulk install path (not just at
    _maybe_apply_peer_delta)."""
    state = _make_state(local_model_id="google/gemma-4-26B-A4B-it")
    new_peer = _new_peer_record(
        "newcomer-from-delta", model_id="other-model", incarnation=1,
    )
    state.recv(MembershipDeltaMsg(members=[new_peer]), now=1.0)
    assert "newcomer-from-delta" not in state.view()
