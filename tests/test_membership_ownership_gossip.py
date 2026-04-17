"""Unit tests for ownership delta TTL'd piggyback and ownership_view union."""
from __future__ import annotations

from model_shard.membership.config import SwimConfig
from model_shard.membership.records import (
    OwnershipDeltaRecord,
    PingMsg,
)
from model_shard.membership.runner import MembershipRunner
from model_shard.membership.state import PeerSpec


def _make_runner(self_port: int = 42001, peer_port: int = 42002) -> MembershipRunner:
    return MembershipRunner(
        self_spec=PeerSpec(shard_id="self", host="127.0.0.1", udp_port=self_port),
        peers=[PeerSpec(shard_id="peer", host="127.0.0.1", udp_port=peer_port)],
        config=SwimConfig(),
    )


def test_announce_enqueues_delta():
    r = _make_runner(42001, 42002)
    try:
        r.announce_ownership_add(layer_idx=15, expert_id=7)
        assert len(r._outbound_ownership) == 1
        d = r._outbound_ownership[0]
        assert d.record.shard_id == "self"
        assert d.record.layer_idx == 15
        assert d.record.expert_id == 7
        assert d.ttl == 5  # default TTL
    finally:
        r.stop()


def test_ownership_view_includes_received_and_self():
    r = _make_runner(42003, 42004)
    try:
        r.announce_ownership_add(layer_idx=15, expert_id=7)
        # Simulate a received delta from a peer.
        r._on_recv_decoded(PingMsg(
            from_shard_id="peer", from_incarnation=1, deltas=[],
            ownership=[OwnershipDeltaRecord(
                shard_id="peer", layer_idx=15, expert_id=3, action=0, ts_unix_ms=1
            )],
        ))
        view = r.ownership_view()
        assert ("self", 15, 7) in view
        assert ("peer", 15, 3) in view
    finally:
        r.stop()


def test_drain_ownership_decrements_ttl():
    r = _make_runner(42005, 42006)
    try:
        r.announce_ownership_add(layer_idx=15, expert_id=7)
        first = r._drain_outbound_ownership()
        assert len(first) == 1
        # TTL should be 4 after one drain.
        remaining = r._outbound_ownership[0]
        assert remaining.ttl == 4
    finally:
        r.stop()


def test_drain_evicts_after_ttl_zero():
    r = _make_runner(42007, 42008)
    try:
        r.announce_ownership_add(layer_idx=15, expert_id=7)
        for _ in range(5):
            r._drain_outbound_ownership()
        assert r._outbound_ownership == []
    finally:
        r.stop()
