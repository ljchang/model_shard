"""Phase 6-C: ADD/REMOVE convergence via last-writer-wins on ts_unix_ms."""
from __future__ import annotations

from model_shard.membership.config import SwimConfig
from model_shard.membership.records import (
    OwnershipDeltaRecord,
    PingMsg,
)
from model_shard.membership.runner import MembershipRunner
from model_shard.membership.state import PeerSpec


def _make_runner(self_port: int, peer_port: int) -> MembershipRunner:
    return MembershipRunner(
        self_spec=PeerSpec(shard_id="self", host="127.0.0.1", udp_port=self_port),
        peers=[PeerSpec(shard_id="peer", host="127.0.0.1", udp_port=peer_port)],
        config=SwimConfig(),
    )


def test_add_then_remove_last_writer_wins():
    r = _make_runner(42101, 42102)
    try:
        r._on_recv_decoded(PingMsg(
            from_shard_id="peer", from_incarnation=1, deltas=[],
            ownership=[OwnershipDeltaRecord(
                shard_id="peer", layer_idx=15, expert_id=7, action=0, ts_unix_ms=1000
            )],
        ))
        assert ("peer", 15, 7) in r.ownership_view()

        r._on_recv_decoded(PingMsg(
            from_shard_id="peer", from_incarnation=1, deltas=[],
            ownership=[OwnershipDeltaRecord(
                shard_id="peer", layer_idx=15, expert_id=7, action=1, ts_unix_ms=2000
            )],
        ))
        assert ("peer", 15, 7) not in r.ownership_view()
    finally:
        r.stop()


def test_remove_then_older_add_drops():
    r = _make_runner(42103, 42104)
    try:
        # Receive a REMOVE at t=2000 first.
        r._on_recv_decoded(PingMsg(
            from_shard_id="peer", from_incarnation=1, deltas=[],
            ownership=[OwnershipDeltaRecord(
                shard_id="peer", layer_idx=15, expert_id=7, action=1, ts_unix_ms=2000
            )],
        ))
        # Then an ADD from t=1000 (older, stale).
        r._on_recv_decoded(PingMsg(
            from_shard_id="peer", from_incarnation=1, deltas=[],
            ownership=[OwnershipDeltaRecord(
                shard_id="peer", layer_idx=15, expert_id=7, action=0, ts_unix_ms=1000
            )],
        ))
        # REMOVE is newer → view should NOT include the key as an owner.
        assert ("peer", 15, 7) not in r.ownership_view()
    finally:
        r.stop()


def test_announce_remove_enqueues_delta():
    r = _make_runner(42105, 42106)
    try:
        r.announce_ownership_remove(layer_idx=15, expert_id=7)
        assert len(r._outbound_ownership) == 1
        d = r._outbound_ownership[0]
        assert d.record.shard_id == "self"
        assert d.record.action == 1
    finally:
        r.stop()


def test_announce_remove_updates_local_view_immediately():
    r = _make_runner(42107, 42108)
    try:
        # First announce an ADD so self owns the expert.
        r.announce_ownership_add(layer_idx=15, expert_id=7)
        assert ("self", 15, 7) in r.ownership_view()
        # Then REMOVE — local view must update synchronously.
        r.announce_ownership_remove(layer_idx=15, expert_id=7)
        assert ("self", 15, 7) not in r.ownership_view()
    finally:
        r.stop()


def test_ownership_view_returns_only_adds():
    r = _make_runner(42109, 42110)
    try:
        # One ADD from peer, one REMOVE from other_peer.
        r._on_recv_decoded(PingMsg(
            from_shard_id="peer", from_incarnation=1, deltas=[],
            ownership=[
                OwnershipDeltaRecord(
                    shard_id="peer", layer_idx=15, expert_id=1, action=0, ts_unix_ms=1000
                ),
                OwnershipDeltaRecord(
                    shard_id="other", layer_idx=15, expert_id=2, action=1, ts_unix_ms=1000
                ),
            ],
        ))
        view = r.ownership_view()
        assert ("peer", 15, 1) in view
        assert ("other", 15, 2) not in view
    finally:
        r.stop()
