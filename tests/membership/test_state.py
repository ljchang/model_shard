"""Pure state machine tests. Virtual clock; no sockets, no threads."""

import random
from typing import Any

from model_shard.membership.config import SwimConfig
from model_shard.membership.records import AckMsg, MemberState, PingMsg, PingReqAckMsg, PingReqMsg
from model_shard.membership.state import MembershipState, PeerSpec


def make_state(
    self_id: str = "n0",
    peers: tuple[str, ...] = ("n1", "n2"),
    seed: int = 0,
    cfg: SwimConfig | None = None,
) -> MembershipState:
    """Test helper: build a MembershipState with the named peers."""
    self_spec = PeerSpec(shard_id=self_id, host="127.0.0.1", udp_port=10000)
    peer_specs = [
        PeerSpec(shard_id=p, host="127.0.0.1", udp_port=10000 + i + 1)
        for i, p in enumerate(peers)
    ]
    return MembershipState(
        self_spec=self_spec,
        peer_specs=peer_specs,
        rng=random.Random(seed),
        config=cfg or SwimConfig(),
    )


def test_initial_view_contains_self_alive_at_incarnation_zero() -> None:
    s = make_state()
    view = s.view()
    assert "n0" in view
    rec = view["n0"]
    assert rec.state == MemberState.ALIVE
    assert rec.incarnation == 0


def test_initial_view_contains_each_peer_alive_at_incarnation_zero() -> None:
    s = make_state(peers=("n1", "n2", "n3"))
    view = s.view()
    for name in ("n1", "n2", "n3"):
        assert name in view, f"missing peer {name}"
        assert view[name].state == MemberState.ALIVE
        assert view[name].incarnation == 0


def test_view_returns_a_copy_not_internal_reference() -> None:
    s = make_state()
    view = s.view()
    view.clear()
    # Mutating the returned dict must not affect internal state.
    assert "n0" in s.view()


def test_tick_emits_no_message_before_first_protocol_period() -> None:
    s = make_state()
    # The first protocol period fires at t = T_PING; before that, no ping.
    out = s.tick(now=0.5)
    assert out == []


def test_tick_emits_ping_at_first_protocol_period() -> None:
    s = make_state(peers=("n1", "n2"), seed=0)
    out = s.tick(now=1.0)  # exactly T_PING = 1000ms
    assert len(out) == 1
    msg = out[0]
    assert isinstance(msg.payload, PingMsg)
    assert msg.payload.from_shard_id == "n0"
    assert msg.payload.from_incarnation == 0
    # Target must be one of the peers, never self.
    assert msg.target_shard_id in {"n1", "n2"}
    assert msg.target_shard_id != "n0"


def test_tick_does_not_re_emit_within_one_period() -> None:
    s = make_state(seed=0)
    s.tick(now=1.0)
    # 1.4s is before the 500ms ack-timeout, so no escalation yet.
    out = s.tick(now=1.4)
    assert out == []


def test_tick_emits_again_after_full_period() -> None:
    s = make_state(seed=0)
    s.tick(now=1.0)
    # Deliver ack to clear the pending probe so escalation won't fire at 2.0.
    pending_target = s._pending_probe.target_id  # type: ignore[union-attr]
    s.recv(AckMsg(from_shard_id=pending_target, from_incarnation=0, deltas=[]), now=1.2)
    out = s.tick(now=2.0)
    assert len(out) == 1


def test_tick_emits_no_ping_when_no_alive_peers() -> None:
    s = make_state(peers=())
    out = s.tick(now=10.0)
    assert out == []


def test_recv_ping_emits_ack_to_sender() -> None:
    s = make_state(self_id="n0", peers=("n1",))
    msg = PingMsg(from_shard_id="n1", from_incarnation=0, deltas=[])
    out = s.recv(msg, now=0.0)
    assert len(out) == 1
    assert out[0].target_shard_id == "n1"
    payload = out[0].payload
    assert isinstance(payload, AckMsg)
    assert payload.from_shard_id == "n0"
    assert payload.from_incarnation == 0


def test_recv_ping_from_unknown_peer_is_dropped() -> None:
    s = make_state(self_id="n0", peers=("n1",))
    msg = PingMsg(from_shard_id="ghost", from_incarnation=0, deltas=[])
    out = s.recv(msg, now=0.0)
    assert out == []


def test_recv_ack_clears_pending_probe() -> None:
    s = make_state(seed=0)
    s.tick(now=1.0)  # produces a Ping
    pending_target = s._pending_probe.target_id  # type: ignore[union-attr]
    ack = AckMsg(from_shard_id=pending_target, from_incarnation=0, deltas=[])
    s.recv(ack, now=1.2)
    assert s._pending_probe is None


def test_recv_ack_from_unrelated_peer_is_ignored() -> None:
    s = make_state(seed=0)
    s.tick(now=1.0)
    pending = s._pending_probe
    assert pending is not None
    # Determine the "other" peer (not the target of the pending probe)
    other = "n2" if pending.target_id == "n1" else "n1"
    ack = AckMsg(from_shard_id=other, from_incarnation=0, deltas=[])
    s.recv(ack, now=1.2)
    # pending probe still in place
    assert s._pending_probe is pending


def test_tick_escalates_to_pingreq_after_timeout() -> None:
    # 4 peers so K_INDIRECT=2 random peers can be picked, plus the target.
    s = make_state(self_id="n0", peers=("n1", "n2", "n3", "n4"), seed=0)
    s.tick(now=1.0)
    target = s._pending_probe.target_id  # type: ignore[union-attr]

    # T_TIMEOUT = 500ms; escalate at t = 1.5s.
    out = s.tick(now=1.5)

    pingreqs = [m for m in out if isinstance(m.payload, PingReqMsg)]
    assert len(pingreqs) == 2  # K_INDIRECT
    for m in pingreqs:
        assert m.target_shard_id != target
        assert m.target_shard_id != "n0"
        payload = m.payload
        assert isinstance(payload, PingReqMsg)
        assert payload.target_shard_id == target


def test_tick_does_not_escalate_twice() -> None:
    s = make_state(peers=("n1", "n2", "n3", "n4"), seed=0)
    s.tick(now=1.0)
    s.tick(now=1.5)  # first escalation
    out = s.tick(now=1.6)
    assert all(not isinstance(m.payload, PingReqMsg) for m in out)


def test_tick_does_not_escalate_if_ack_arrived_first() -> None:
    s = make_state(peers=("n1", "n2", "n3", "n4"), seed=0)
    s.tick(now=1.0)
    target = s._pending_probe.target_id  # type: ignore[union-attr]
    s.recv(AckMsg(from_shard_id=target, from_incarnation=0, deltas=[]), now=1.2)
    out = s.tick(now=1.5)
    assert all(not isinstance(m.payload, PingReqMsg) for m in out)


def test_recv_pingreq_emits_ping_to_target_and_tracks_help() -> None:
    s = make_state(self_id="helper", peers=("requester", "target"), seed=0)
    msg = PingReqMsg(
        from_shard_id="requester",
        target_shard_id="target",
        probe_id="r:1",
        deltas=[],
    )
    out = s.recv(msg, now=2.0)
    pings = [m for m in out if isinstance(m.payload, PingMsg)]
    assert len(pings) == 1
    assert pings[0].target_shard_id == "target"
    # No PingReqAck yet — we await the target's Ack.
    assert all(not isinstance(m.payload, PingReqAckMsg) for m in out)


def test_recv_target_ack_during_help_emits_pingreqack_success() -> None:
    s = make_state(self_id="helper", peers=("requester", "target"), seed=0)
    s.recv(
        PingReqMsg(
            from_shard_id="requester",
            target_shard_id="target",
            probe_id="r:1",
            deltas=[],
        ),
        now=2.0,
    )
    out = s.recv(
        AckMsg(from_shard_id="target", from_incarnation=0, deltas=[]), now=2.1
    )
    pra = [m for m in out if isinstance(m.payload, PingReqAckMsg)]
    assert len(pra) == 1
    assert pra[0].target_shard_id == "requester"
    payload = pra[0].payload
    assert isinstance(payload, PingReqAckMsg)
    assert payload.success is True
    assert payload.probe_id == "r:1"


def test_help_times_out_emits_pingreqack_failure() -> None:
    s = make_state(self_id="helper", peers=("requester", "target"), seed=0)
    s.recv(
        PingReqMsg(
            from_shard_id="requester",
            target_shard_id="target",
            probe_id="r:1",
            deltas=[],
        ),
        now=2.0,
    )
    # T_TIMEOUT = 500ms — helper gives up at t=2.5
    out = s.tick(now=2.5)
    pra = [m for m in out if isinstance(m.payload, PingReqAckMsg)]
    assert len(pra) == 1
    payload = pra[0].payload
    assert isinstance(payload, PingReqAckMsg)
    assert payload.success is False
    assert pra[0].target_shard_id == "requester"


def _drive_to_indirect_phase(s: MembershipState) -> Any:
    """Helper: advance s to the post-escalation phase and return the probe."""
    s.tick(now=1.0)
    s.tick(now=1.5)  # escalates to PingReq
    probe = s._pending_probe
    assert probe is not None
    assert probe.indirect_sent_at is not None
    return probe


def test_positive_pingreqack_clears_probe() -> None:
    s = make_state(peers=("n1", "n2", "n3", "n4"), seed=0)
    probe = _drive_to_indirect_phase(s)
    helper = probe.indirect_targets[0]
    s.recv(
        PingReqAckMsg(
            from_shard_id=helper,
            target_shard_id=probe.target_id,
            probe_id=probe.probe_id,
            success=True,
            deltas=[],
        ),
        now=1.6,
    )
    assert s._pending_probe is None


def test_all_negative_pingreqacks_mark_target_suspect() -> None:
    s = make_state(peers=("n1", "n2", "n3", "n4"), seed=0)
    probe = _drive_to_indirect_phase(s)
    for helper in probe.indirect_targets:
        s.recv(
            PingReqAckMsg(
                from_shard_id=helper,
                target_shard_id=probe.target_id,
                probe_id=probe.probe_id,
                success=False,
                deltas=[],
            ),
            now=1.7,
        )
    rec = s.view()[probe.target_id]
    assert rec.state == MemberState.SUSPECT
    assert rec.suspect_deadline is not None
    # deadline = now + T_SUSPECT (4000ms)
    assert abs(rec.suspect_deadline - (1.7 + 4.0)) < 1e-9


def test_partial_negative_pingreqacks_does_not_mark_suspect() -> None:
    s = make_state(peers=("n1", "n2", "n3", "n4"), seed=0)
    probe = _drive_to_indirect_phase(s)
    helper = probe.indirect_targets[0]
    s.recv(
        PingReqAckMsg(
            from_shard_id=helper,
            target_shard_id=probe.target_id,
            probe_id=probe.probe_id,
            success=False,
            deltas=[],
        ),
        now=1.7,
    )
    rec = s.view()[probe.target_id]
    assert rec.state == MemberState.ALIVE
