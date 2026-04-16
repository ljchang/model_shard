"""Pure state machine tests. Virtual clock; no sockets, no threads."""

import random
from typing import Any

from model_shard.membership.config import SwimConfig
from model_shard.membership.records import (
    AckMsg,
    JoinMsg,
    MemberRecord,
    MembershipDeltaMsg,
    MemberState,
    PingMsg,
    PingReqAckMsg,
    PingReqMsg,
)
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


def test_tick_promotes_suspect_to_dead_at_deadline() -> None:
    s = make_state(peers=("n1", "n2", "n3", "n4"), seed=0)
    probe = _drive_to_indirect_phase(s)
    target = probe.target_id
    for helper in probe.indirect_targets:
        s.recv(
            PingReqAckMsg(
                from_shard_id=helper,
                target_shard_id=target,
                probe_id=probe.probe_id,
                success=False,
                deltas=[],
            ),
            now=1.7,
        )
    # Suspect deadline = 1.7 + 4.0 = 5.7
    s.tick(now=5.7)
    assert s.view()[target].state == MemberState.DEAD


def test_tick_does_not_promote_before_deadline() -> None:
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
    s.tick(now=5.0)  # before 1.7 + 4.0 = 5.7
    assert s.view()[probe.target_id].state == MemberState.SUSPECT


def _suspect_self_record(self_id: str, incarnation: int) -> MemberRecord:
    return MemberRecord(
        shard_id=self_id,
        host="127.0.0.1",
        udp_port=10000,
        state=MemberState.SUSPECT,
        incarnation=incarnation,
        last_state_change=10.0,
        suspect_deadline=14.0,
    )


def test_recv_ping_with_suspect_self_delta_bumps_own_incarnation() -> None:
    s = make_state(self_id="n0", peers=("n1",))
    delta = _suspect_self_record("n0", incarnation=0)
    msg = PingMsg(from_shard_id="n1", from_incarnation=0, deltas=[delta])
    s.recv(msg, now=10.0)
    assert s._self_incarnation == 1
    assert s.view()["n0"].state == MemberState.ALIVE
    assert s.view()["n0"].incarnation == 1


def test_refutation_emits_alive_self_in_ack_deltas() -> None:
    s = make_state(self_id="n0", peers=("n1",))
    delta = _suspect_self_record("n0", incarnation=0)
    msg = PingMsg(from_shard_id="n1", from_incarnation=0, deltas=[delta])
    out = s.recv(msg, now=10.0)
    assert len(out) == 1
    payload = out[0].payload
    assert isinstance(payload, AckMsg)
    refutation = next((d for d in payload.deltas if d.shard_id == "n0"), None)
    assert refutation is not None
    assert refutation.state == MemberState.ALIVE
    assert refutation.incarnation == 1


def test_refutation_floors_to_higher_incarnation_after_restart() -> None:
    """Simulates: this node was at incarnation 5, was marked dead, restarted
    at incarnation 0. Gossip arrives saying 'n0 dead at inc=5'. Node must
    refute at inc=6, not inc=1."""
    s = make_state(self_id="n0", peers=("n1",))
    assert s._self_incarnation == 0
    delta = MemberRecord(
        shard_id="n0",
        host="127.0.0.1",
        udp_port=10000,
        state=MemberState.DEAD,
        incarnation=5,
        last_state_change=10.0,
        suspect_deadline=None,
    )
    s.recv(PingMsg(from_shard_id="n1", from_incarnation=0, deltas=[delta]), now=10.0)
    assert s._self_incarnation == 6
    assert s.view()["n0"].incarnation == 6
    assert s.view()["n0"].state == MemberState.ALIVE


def test_stale_gossip_about_self_at_lower_incarnation_is_ignored() -> None:
    s = make_state(self_id="n0", peers=("n1",))
    # First lift our incarnation to 5.
    s.recv(
        PingMsg(
            from_shard_id="n1",
            from_incarnation=0,
            deltas=[
                MemberRecord(
                    shard_id="n0",
                    host="127.0.0.1",
                    udp_port=10000,
                    state=MemberState.SUSPECT,
                    incarnation=4,
                    last_state_change=1.0,
                    suspect_deadline=5.0,
                )
            ],
        ),
        now=1.0,
    )
    assert s._self_incarnation == 5
    # Stale dead-at-inc=2 must not affect us.
    s.recv(
        PingMsg(
            from_shard_id="n1",
            from_incarnation=0,
            deltas=[
                MemberRecord(
                    shard_id="n0",
                    host="127.0.0.1",
                    udp_port=10000,
                    state=MemberState.DEAD,
                    incarnation=2,
                    last_state_change=2.0,
                    suspect_deadline=None,
                )
            ],
        ),
        now=2.0,
    )
    assert s._self_incarnation == 5
    assert s.view()["n0"].state == MemberState.ALIVE


def test_same_incarnation_dead_overrides_alive() -> None:
    # n1 currently alive at inc=0. Gossip says n1 dead at inc=0.
    dead_n1 = MemberRecord("n1", "127.0.0.1", 10001, MemberState.DEAD, 0, 5.0, None)
    s2 = make_state(self_id="me", peers=("n1", "src"))
    s2.recv(
        PingMsg(from_shard_id="src", from_incarnation=0, deltas=[dead_n1]),
        now=5.0,
    )
    assert s2.view()["n1"].state == MemberState.DEAD


def test_same_incarnation_suspect_overrides_alive() -> None:
    s = make_state(self_id="me", peers=("n1", "src"))
    suspect = MemberRecord(
        "n1", "127.0.0.1", 10001, MemberState.SUSPECT, 0, 5.0, 9.0
    )
    s.recv(
        PingMsg(from_shard_id="src", from_incarnation=0, deltas=[suspect]),
        now=5.0,
    )
    assert s.view()["n1"].state == MemberState.SUSPECT


def test_same_incarnation_alive_does_not_override_dead() -> None:
    s = make_state(self_id="me", peers=("n1", "src"))
    # First mark n1 dead via gossip at inc=2.
    dead = MemberRecord("n1", "127.0.0.1", 10001, MemberState.DEAD, 2, 1.0, None)
    s.recv(PingMsg(from_shard_id="src", from_incarnation=0, deltas=[dead]), now=1.0)
    assert s.view()["n1"].state == MemberState.DEAD
    # Now alive gossip at the same inc must not resurrect.
    alive = MemberRecord("n1", "127.0.0.1", 10001, MemberState.ALIVE, 2, 2.0, None)
    s.recv(PingMsg(from_shard_id="src", from_incarnation=0, deltas=[alive]), now=2.0)
    assert s.view()["n1"].state == MemberState.DEAD


def test_higher_incarnation_alive_does_resurrect_dead() -> None:
    s = make_state(self_id="me", peers=("n1", "src"))
    dead = MemberRecord("n1", "127.0.0.1", 10001, MemberState.DEAD, 2, 1.0, None)
    s.recv(PingMsg(from_shard_id="src", from_incarnation=0, deltas=[dead]), now=1.0)
    alive = MemberRecord("n1", "127.0.0.1", 10001, MemberState.ALIVE, 3, 2.0, None)
    s.recv(PingMsg(from_shard_id="src", from_incarnation=0, deltas=[alive]), now=2.0)
    assert s.view()["n1"].state == MemberState.ALIVE
    assert s.view()["n1"].incarnation == 3


def test_outgoing_pings_carry_recent_transitions_in_deltas() -> None:
    s = make_state(self_id="me", peers=("n1", "src", "n2"), seed=0)
    # Mark n2 dead via incoming gossip → that adds a transition to the backlog.
    dead = MemberRecord("n2", "127.0.0.1", 10003, MemberState.DEAD, 5, 1.0, None)
    s.recv(PingMsg(from_shard_id="src", from_incarnation=0, deltas=[dead]), now=1.0)
    # Drive a protocol period; expect outgoing Ping to carry the n2-dead delta.
    out = s.tick(now=1.0)
    pings = [m for m in out if isinstance(m.payload, PingMsg)]
    assert len(pings) == 1
    payload = pings[0].payload
    assert isinstance(payload, PingMsg)
    n2_delta = next((d for d in payload.deltas if d.shard_id == "n2"), None)
    assert n2_delta is not None
    assert n2_delta.state == MemberState.DEAD


def test_backlog_caps_at_k_gossip_per_message() -> None:
    cfg = SwimConfig(k_gossip=2)
    s = make_state(
        self_id="me", peers=("a", "b", "c", "d", "src"), seed=0, cfg=cfg
    )
    # Inject 4 transitions (more than K_GOSSIP).
    for i, name in enumerate(("a", "b", "c", "d")):
        d = MemberRecord(name, "127.0.0.1", 10001 + i, MemberState.DEAD, 1, 1.0, None)
        s.recv(PingMsg(from_shard_id="src", from_incarnation=0, deltas=[d]), now=1.0)
    out = s.tick(now=1.0)
    pings = [m for m in out if isinstance(m.payload, PingMsg)]
    payload = pings[0].payload
    assert isinstance(payload, PingMsg)
    # K_GOSSIP=2 entries from backlog, plus self always — 3 total at most.
    assert len(payload.deltas) <= 3


def test_backlog_drains_oldest_first_across_calls() -> None:
    cfg = SwimConfig(k_gossip=1)
    s = make_state(self_id="me", peers=("a", "b", "src"), seed=0, cfg=cfg)
    da = MemberRecord("a", "127.0.0.1", 10001, MemberState.DEAD, 1, 1.0, None)
    db = MemberRecord("b", "127.0.0.1", 10002, MemberState.DEAD, 1, 1.0, None)
    s.recv(PingMsg(from_shard_id="src", from_incarnation=0, deltas=[da]), now=1.0)
    s.recv(PingMsg(from_shard_id="src", from_incarnation=0, deltas=[db]), now=1.0)
    # Two ticks should drain a, then b.
    out1 = s.tick(now=1.0)
    out2 = s.tick(now=2.0)

    def first_non_self(p: PingMsg) -> str | None:
        return next((d.shard_id for d in p.deltas if d.shard_id != "me"), None)

    p1 = out1[0].payload
    p2 = out2[0].payload
    assert isinstance(p1, PingMsg)
    assert isinstance(p2, PingMsg)
    assert first_non_self(p1) == "a"
    assert first_non_self(p2) == "b"


def test_recv_join_emits_membership_delta_with_full_view() -> None:
    s = make_state(self_id="seed", peers=("n1",))
    new_node = MemberRecord(
        shard_id="newcomer",
        host="127.0.0.1",
        udp_port=10099,
        state=MemberState.ALIVE,
        incarnation=0,
        last_state_change=0.0,
        suspect_deadline=None,
    )
    out = s.recv(JoinMsg(self_record=new_node), now=3.0)
    deltas = [m for m in out if isinstance(m.payload, MembershipDeltaMsg)]
    assert len(deltas) == 1
    payload = deltas[0].payload
    assert isinstance(payload, MembershipDeltaMsg)
    ids = {m.shard_id for m in payload.members}
    assert {"seed", "n1", "newcomer"} <= ids
    assert deltas[0].target_shard_id == "newcomer"


def test_recv_join_installs_unknown_newcomer_in_view() -> None:
    s = make_state(self_id="seed", peers=("n1",))
    new_node = MemberRecord(
        shard_id="newcomer",
        host="127.0.0.1",
        udp_port=10099,
        state=MemberState.ALIVE,
        incarnation=0,
        last_state_change=0.0,
        suspect_deadline=None,
    )
    s.recv(JoinMsg(self_record=new_node), now=3.0)
    assert "newcomer" in s.view()
    assert s.view()["newcomer"].state == MemberState.ALIVE


def test_gossip_about_unknown_shard_id_is_dropped(caplog: Any) -> None:
    import logging
    s = make_state(self_id="me", peers=("n1", "src"))
    ghost = MemberRecord(
        "ghost-shard", "10.0.0.99", 10099, MemberState.ALIVE, 0, 1.0, None
    )
    with caplog.at_level(logging.WARNING, logger="model_shard.membership.state"):
        s.recv(
            PingMsg(from_shard_id="src", from_incarnation=0, deltas=[ghost]),
            now=1.0,
        )
    assert "ghost-shard" not in s.view()
    assert any("ghost-shard" in r.message for r in caplog.records)
