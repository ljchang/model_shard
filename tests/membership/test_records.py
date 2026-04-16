import dataclasses

from model_shard.membership.records import (
    AckMsg,
    JoinMsg,
    MemberRecord,
    MembershipDeltaMsg,
    MemberState,
    PingMsg,
    PingReqAckMsg,
    PingReqMsg,
    StateTransition,
)


def test_member_state_ordering_is_dead_gt_suspect_gt_alive() -> None:
    # The numeric values must match the wire MemberRecordPb.state encoding.
    assert MemberState.ALIVE.value == 0
    assert MemberState.SUSPECT.value == 1
    assert MemberState.DEAD.value == 2
    # Severity ordering is used by the same-incarnation tiebreaker.
    assert MemberState.DEAD > MemberState.SUSPECT > MemberState.ALIVE


def test_member_record_is_immutable_dataclass() -> None:
    rec = MemberRecord(
        shard_id="x",
        host="127.0.0.1",
        udp_port=10001,
        state=MemberState.ALIVE,
        incarnation=0,
        last_state_change=0.0,
        suspect_deadline=None,
    )
    assert dataclasses.is_dataclass(rec)
    try:
        rec.incarnation = 1  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("MemberRecord should be frozen")


def test_state_transition_carries_old_and_new() -> None:
    rec = MemberRecord("x", "127.0.0.1", 10001, MemberState.SUSPECT, 3, 1.0, 5.0)
    t = StateTransition(
        shard_id="x",
        old_state=MemberState.ALIVE,
        new_record=rec,
    )
    assert t.shard_id == "x"
    assert t.old_state == MemberState.ALIVE
    assert t.new_record.state == MemberState.SUSPECT


def test_message_dataclasses_construct() -> None:
    rec = MemberRecord("x", "127.0.0.1", 10001, MemberState.ALIVE, 0, 0.0, None)
    PingMsg(from_shard_id="a", from_incarnation=2, deltas=[rec])
    AckMsg(from_shard_id="a", from_incarnation=2, deltas=[rec])
    PingReqMsg(from_shard_id="a", target_shard_id="b", probe_id="p1", deltas=[])
    PingReqAckMsg(from_shard_id="a", target_shard_id="b", probe_id="p1", success=True, deltas=[])
    JoinMsg(self_record=rec)
    MembershipDeltaMsg(members=[rec])
