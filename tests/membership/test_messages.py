import pytest

from model_shard._pb import wire_pb2
from model_shard.membership.messages import (
    decode_membership_envelope,
    encode_membership_envelope,
)
from model_shard.membership.records import (
    AckMsg,
    IncomingMessage,
    JoinMsg,
    MemberRecord,
    MembershipDeltaMsg,
    MemberState,
    PingMsg,
    PingReqAckMsg,
    PingReqMsg,
)


def _rec(shard_id: str, state: MemberState = MemberState.ALIVE) -> MemberRecord:
    return MemberRecord(
        shard_id=shard_id,
        host="127.0.0.1",
        udp_port=10001,
        state=state,
        incarnation=3,
        model_id="",
        last_state_change=1.0,
        suspect_deadline=None,
    )


@pytest.mark.parametrize(
    "msg",
    [
        PingMsg(from_shard_id="a", from_incarnation=2, deltas=[_rec("b")]),
        AckMsg(from_shard_id="a", from_incarnation=2, deltas=[]),
        PingReqMsg(
            from_shard_id="a", target_shard_id="b", probe_id="p1", deltas=[]
        ),
        PingReqAckMsg(
            from_shard_id="a",
            target_shard_id="b",
            probe_id="p1",
            success=True,
            deltas=[_rec("c", MemberState.SUSPECT)],
        ),
        JoinMsg(self_record=_rec("a")),
        MembershipDeltaMsg(members=[_rec("a"), _rec("b", MemberState.DEAD)]),
    ],
)
def test_round_trip_through_protobuf(msg: IncomingMessage) -> None:
    raw = encode_membership_envelope(msg)
    decoded = decode_membership_envelope(raw)
    assert decoded == msg


def test_decode_unknown_envelope_oneof_returns_none() -> None:
    env = wire_pb2.Envelope()
    env.begin.protocol_version = 1  # not a membership message
    raw = env.SerializeToString()
    assert decode_membership_envelope(raw) is None
