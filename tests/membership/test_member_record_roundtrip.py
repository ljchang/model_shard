"""Roundtrip tests for MemberRecord <-> protobuf serialization."""
from model_shard.membership.messages import _record_from_pb, _record_to_pb
from model_shard.membership.records import MemberRecord, MemberState


def test_record_roundtrip_preserves_model_id():
    r = MemberRecord(
        shard_id="x", host="127.0.0.1", udp_port=9001,
        state=MemberState.ALIVE, incarnation=42,
        model_id="google/gemma-4-26B-A4B-it",
        last_state_change=0.0, suspect_deadline=None,
    )
    rt = _record_from_pb(_record_to_pb(r))
    assert rt.model_id == "google/gemma-4-26B-A4B-it"
