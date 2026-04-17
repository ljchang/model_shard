"""Dataclass shape tests for Phase 5b membership records."""
from __future__ import annotations

from model_shard.membership.messages import (
    decode_membership_envelope,
    encode_membership_envelope,
)
from model_shard.membership.records import (
    AckMsg,
    HeatReportRecord,
    OwnershipDeltaRecord,
    PingMsg,
    PingReqAckMsg,
    PingReqMsg,
)


def test_heat_report_record_is_frozen_and_equal_by_value():
    a = HeatReportRecord(
        shard_id="a",
        entries=((15, 7, 500),),
        ts_unix_ms=1234,
    )
    b = HeatReportRecord(
        shard_id="a",
        entries=((15, 7, 500),),
        ts_unix_ms=1234,
    )
    assert a == b
    # Frozen: assignment should raise.
    try:
        a.shard_id = "b"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("HeatReportRecord should be frozen")


def test_ownership_delta_record_add():
    d = OwnershipDeltaRecord(
        shard_id="a", layer_idx=15, expert_id=7, action=0, ts_unix_ms=1
    )
    assert d.action == 0


def test_ping_carries_heat_and_ownership_defaults_empty():
    p = PingMsg(
        from_shard_id="a", from_incarnation=1, deltas=[],
    )
    assert p.heat == []
    assert p.ownership == []


def test_ack_carries_heat_and_ownership():
    hr = HeatReportRecord(shard_id="a", entries=((15, 7, 500),), ts_unix_ms=1)
    od = OwnershipDeltaRecord(shard_id="a", layer_idx=15, expert_id=7, action=0, ts_unix_ms=1)
    a = AckMsg(
        from_shard_id="a", from_incarnation=1, deltas=[],
        heat=[hr], ownership=[od],
    )
    assert a.heat == [hr]
    assert a.ownership == [od]


def test_ping_heat_and_ownership_roundtrip():
    p = PingMsg(
        from_shard_id="a",
        from_incarnation=1,
        deltas=[],
        heat=[HeatReportRecord(
            shard_id="a", entries=((15, 7, 500), (15, 3, 300)), ts_unix_ms=1
        )],
        ownership=[OwnershipDeltaRecord(
            shard_id="a", layer_idx=15, expert_id=7, action=0, ts_unix_ms=1
        )],
    )
    raw = encode_membership_envelope(p)
    parsed = decode_membership_envelope(raw)
    assert isinstance(parsed, PingMsg)
    assert parsed.heat == p.heat
    assert parsed.ownership == p.ownership


def test_ack_heat_and_ownership_roundtrip():
    a = AckMsg(
        from_shard_id="a",
        from_incarnation=1,
        deltas=[],
        heat=[HeatReportRecord(
            shard_id="a", entries=((15, 7, 500),), ts_unix_ms=1
        )],
        ownership=[OwnershipDeltaRecord(
            shard_id="a", layer_idx=15, expert_id=7, action=0, ts_unix_ms=1
        )],
    )
    raw = encode_membership_envelope(a)
    parsed = decode_membership_envelope(raw)
    assert isinstance(parsed, AckMsg)
    assert parsed.heat == a.heat
    assert parsed.ownership == a.ownership


def test_ping_req_heat_and_ownership_roundtrip():
    pr = PingReqMsg(
        from_shard_id="a",
        target_shard_id="b",
        probe_id="p1",
        deltas=[],
        heat=[HeatReportRecord(
            shard_id="a", entries=((15, 7, 500),), ts_unix_ms=1
        )],
        ownership=[OwnershipDeltaRecord(
            shard_id="a", layer_idx=15, expert_id=7, action=0, ts_unix_ms=1
        )],
    )
    raw = encode_membership_envelope(pr)
    parsed = decode_membership_envelope(raw)
    assert isinstance(parsed, PingReqMsg)
    assert parsed.heat == pr.heat
    assert parsed.ownership == pr.ownership


def test_ping_req_ack_heat_and_ownership_roundtrip():
    pra = PingReqAckMsg(
        from_shard_id="a",
        target_shard_id="b",
        probe_id="p1",
        success=True,
        deltas=[],
        heat=[HeatReportRecord(
            shard_id="a", entries=((15, 7, 500),), ts_unix_ms=1
        )],
        ownership=[OwnershipDeltaRecord(
            shard_id="a", layer_idx=15, expert_id=7, action=0, ts_unix_ms=1
        )],
    )
    raw = encode_membership_envelope(pra)
    parsed = decode_membership_envelope(raw)
    assert isinstance(parsed, PingReqAckMsg)
    assert parsed.heat == pra.heat
    assert parsed.ownership == pra.ownership
