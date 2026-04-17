"""Dataclass shape tests for Phase 5b membership records."""
from __future__ import annotations

from model_shard.membership.records import (
    AckMsg,
    HeatReportRecord,
    OwnershipDeltaRecord,
    PingMsg,
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
