"""Round-trip tests for LoadReportRecord via encode/decode_membership_envelope."""

from __future__ import annotations

from model_shard.membership.messages import (
    decode_membership_envelope,
    encode_membership_envelope,
)
from model_shard.membership.records import (
    AckMsg,
    LoadReportRecord,
    PingMsg,
)


def test_ping_loads_roundtrip() -> None:
    msg = PingMsg(
        from_shard_id="head",
        from_incarnation=1,
        deltas=[],
        loads=[
            LoadReportRecord(shard_id="head", queue_depth_ema=250, ts_unix_ms=100),
            LoadReportRecord(shard_id="mid", queue_depth_ema=50, ts_unix_ms=100),
        ],
    )
    raw = encode_membership_envelope(msg)
    got = decode_membership_envelope(raw)
    assert isinstance(got, PingMsg)
    assert len(got.loads) == 2
    assert got.loads[0].shard_id == "head"
    assert got.loads[0].queue_depth_ema == 250
    assert got.loads[1].shard_id == "mid"


def test_ack_loads_absent_defaults_empty() -> None:
    msg = AckMsg(from_shard_id="mid", from_incarnation=3, deltas=[])
    raw = encode_membership_envelope(msg)
    got = decode_membership_envelope(raw)
    assert isinstance(got, AckMsg)
    assert got.loads == []
