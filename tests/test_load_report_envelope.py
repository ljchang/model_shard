"""Roundtrip tests for the Phase 4 LoadReport piggybacked on SWIM messages."""

from __future__ import annotations

from model_shard._pb import wire_pb2


def test_load_report_roundtrip_on_ping() -> None:
    env = wire_pb2.Envelope()
    env.ping.protocol_version = 1
    env.ping.from_shard_id = "head"
    env.ping.from_incarnation = 7
    lr = env.ping.loads.add()
    lr.shard_id = "head"
    lr.queue_depth_ema = 250
    lr.ts_unix_ms = 1713000000_000

    raw = env.SerializeToString()
    out = wire_pb2.Envelope()
    out.ParseFromString(raw)
    assert out.WhichOneof("payload") == "ping"
    assert len(out.ping.loads) == 1
    assert out.ping.loads[0].shard_id == "head"
    assert out.ping.loads[0].queue_depth_ema == 250
    assert out.ping.loads[0].ts_unix_ms == 1713000000_000


def test_load_report_roundtrip_on_ack_multiple_entries() -> None:
    env = wire_pb2.Envelope()
    env.ack.protocol_version = 1
    env.ack.from_shard_id = "mid"
    env.ack.from_incarnation = 2
    for sid, ema in [("head", 100), ("mid", 50), ("tail", 300)]:
        lr = env.ack.loads.add()
        lr.shard_id = sid
        lr.queue_depth_ema = ema
        lr.ts_unix_ms = 0

    out = wire_pb2.Envelope()
    out.ParseFromString(env.SerializeToString())
    sids = [lr.shard_id for lr in out.ack.loads]
    emas = [lr.queue_depth_ema for lr in out.ack.loads]
    assert sids == ["head", "mid", "tail"]
    assert emas == [100, 50, 300]


def test_load_report_absent_on_ping_req_defaults_empty() -> None:
    env = wire_pb2.Envelope()
    env.ping_req.protocol_version = 1
    env.ping_req.from_shard_id = "head"
    env.ping_req.target_shard_id = "mid"
    env.ping_req.probe_id = "p1"
    out = wire_pb2.Envelope()
    out.ParseFromString(env.SerializeToString())
    assert list(out.ping_req.loads) == []
