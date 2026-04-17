"""MembershipRunner exposes a pluggable load-source hook and caches peer loads."""

from __future__ import annotations

import time

from model_shard.membership.config import SwimConfig
from model_shard.membership.records import (
    LoadReportRecord,
    PingMsg,
)
from model_shard.membership.runner import MembershipRunner
from model_shard.membership.state import PeerSpec


def _spec(shard_id: str, port: int) -> PeerSpec:
    return PeerSpec(shard_id=shard_id, host="127.0.0.1", udp_port=port)


def test_runner_start_load_source_and_latest_loads_roundtrip() -> None:
    self_spec = _spec("head", 40000)
    peers = [_spec("mid", 40001), _spec("tail", 40002)]
    runner = MembershipRunner(self_spec=self_spec, peers=peers, config=SwimConfig())
    try:
        assert runner.latest_loads() == {}

        runner.start_load_source(
            lambda: LoadReportRecord(shard_id="head", queue_depth_ema=123, ts_unix_ms=0)
        )
        assert runner._load_source is not None
        assert runner._load_source().queue_depth_ema == 123

        msg = PingMsg(
            from_shard_id="mid",
            from_incarnation=0,
            deltas=[],
            loads=[LoadReportRecord(shard_id="mid", queue_depth_ema=42, ts_unix_ms=int(time.time() * 1000))],
        )
        runner._on_recv_decoded(msg)

        loads = runner.latest_loads()
        assert "mid" in loads
        assert loads["mid"].queue_depth_ema == 42
    finally:
        runner.stop()


def test_runner_latest_loads_overwrites_with_newer() -> None:
    self_spec = _spec("head", 40010)
    peers = [_spec("mid", 40011)]
    runner = MembershipRunner(self_spec=self_spec, peers=peers, config=SwimConfig())
    try:
        msg = PingMsg(
            from_shard_id="mid", from_incarnation=0, deltas=[],
            loads=[LoadReportRecord(shard_id="mid", queue_depth_ema=1, ts_unix_ms=1)],
        )
        runner._on_recv_decoded(msg)
        msg2 = PingMsg(
            from_shard_id="mid", from_incarnation=0, deltas=[],
            loads=[LoadReportRecord(shard_id="mid", queue_depth_ema=99, ts_unix_ms=9999)],
        )
        runner._on_recv_decoded(msg2)
        assert runner.latest_loads()["mid"].queue_depth_ema == 99
    finally:
        runner.stop()
