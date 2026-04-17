"""Unit tests for MembershipRunner heat piggyback and reception."""
from __future__ import annotations

import dataclasses

from model_shard.membership.config import SwimConfig
from model_shard.membership.records import (
    AckMsg,
    HeatReportRecord,
    PingMsg,
)
from model_shard.membership.runner import MembershipRunner
from model_shard.membership.state import PeerSpec


def _make_runner(self_port: int = 41001, peer_port: int = 41002) -> MembershipRunner:
    return MembershipRunner(
        self_spec=PeerSpec(shard_id="self", host="127.0.0.1", udp_port=self_port),
        peers=[PeerSpec(shard_id="peer", host="127.0.0.1", udp_port=peer_port)],
        config=SwimConfig(),
    )


def test_start_heat_source_registers_callable():
    r = _make_runner(41001, 41002)
    try:
        r.start_heat_source(lambda: HeatReportRecord(
            shard_id="self", entries=((15, 7, 500),), ts_unix_ms=1
        ))
        assert r._heat_source is not None  # private but the simplest signal
    finally:
        r.stop()


def test_latest_heat_updates_on_recv():
    r = _make_runner(41003, 41004)
    try:
        hr = HeatReportRecord(shard_id="peer", entries=((15, 3, 200),), ts_unix_ms=42)
        ping = PingMsg(
            from_shard_id="peer", from_incarnation=1, deltas=[],
            heat=[hr],
        )
        r._on_recv_decoded(ping)
        snap = r.latest_heat()
        assert snap["peer"] == hr
    finally:
        r.stop()


def test_latest_heat_snapshot_is_isolated():
    r = _make_runner(41005, 41006)
    try:
        hr = HeatReportRecord(shard_id="peer", entries=((15, 3, 200),), ts_unix_ms=42)
        r._on_recv_decoded(PingMsg(
            from_shard_id="peer", from_incarnation=1, deltas=[], heat=[hr],
        ))
        snap = r.latest_heat()
        snap["bogus"] = "should not propagate"  # type: ignore[assignment]
        assert "bogus" not in r.latest_heat()
    finally:
        r.stop()
