"""Protobuf <-> dataclass adapters for membership messages.

The state machine works in dataclasses (records.py); the wire is protobuf.
Keep these layers separate so the state machine never imports `_pb`.
"""

from __future__ import annotations

from model_shard._pb import wire_pb2
from model_shard.membership.records import (
    AckMsg,
    HeatReportRecord,
    IncomingMessage,
    JoinMsg,
    LoadReportRecord,
    MemberRecord,
    MembershipDeltaMsg,
    MemberState,
    OwnershipDeltaRecord,
    PingMsg,
    PingReqAckMsg,
    PingReqMsg,
)

_PROTOCOL_VERSION = 1


def _record_to_pb(r: MemberRecord) -> wire_pb2.MemberRecordPb:
    return wire_pb2.MemberRecordPb(
        shard_id=r.shard_id,
        host=r.host,
        udp_port=r.udp_port,
        state=int(r.state),
        incarnation=r.incarnation,
        model_id=r.model_id,
    )


def _record_from_pb(pb: wire_pb2.MemberRecordPb) -> MemberRecord:
    return MemberRecord(
        shard_id=pb.shard_id,
        host=pb.host,
        udp_port=int(pb.udp_port),
        state=MemberState(int(pb.state)),
        incarnation=int(pb.incarnation),
        model_id=str(pb.model_id),
        last_state_change=0.0,  # wire does not transport this; receiver re-stamps
        suspect_deadline=None,  # similarly, deadlines are recomputed locally
    )


def _load_to_pb(r: LoadReportRecord) -> wire_pb2.LoadReport:
    return wire_pb2.LoadReport(
        shard_id=r.shard_id,
        queue_depth_ema=r.queue_depth_ema,
        ts_unix_ms=r.ts_unix_ms,
    )


def _load_from_pb(pb: wire_pb2.LoadReport) -> LoadReportRecord:
    return LoadReportRecord(
        shard_id=pb.shard_id,
        queue_depth_ema=int(pb.queue_depth_ema),
        ts_unix_ms=int(pb.ts_unix_ms),
    )


def _heat_to_pb(r: HeatReportRecord) -> wire_pb2.ExpertHeatReport:
    pb = wire_pb2.ExpertHeatReport(
        shard_id=r.shard_id,
        ts_unix_ms=r.ts_unix_ms,
    )
    for layer_idx, expert_id, ema in r.entries:
        e = pb.entries.add()
        e.layer_idx = layer_idx
        e.expert_id = expert_id
        e.heat_ema_x100 = ema
    return pb


def _heat_from_pb(pb: wire_pb2.ExpertHeatReport) -> HeatReportRecord:
    return HeatReportRecord(
        shard_id=pb.shard_id,
        entries=tuple(
            (int(e.layer_idx), int(e.expert_id), int(e.heat_ema_x100))
            for e in pb.entries
        ),
        ts_unix_ms=int(pb.ts_unix_ms),
    )


def _ownership_to_pb(r: OwnershipDeltaRecord) -> wire_pb2.OwnershipDelta:
    return wire_pb2.OwnershipDelta(
        shard_id=r.shard_id,
        layer_idx=r.layer_idx,
        expert_id=r.expert_id,
        action=r.action,
        ts_unix_ms=r.ts_unix_ms,
    )


def _ownership_from_pb(pb: wire_pb2.OwnershipDelta) -> OwnershipDeltaRecord:
    return OwnershipDeltaRecord(
        shard_id=pb.shard_id,
        layer_idx=int(pb.layer_idx),
        expert_id=int(pb.expert_id),
        action=int(pb.action),
        ts_unix_ms=int(pb.ts_unix_ms),
    )


def encode_membership_envelope(msg: IncomingMessage) -> bytes:
    env = wire_pb2.Envelope()
    if isinstance(msg, PingMsg):
        env.ping.protocol_version = _PROTOCOL_VERSION
        env.ping.from_shard_id = msg.from_shard_id
        env.ping.from_incarnation = msg.from_incarnation
        env.ping.deltas.extend(_record_to_pb(d) for d in msg.deltas)
        env.ping.loads.extend(_load_to_pb(lr) for lr in msg.loads)
        env.ping.heat.extend(_heat_to_pb(h) for h in msg.heat)
        env.ping.ownership.extend(_ownership_to_pb(o) for o in msg.ownership)
    elif isinstance(msg, AckMsg):
        env.ack.protocol_version = _PROTOCOL_VERSION
        env.ack.from_shard_id = msg.from_shard_id
        env.ack.from_incarnation = msg.from_incarnation
        env.ack.deltas.extend(_record_to_pb(d) for d in msg.deltas)
        env.ack.loads.extend(_load_to_pb(lr) for lr in msg.loads)
        env.ack.heat.extend(_heat_to_pb(h) for h in msg.heat)
        env.ack.ownership.extend(_ownership_to_pb(o) for o in msg.ownership)
    elif isinstance(msg, PingReqMsg):
        env.ping_req.protocol_version = _PROTOCOL_VERSION
        env.ping_req.from_shard_id = msg.from_shard_id
        env.ping_req.target_shard_id = msg.target_shard_id
        env.ping_req.probe_id = msg.probe_id
        env.ping_req.deltas.extend(_record_to_pb(d) for d in msg.deltas)
        env.ping_req.loads.extend(_load_to_pb(lr) for lr in msg.loads)
        env.ping_req.heat.extend(_heat_to_pb(h) for h in msg.heat)
        env.ping_req.ownership.extend(_ownership_to_pb(o) for o in msg.ownership)
    elif isinstance(msg, PingReqAckMsg):
        env.ping_req_ack.protocol_version = _PROTOCOL_VERSION
        env.ping_req_ack.from_shard_id = msg.from_shard_id
        env.ping_req_ack.target_shard_id = msg.target_shard_id
        env.ping_req_ack.probe_id = msg.probe_id
        env.ping_req_ack.success = msg.success
        env.ping_req_ack.deltas.extend(_record_to_pb(d) for d in msg.deltas)
        env.ping_req_ack.loads.extend(_load_to_pb(lr) for lr in msg.loads)
        env.ping_req_ack.heat.extend(_heat_to_pb(h) for h in msg.heat)
        env.ping_req_ack.ownership.extend(_ownership_to_pb(o) for o in msg.ownership)
    elif isinstance(msg, JoinMsg):
        env.join.protocol_version = _PROTOCOL_VERSION
        env.join.self_record.CopyFrom(_record_to_pb(msg.self_record))
    elif isinstance(msg, MembershipDeltaMsg):
        env.membership_delta.protocol_version = _PROTOCOL_VERSION
        env.membership_delta.members.extend(_record_to_pb(d) for d in msg.members)
    else:  # pragma: no cover - exhaustive above
        raise ValueError(f"unsupported membership message type: {type(msg).__name__}")
    return env.SerializeToString()  # type: ignore[no-any-return]


def decode_membership_envelope(raw: bytes) -> IncomingMessage | None:
    env = wire_pb2.Envelope()
    env.ParseFromString(raw)
    which = env.WhichOneof("payload")
    if which == "ping":
        return PingMsg(
            from_shard_id=env.ping.from_shard_id,
            from_incarnation=int(env.ping.from_incarnation),
            deltas=[_record_from_pb(d) for d in env.ping.deltas],
            loads=[_load_from_pb(lr) for lr in env.ping.loads],
            heat=[_heat_from_pb(h) for h in env.ping.heat],
            ownership=[_ownership_from_pb(o) for o in env.ping.ownership],
        )
    if which == "ack":
        return AckMsg(
            from_shard_id=env.ack.from_shard_id,
            from_incarnation=int(env.ack.from_incarnation),
            deltas=[_record_from_pb(d) for d in env.ack.deltas],
            loads=[_load_from_pb(lr) for lr in env.ack.loads],
            heat=[_heat_from_pb(h) for h in env.ack.heat],
            ownership=[_ownership_from_pb(o) for o in env.ack.ownership],
        )
    if which == "ping_req":
        return PingReqMsg(
            from_shard_id=env.ping_req.from_shard_id,
            target_shard_id=env.ping_req.target_shard_id,
            probe_id=env.ping_req.probe_id,
            deltas=[_record_from_pb(d) for d in env.ping_req.deltas],
            loads=[_load_from_pb(lr) for lr in env.ping_req.loads],
            heat=[_heat_from_pb(h) for h in env.ping_req.heat],
            ownership=[_ownership_from_pb(o) for o in env.ping_req.ownership],
        )
    if which == "ping_req_ack":
        return PingReqAckMsg(
            from_shard_id=env.ping_req_ack.from_shard_id,
            target_shard_id=env.ping_req_ack.target_shard_id,
            probe_id=env.ping_req_ack.probe_id,
            success=bool(env.ping_req_ack.success),
            deltas=[_record_from_pb(d) for d in env.ping_req_ack.deltas],
            loads=[_load_from_pb(lr) for lr in env.ping_req_ack.loads],
            heat=[_heat_from_pb(h) for h in env.ping_req_ack.heat],
            ownership=[_ownership_from_pb(o) for o in env.ping_req_ack.ownership],
        )
    if which == "join":
        return JoinMsg(self_record=_record_from_pb(env.join.self_record))
    if which == "membership_delta":
        return MembershipDeltaMsg(
            members=[_record_from_pb(d) for d in env.membership_delta.members]
        )
    return None


__all__ = ["decode_membership_envelope", "encode_membership_envelope"]
