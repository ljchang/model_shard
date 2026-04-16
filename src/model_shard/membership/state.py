"""Pure SWIM state machine — no I/O, no time, no threads.

The runner is responsible for invoking `tick(now)` on a clock and for
delivering received messages to `recv(msg, now)`. Both methods return a list
of `OutgoingMessage` for the runner to send. State transitions are reported
via `changes_since(watermark)` so the runner can fire observer callbacks.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

from model_shard.membership.config import SwimConfig
from model_shard.membership.records import (
    AckMsg,
    IncomingMessage,
    MemberRecord,
    MemberState,
    OutgoingMessage,
    PingMsg,
    PingReqAckMsg,
    PingReqMsg,
    StateTransition,
)


@dataclass(frozen=True)
class PeerSpec:
    """A peer's static identity. Derived from `shards.yaml` at startup."""

    shard_id: str
    host: str
    udp_port: int


@dataclass(frozen=True)
class _PendingHelp:
    probe_id: str
    target_id: str
    requester_id: str
    sent_at: float


@dataclass(frozen=True)
class _PendingProbe:
    probe_id: str
    target_id: str
    sent_at: float
    indirect_sent_at: float | None  # set when escalated to ping-req
    indirect_targets: tuple[str, ...]  # peers contacted via ping-req
    indirect_acks: int  # count of PingReqAck (success or failure) received
    indirect_success_seen: bool  # any positive PingReqAck received?


class MembershipState:
    def __init__(
        self,
        self_spec: PeerSpec,
        peer_specs: list[PeerSpec],
        rng: random.Random,
        config: SwimConfig,
    ) -> None:
        self._self_id = self_spec.shard_id
        self._self_incarnation = 0
        self._cfg = config
        self._rng = rng
        self._addrs: dict[str, PeerSpec] = {self_spec.shard_id: self_spec}
        for p in peer_specs:
            self._addrs[p.shard_id] = p
        self._members: dict[str, MemberRecord] = {}
        for p in [self_spec, *peer_specs]:
            self._members[p.shard_id] = MemberRecord(
                shard_id=p.shard_id,
                host=p.host,
                udp_port=p.udp_port,
                state=MemberState.ALIVE,
                incarnation=0,
                last_state_change=0.0,
                suspect_deadline=None,
            )
        self._transitions: list[StateTransition] = []

        # Protocol-period state. Each period: pick a peer, ping, await ack,
        # escalate to indirect probe if no ack, finally suspect on no positive
        # PingReqAck.
        self._next_period_at: float = float(self._cfg.t_ping_ms) / 1000.0
        self._pending_probe: _PendingProbe | None = None
        self._probe_counter: int = 0
        self._pending_helps: list[_PendingHelp] = []

    # ---------------------------------------------------------- read-only API

    def view(self) -> dict[str, MemberRecord]:
        """Return a snapshot copy of every known member's record."""
        return dict(self._members)

    def changes_since(self, watermark: int) -> list[StateTransition]:
        """Return transitions appended after the given watermark index."""
        return list(self._transitions[watermark:])

    @property
    def transition_watermark(self) -> int:
        """Current length of the transitions log; pass to `changes_since`."""
        return len(self._transitions)

    # --------------------------------------------------------- mutation hooks
    # Filled in by later tasks. tick/recv/local_event are listed here as stubs
    # so type-checkers and importers see the expected surface from Task 4 on.

    def tick(self, now: float) -> list[OutgoingMessage]:
        out: list[OutgoingMessage] = []

        # 1. Resolve any pending help requests that have timed out.
        out.extend(self._maybe_timeout_helps(now))

        # 2. Promote suspects to dead if deadline has passed.
        out.extend(self._maybe_promote_dead(now))

        # 3. Escalate pending probe to indirect ping-req if ack overdue.
        out.extend(self._maybe_escalate_probe(now))

        # 4. Start a new protocol period if it's time.
        if now >= self._next_period_at:
            out.extend(self._start_protocol_period(now))

        return out

    def _maybe_escalate_probe(self, now: float) -> list[OutgoingMessage]:
        probe = self._pending_probe
        if probe is None or probe.indirect_sent_at is not None:
            return []
        if now < probe.sent_at + self._cfg.t_timeout_ms / 1000.0:
            return []

        # Pick K_INDIRECT random alive peers, excluding self and the target.
        candidates = [
            r.shard_id
            for r in self._members.values()
            if r.shard_id not in (self._self_id, probe.target_id)
            and r.state == MemberState.ALIVE
        ]
        self._rng.shuffle(candidates)
        chosen = tuple(candidates[: self._cfg.k_indirect])
        out: list[OutgoingMessage] = []
        for helper in chosen:
            out.append(
                OutgoingMessage(
                    target_shard_id=helper,
                    payload=PingReqMsg(
                        from_shard_id=self._self_id,
                        target_shard_id=probe.target_id,
                        probe_id=probe.probe_id,
                        deltas=[],
                    ),
                )
            )
        self._pending_probe = replace(
            probe, indirect_sent_at=now, indirect_targets=chosen
        )
        return out

    def _start_protocol_period(self, now: float) -> list[OutgoingMessage]:
        candidates = [
            r.shard_id
            for r in self._members.values()
            if r.shard_id != self._self_id and r.state == MemberState.ALIVE
        ]
        if not candidates:
            self._next_period_at = now + self._cfg.t_ping_ms / 1000.0
            return []

        target = self._rng.choice(candidates)
        self._probe_counter += 1
        probe_id = f"{self._self_id}:{self._probe_counter}"
        self._pending_probe = _PendingProbe(
            probe_id=probe_id,
            target_id=target,
            sent_at=now,
            indirect_sent_at=None,
            indirect_targets=(),
            indirect_acks=0,
            indirect_success_seen=False,
        )
        self._next_period_at = now + self._cfg.t_ping_ms / 1000.0
        return [
            OutgoingMessage(
                target_shard_id=target,
                payload=PingMsg(
                    from_shard_id=self._self_id,
                    from_incarnation=self._self_incarnation,
                    deltas=[],
                ),
            )
        ]

    def recv(self, msg: IncomingMessage, now: float) -> list[OutgoingMessage]:
        if isinstance(msg, PingMsg):
            return self._handle_ping(msg, now)
        if isinstance(msg, AckMsg):
            return self._handle_ack(msg, now)
        if isinstance(msg, PingReqMsg):
            return self._handle_pingreq(msg, now)
        if isinstance(msg, PingReqAckMsg):
            return self._handle_pingreqack(msg, now)
        return []

    def _handle_pingreqack(
        self, msg: PingReqAckMsg, now: float
    ) -> list[OutgoingMessage]:
        probe = self._pending_probe
        if probe is None or probe.probe_id != msg.probe_id:
            return []

        if msg.success:
            self._pending_probe = None
            return []

        new_acks = probe.indirect_acks + 1
        new_success = probe.indirect_success_seen or msg.success
        if (
            new_acks >= len(probe.indirect_targets)
            and not new_success
        ):
            self._mark_suspect(probe.target_id, now)
            self._pending_probe = None
        else:
            self._pending_probe = replace(
                probe,
                indirect_acks=new_acks,
                indirect_success_seen=new_success,
            )
        return []

    def _mark_suspect(self, shard_id: str, now: float) -> None:
        prev = self._members[shard_id]
        if prev.state in (MemberState.SUSPECT, MemberState.DEAD):
            return
        new = MemberRecord(
            shard_id=shard_id,
            host=prev.host,
            udp_port=prev.udp_port,
            state=MemberState.SUSPECT,
            incarnation=prev.incarnation,
            last_state_change=now,
            suspect_deadline=now + self._cfg.t_suspect_ms / 1000.0,
        )
        self._members[shard_id] = new
        self._transitions.append(
            StateTransition(
                shard_id=shard_id,
                old_state=prev.state,
                new_record=new,
            )
        )

    def _maybe_promote_dead(self, now: float) -> list[OutgoingMessage]:
        for shard_id, rec in list(self._members.items()):
            if (
                rec.state == MemberState.SUSPECT
                and rec.suspect_deadline is not None
                and now >= rec.suspect_deadline
            ):
                new = MemberRecord(
                    shard_id=shard_id,
                    host=rec.host,
                    udp_port=rec.udp_port,
                    state=MemberState.DEAD,
                    incarnation=rec.incarnation,
                    last_state_change=now,
                    suspect_deadline=None,
                )
                self._members[shard_id] = new
                self._transitions.append(
                    StateTransition(
                        shard_id=shard_id,
                        old_state=MemberState.SUSPECT,
                        new_record=new,
                    )
                )
        return []

    def _handle_ping(self, msg: PingMsg, now: float) -> list[OutgoingMessage]:
        if msg.from_shard_id not in self._members:
            return []
        return [
            OutgoingMessage(
                target_shard_id=msg.from_shard_id,
                payload=AckMsg(
                    from_shard_id=self._self_id,
                    from_incarnation=self._self_incarnation,
                    deltas=[],
                ),
            )
        ]

    def _handle_ack(self, msg: AckMsg, now: float) -> list[OutgoingMessage]:
        probe = self._pending_probe
        if probe is not None and probe.target_id == msg.from_shard_id:
            self._pending_probe = None

        out: list[OutgoingMessage] = []
        remaining: list[_PendingHelp] = []
        for h in self._pending_helps:
            if h.target_id == msg.from_shard_id:
                out.append(
                    OutgoingMessage(
                        target_shard_id=h.requester_id,
                        payload=PingReqAckMsg(
                            from_shard_id=self._self_id,
                            target_shard_id=h.target_id,
                            probe_id=h.probe_id,
                            success=True,
                            deltas=[],
                        ),
                    )
                )
            else:
                remaining.append(h)
        self._pending_helps = remaining
        return out

    def _handle_pingreq(
        self, msg: PingReqMsg, now: float
    ) -> list[OutgoingMessage]:
        if msg.target_shard_id not in self._members:
            return []
        self._pending_helps.append(
            _PendingHelp(
                probe_id=msg.probe_id,
                target_id=msg.target_shard_id,
                requester_id=msg.from_shard_id,
                sent_at=now,
            )
        )
        return [
            OutgoingMessage(
                target_shard_id=msg.target_shard_id,
                payload=PingMsg(
                    from_shard_id=self._self_id,
                    from_incarnation=self._self_incarnation,
                    deltas=[],
                ),
            )
        ]

    def _maybe_timeout_helps(self, now: float) -> list[OutgoingMessage]:
        out: list[OutgoingMessage] = []
        remaining: list[_PendingHelp] = []
        for h in self._pending_helps:
            if now >= h.sent_at + self._cfg.t_timeout_ms / 1000.0:
                out.append(
                    OutgoingMessage(
                        target_shard_id=h.requester_id,
                        payload=PingReqAckMsg(
                            from_shard_id=self._self_id,
                            target_shard_id=h.target_id,
                            probe_id=h.probe_id,
                            success=False,
                            deltas=[],
                        ),
                    )
                )
            else:
                remaining.append(h)
        self._pending_helps = remaining
        return out


__all__ = ["MembershipState", "PeerSpec"]
