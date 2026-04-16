"""Pure SWIM state machine — no I/O, no time, no threads.

The runner is responsible for invoking `tick(now)` on a clock and for
delivering received messages to `recv(msg, now)`. Both methods return a list
of `OutgoingMessage` for the runner to send. State transitions are reported
via `changes_since(watermark)` so the runner can fire observer callbacks.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from model_shard.membership.config import SwimConfig
from model_shard.membership.records import (
    AckMsg,
    IncomingMessage,
    MemberRecord,
    MemberState,
    OutgoingMessage,
    PingMsg,
    StateTransition,
)


@dataclass(frozen=True)
class PeerSpec:
    """A peer's static identity. Derived from `shards.yaml` at startup."""

    shard_id: str
    host: str
    udp_port: int


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
        if now < self._next_period_at:
            return out

        # Start of new protocol period. Pick a random ALIVE peer (excluding self).
        candidates = [
            r.shard_id
            for r in self._members.values()
            if r.shard_id != self._self_id and r.state == MemberState.ALIVE
        ]
        if not candidates:
            self._next_period_at = now + self._cfg.t_ping_ms / 1000.0
            return out

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
        out.append(
            OutgoingMessage(
                target_shard_id=target,
                payload=PingMsg(
                    from_shard_id=self._self_id,
                    from_incarnation=self._self_incarnation,
                    deltas=[],
                ),
            )
        )
        self._next_period_at = now + self._cfg.t_ping_ms / 1000.0
        return out

    def recv(self, msg: IncomingMessage, now: float) -> list[OutgoingMessage]:
        if isinstance(msg, PingMsg):
            return self._handle_ping(msg, now)
        if isinstance(msg, AckMsg):
            return self._handle_ack(msg, now)
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
        return []


__all__ = ["MembershipState", "PeerSpec"]
