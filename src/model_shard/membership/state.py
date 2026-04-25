"""Pure SWIM state machine — no I/O, no time, no threads.

The runner is responsible for invoking `tick(now)` on a clock and for
delivering received messages to `recv(msg, now)`. Both methods return a list
of `OutgoingMessage` for the runner to send. State transitions are reported
via `changes_since(watermark)` so the runner can fire observer callbacks.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, replace

from model_shard.membership.config import SwimConfig
from model_shard.membership.records import (
    AckMsg,
    IncomingMessage,
    JoinMsg,
    MemberRecord,
    MembershipDeltaMsg,
    MemberState,
    OutgoingMessage,
    PingMsg,
    PingReqAckMsg,
    PingReqMsg,
    StateTransition,
)

_LOG = logging.getLogger(__name__)


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


@dataclass
class _BacklogEntry:
    record: MemberRecord
    priority: int  # number of times this entry has been gossiped
    enqueued_at: float
    seq: int  # insertion-order counter for stable tie-breaking


class MembershipState:
    def __init__(
        self,
        self_spec: PeerSpec,
        peer_specs: list[PeerSpec],
        rng: random.Random,
        config: SwimConfig,
        local_model_id: str = "",
        initial_incarnation: int = 0,
    ) -> None:
        self._self_id = self_spec.shard_id
        self._self_incarnation = initial_incarnation
        self._cfg = config
        self._rng = rng
        self._local_model_id = local_model_id  # Phase 7-C-3b
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
                # All peers in a shards.yaml share the same model_id by
                # definition — it's a cluster-wide invariant. Populating
                # the initial view with our own local_model_id avoids
                # gossip about peers carrying empty model_id and being
                # rejected by _admit() before peers self-announce. When
                # a peer's actual self-record arrives via gossip (with
                # its own model_id), _maybe_apply_peer_delta still
                # validates via _admit, so a misconfigured peer is
                # caught the first time it actually speaks.
                model_id=self._local_model_id,
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
        self._backlog: list[_BacklogEntry] = []
        self._backlog_seq: int = 0

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
        self._gc_backlog(now)

        # 1. Resolve any pending help requests that have timed out.
        out.extend(self._maybe_timeout_helps(now))

        # 2. Promote suspects to dead if deadline has passed.
        out.extend(self._maybe_promote_dead(now))

        # 3. Start a new protocol period if it's time. This must run before
        #    escalation so that a new period resets the probe — preventing the
        #    escalation of the just-started probe in the same tick.
        if now >= self._next_period_at:
            out.extend(self._start_protocol_period(now))

        # 4. Escalate pending probe to indirect ping-req if ack overdue.
        out.extend(self._maybe_escalate_probe(now))

        return out

    def _enqueue_backlog(self, rec: MemberRecord, now: float) -> None:
        # Replace any existing entry for this shard_id with the latest record;
        # priority resets to 0 so the new state propagates. The seq counter
        # preserves insertion order as a stable sort tiebreaker.
        self._backlog = [b for b in self._backlog if b.record.shard_id != rec.shard_id]
        self._backlog.append(
            _BacklogEntry(record=rec, priority=0, enqueued_at=now, seq=self._backlog_seq)
        )
        self._backlog_seq += 1

    def _gc_backlog(self, now: float) -> None:
        cutoff = 3 * self._cfg.t_suspect_ms / 1000.0
        self._backlog = [b for b in self._backlog if (now - b.enqueued_at) <= cutoff]

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
                        deltas=self._select_outgoing_deltas(),
                    ),
                )
            )
        self._pending_probe = replace(
            probe, indirect_sent_at=now, indirect_targets=chosen
        )
        # If no indirect helpers are available, we cannot wait for PingReqAcks
        # that will never arrive — immediately promote the target to SUSPECT.
        if not chosen:
            self._mark_suspect(probe.target_id, now)
            self._pending_probe = None
        return out

    def _start_protocol_period(self, now: float) -> list[OutgoingMessage]:
        # If the previous probe was in the indirect phase and never got a
        # successful PingReqAck, the period expired without confirmation —
        # declare the target suspect before starting a fresh period.
        prev = self._pending_probe
        if (
            prev is not None
            and prev.indirect_sent_at is not None
            and not prev.indirect_success_seen
        ):
            self._mark_suspect(prev.target_id, now)
        self._pending_probe = None

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
                    deltas=self._select_outgoing_deltas(),
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
        if isinstance(msg, JoinMsg):
            return self._handle_join(msg, now)
        if isinstance(msg, MembershipDeltaMsg):
            return self._handle_delta(msg, now)
        return []

    def _admit(self, record: MemberRecord) -> bool:
        """Phase 7-C-3b cluster admission contract.

        Reject peers whose model_id doesn't match the local node's. The
        "if local is empty, accept any peer" branch is intentional
        permissiveness during rolling upgrade — once production is fully
        on Phase 7-C-3b, every node sets model_id and there's no
        permissive case."""
        # Rolling-upgrade fallback — see docstring; do NOT remove until
        # all clusters are on 7-C-3b+.
        if not self._local_model_id:
            return True
        if record.model_id != self._local_model_id:
            _LOG.warning(
                "rejecting peer %s with model_id mismatch: "
                "local=%r peer=%r",
                record.shard_id, self._local_model_id, record.model_id,
            )
            return False
        return True

    def _handle_join(self, msg: JoinMsg, now: float) -> list[OutgoingMessage]:
        rec = msg.self_record
        if not self._admit(rec):
            # Rejected — don't install, don't echo back. JoinMsg is fire-
            # and-forget (no acks, no retransmit); the newcomer sees no
            # response and times out. This is fail-closed by design.
            return []
        prev = self._members.get(rec.shard_id)
        installed = MemberRecord(
            shard_id=rec.shard_id,
            host=rec.host,
            udp_port=rec.udp_port,
            state=MemberState.ALIVE,
            incarnation=rec.incarnation,
            model_id=rec.model_id,
            last_state_change=now,
            suspect_deadline=None,
        )
        # If we previously recorded the newcomer as DEAD at higher incarnation,
        # echo *that* record back so the newcomer applies the floor rule.
        if prev is not None and prev.state == MemberState.DEAD and prev.incarnation > rec.incarnation:
            members = list(self._members.values())
        else:
            self._members[rec.shard_id] = installed
            self._transitions.append(
                StateTransition(
                    shard_id=rec.shard_id,
                    old_state=(prev.state if prev else None),
                    new_record=installed,
                )
            )
            self._enqueue_backlog(installed, now)
            members = list(self._members.values())
        return [
            OutgoingMessage(
                target_shard_id=rec.shard_id,
                payload=MembershipDeltaMsg(members=members),
            )
        ]

    def _handle_delta(
        self, msg: MembershipDeltaMsg, now: float
    ) -> list[OutgoingMessage]:
        # Bulk install of records (used by joining nodes after Join).
        for rec in msg.members:
            if rec.shard_id == self._self_id:
                self._maybe_refute(rec, now)
                continue
            prev = self._members.get(rec.shard_id)
            if prev is None:
                if not self._admit(rec):
                    continue
                self._members[rec.shard_id] = rec
                self._transitions.append(
                    StateTransition(
                        shard_id=rec.shard_id,
                        old_state=None,
                        new_record=rec,
                    )
                )
                self._enqueue_backlog(rec, now)
                continue
            self._maybe_apply_peer_delta(rec, now)
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
            model_id=prev.model_id,
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
        self._enqueue_backlog(new, now)

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
                    model_id=rec.model_id,
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
                self._enqueue_backlog(new, now)
        return []

    def _handle_ping(self, msg: PingMsg, now: float) -> list[OutgoingMessage]:
        if msg.from_shard_id not in self._members:
            return []
        self._apply_deltas(msg.deltas, now)
        return [
            OutgoingMessage(
                target_shard_id=msg.from_shard_id,
                payload=AckMsg(
                    from_shard_id=self._self_id,
                    from_incarnation=self._self_incarnation,
                    deltas=self._select_outgoing_deltas(),
                ),
            )
        ]

    def _apply_deltas(self, deltas: list[MemberRecord], now: float) -> None:
        for d in deltas:
            if d.shard_id == self._self_id:
                self._maybe_refute(d, now)
                continue
            if d.shard_id not in self._members:
                _LOG.warning(
                    "dropping gossip about unknown shard_id %r (not in shards.yaml)",
                    d.shard_id,
                )
                continue
            self._maybe_apply_peer_delta(d, now)

    def _maybe_refute(self, delta_about_self: MemberRecord, now: float) -> None:
        # If gossip claims we are suspect/dead, refute by lifting our
        # incarnation above whatever the gossip asserts.
        if delta_about_self.state == MemberState.ALIVE:
            return
        if delta_about_self.incarnation < self._self_incarnation:
            return  # stale gossip
        self._self_incarnation = delta_about_self.incarnation + 1
        prev = self._members[self._self_id]
        new = MemberRecord(
            shard_id=self._self_id,
            host=prev.host,
            udp_port=prev.udp_port,
            state=MemberState.ALIVE,
            incarnation=self._self_incarnation,
            model_id=prev.model_id,
            last_state_change=now,
            suspect_deadline=None,
        )
        self._members[self._self_id] = new
        self._transitions.append(
            StateTransition(
                shard_id=self._self_id,
                old_state=prev.state,
                new_record=new,
            )
        )
        self._enqueue_backlog(new, now)

    def _maybe_apply_peer_delta(self, d: MemberRecord, now: float) -> None:
        if not self._admit(d):
            return
        prev = self._members[d.shard_id]
        if d.incarnation < prev.incarnation:
            return
        if d.incarnation == prev.incarnation and d.state <= prev.state:
            # Same incarnation: only apply if the new state is *more severe*
            # (DEAD > SUSPECT > ALIVE; the IntEnum ordering encodes this).
            return
        new = MemberRecord(
            shard_id=d.shard_id,
            host=prev.host,
            udp_port=prev.udp_port,
            state=d.state,
            incarnation=d.incarnation,
            model_id=d.model_id,
            last_state_change=now,
            suspect_deadline=(
                now + self._cfg.t_suspect_ms / 1000.0
                if d.state == MemberState.SUSPECT
                else None
            ),
        )
        self._members[d.shard_id] = new
        self._transitions.append(
            StateTransition(
                shard_id=d.shard_id,
                old_state=prev.state,
                new_record=new,
            )
        )
        self._enqueue_backlog(new, now)

    def _select_outgoing_deltas(self) -> list[MemberRecord]:
        # Always include our own current record so refutations and incarnation
        # bumps propagate immediately. Then up to K_GOSSIP backlog entries
        # ordered by ascending priority (oldest gossiped first).
        deltas: list[MemberRecord] = [self._members[self._self_id]]
        self._backlog.sort(key=lambda b: (b.priority, b.seq))
        for entry in self._backlog[: self._cfg.k_gossip]:
            deltas.append(entry.record)
            entry.priority += 1
        return deltas

    def _handle_ack(self, msg: AckMsg, now: float) -> list[OutgoingMessage]:
        probe = self._pending_probe
        if probe is not None and probe.target_id == msg.from_shard_id:
            self._pending_probe = None

        # Apply piggybacked gossip deltas, just as we do on Ping.  This is
        # critical for dead-node rejoin: when a restarted node sends a Ping and
        # receives an Ack that carries its own DEAD record, `_apply_deltas`
        # calls `_maybe_refute`, which bumps the node's incarnation above the
        # DEAD epoch and propagates ALIVE at the new incarnation.
        self._apply_deltas(msg.deltas, now)

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
                            deltas=self._select_outgoing_deltas(),
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
                    deltas=self._select_outgoing_deltas(),
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
                            deltas=self._select_outgoing_deltas(),
                        ),
                    )
                )
            else:
                remaining.append(h)
        self._pending_helps = remaining
        return out


__all__ = ["MembershipState", "PeerSpec"]
