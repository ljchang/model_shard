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
    IncomingMessage,
    MemberRecord,
    MemberState,
    OutgoingMessage,
    StateTransition,
)


@dataclass(frozen=True)
class PeerSpec:
    """A peer's static identity. Derived from `shards.yaml` at startup."""

    shard_id: str
    host: str
    udp_port: int


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
        return []

    def recv(self, msg: IncomingMessage, now: float) -> list[OutgoingMessage]:
        return []


__all__ = ["MembershipState", "PeerSpec"]
