"""Pure data types for the SWIM membership layer.

Every type here is frozen and free of I/O imports. The state machine
(`state.py`) operates exclusively on these types; conversion to/from
protobuf lives in `messages.py`.
"""

from dataclasses import dataclass
from enum import IntEnum


class MemberState(IntEnum):
    """Membership states. Ordering is meaningful: severity ALIVE < SUSPECT < DEAD.

    Used by the same-incarnation tiebreaker in `state.py`. Numeric values
    must match the `state` field of wire `MemberRecordPb`.
    """

    ALIVE = 0
    SUSPECT = 1
    DEAD = 2


@dataclass(frozen=True)
class MemberRecord:
    shard_id: str
    host: str
    udp_port: int
    state: MemberState
    incarnation: int
    last_state_change: float
    suspect_deadline: float | None  # set iff state == SUSPECT


@dataclass(frozen=True)
class StateTransition:
    """Emitted from `MembershipState` whenever a member's recorded state changes.

    `new_record` is the post-transition record; `old_state` is the prior state
    (or None if this is a brand-new member entering the view).
    """

    shard_id: str
    old_state: MemberState | None
    new_record: "MemberRecord"


# ----- Wire-level message dataclasses (mirror the protobuf shapes) -----------


@dataclass(frozen=True)
class PingMsg:
    from_shard_id: str
    from_incarnation: int
    deltas: list[MemberRecord]


@dataclass(frozen=True)
class AckMsg:
    from_shard_id: str
    from_incarnation: int
    deltas: list[MemberRecord]


@dataclass(frozen=True)
class PingReqMsg:
    from_shard_id: str
    target_shard_id: str
    probe_id: str
    deltas: list[MemberRecord]


@dataclass(frozen=True)
class PingReqAckMsg:
    from_shard_id: str
    target_shard_id: str
    probe_id: str
    success: bool
    deltas: list[MemberRecord]


@dataclass(frozen=True)
class JoinMsg:
    self_record: MemberRecord


@dataclass(frozen=True)
class MembershipDeltaMsg:
    members: list[MemberRecord]


IncomingMessage = PingMsg | AckMsg | PingReqMsg | PingReqAckMsg | JoinMsg | MembershipDeltaMsg
"""A union of every membership message that `MembershipState.recv` accepts."""


@dataclass(frozen=True)
class OutgoingMessage:
    """A message produced by the state machine, ready to be UDP-sent.

    `target` is the wire address the runner should `sendto`; `payload` is one
    of the message dataclasses above (the runner serialises via messages.py).
    """

    target_shard_id: str
    payload: IncomingMessage  # same set of types; outgoing == incoming


__all__ = [
    "AckMsg",
    "IncomingMessage",
    "JoinMsg",
    "MemberRecord",
    "MemberState",
    "MembershipDeltaMsg",
    "OutgoingMessage",
    "PingMsg",
    "PingReqAckMsg",
    "PingReqMsg",
    "StateTransition",
]
