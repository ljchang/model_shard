"""Sequential seed contact for membership bootstrap.

Reads peer addresses from `shards.yaml` (via the `peers` argument), filters
self out, and tries each in YAML order. The first seed that returns a
`MembershipDelta` reply within the timeout wins; the joining node installs
that view as its initial state and starts the tick loop.

If every seed times out, returns `BootstrapResult(success=False, members=[])`
so the caller can install a single-node view and start anyway.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from model_shard.membership.messages import (
    decode_membership_envelope,
    encode_membership_envelope,
)
from model_shard.membership.records import (
    JoinMsg,
    MemberRecord,
    MembershipDeltaMsg,
    MemberState,
)
from model_shard.membership.state import PeerSpec

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class BootstrapResult:
    success: bool
    members: list[MemberRecord]


# A request function takes (addr, payload_bytes, timeout) and returns the
# response bytes, or None on timeout. Decoupled so tests can stub it.
RequestFn = Callable[[tuple[str, int], bytes, float], bytes | None]


def bootstrap_sequential(
    self_spec: PeerSpec,
    peers: list[PeerSpec],
    request_fn: RequestFn,
    timeout_s: float = 0.5,
) -> BootstrapResult:
    self_record = MemberRecord(
        shard_id=self_spec.shard_id,
        host=self_spec.host,
        udp_port=self_spec.udp_port,
        state=MemberState.ALIVE,
        incarnation=0,
        model_id="",
        last_state_change=0.0,
        suspect_deadline=None,
    )
    join_payload = encode_membership_envelope(JoinMsg(self_record=self_record))

    for peer in peers:
        if peer.shard_id == self_spec.shard_id:
            continue
        addr = (peer.host, peer.udp_port)
        _LOG.info("bootstrap: contacting seed %s at %s:%d", peer.shard_id, *addr)
        reply = request_fn(addr, join_payload, timeout_s)
        if reply is None:
            _LOG.info("bootstrap: seed %s timed out", peer.shard_id)
            continue
        decoded = decode_membership_envelope(reply)
        if isinstance(decoded, MembershipDeltaMsg):
            _LOG.info(
                "bootstrap: seed %s replied with %d members",
                peer.shard_id,
                len(decoded.members),
            )
            return BootstrapResult(success=True, members=list(decoded.members))
        _LOG.warning(
            "bootstrap: seed %s replied with unexpected message %r",
            peer.shard_id,
            type(decoded).__name__ if decoded else "None",
        )

    _LOG.warning("bootstrap: all seeds failed; starting in single-node view")
    return BootstrapResult(success=False, members=[])


__all__ = ["BootstrapResult", "RequestFn", "bootstrap_sequential"]
