"""Threaded runner that drives `MembershipState` against a real UDP transport.

One thread owns both the tick loop (every T_TICK ms) and the receive callback
plumbing (the transport invokes `_on_recv` from its own thread, which posts
work onto an internal queue the runner thread drains).

Observer callbacks are invoked from the runner thread, after `state.tick` /
`state.recv` returns — never reentrantly. Exceptions are caught and logged.
"""

from __future__ import annotations

import dataclasses
import logging
import queue
import random
import threading
import time
from collections.abc import Callable
from typing import Any, Final

from model_shard.membership.config import SwimConfig
from model_shard.membership.messages import (
    decode_membership_envelope,
    encode_membership_envelope,
)
from model_shard.membership.records import (
    AckMsg,
    HeatReportRecord,
    IncomingMessage,
    LoadReportRecord,
    OutgoingMessage,
    OwnershipDeltaRecord,
    PingMsg,
    PingReqAckMsg,
    PingReqMsg,
    StateTransition,
)
from model_shard.membership.state import MembershipState, PeerSpec
from model_shard.membership.transport import UDPTransport

_LOG = logging.getLogger(__name__)


_INTERNAL_QUEUE_MAX: Final[int] = 4096
_DEFAULT_OWNERSHIP_TTL: Final[int] = 5


@dataclasses.dataclass
class _OutboundOwnership:
    record: OwnershipDeltaRecord
    ttl: int


ObserverCallback = Callable[[StateTransition], None]


class MembershipRunner:
    def __init__(
        self,
        self_spec: PeerSpec,
        peers: list[PeerSpec],
        config: SwimConfig,
        rng_seed: int | None = None,
        local_model_id: str = "",
    ) -> None:
        self._cfg = config
        self._self_spec = self_spec
        self._peers = peers
        self._rng = random.Random(rng_seed)
        # Seed self_incarnation from the wall clock so a restarted node
        # always announces a higher incarnation than peers' cached records
        # from prior runs. Otherwise an incarnation=0 ALIVE announcement
        # loses LWW to a DEAD record from the previous run.
        import time as _time
        self._state = MembershipState(
            self_spec=self_spec,
            peer_specs=peers,
            rng=self._rng,
            config=config,
            local_model_id=local_model_id,
            initial_incarnation=int(_time.time()),
        )
        self._addr_by_id: dict[str, tuple[str, int]] = {
            self_spec.shard_id: (self_spec.host, self_spec.udp_port)
        }
        for p in peers:
            self._addr_by_id[p.shard_id] = (p.host, p.udp_port)

        self._inbox: queue.Queue[IncomingMessage] = queue.Queue(_INTERNAL_QUEUE_MAX)
        self._observers: list[ObserverCallback] = []
        self._observers_lock = threading.Lock()
        self._watermark = 0
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None

        self._load_source: Callable[[], LoadReportRecord] | None = None
        self._peer_loads: dict[str, LoadReportRecord] = {}
        self._peer_loads_lock = threading.Lock()

        self._heat_source: Callable[[], HeatReportRecord] | None = None
        self._peer_heat: dict[str, HeatReportRecord] = {}
        self._peer_heat_lock = threading.Lock()

        self._outbound_ownership: list[_OutboundOwnership] = []
        self._outbound_ownership_lock = threading.Lock()
        self._ownership_view_internal: dict[
            tuple[str, int, int], tuple[int, int]
        ] = {}
        self._ownership_seen_lock = threading.Lock()

        # Callbacks invoked when ownership deltas arrive from gossip peers.
        # Signature: (shard_id, layer_idx, expert_id, action, ts_unix_ms) -> None
        self._ownership_observers: list[
            Callable[[str, int, int, int, int], None]
        ] = []
        self._ownership_observers_lock = threading.Lock()

        self._transport = UDPTransport(
            host=self_spec.host,
            port=self_spec.udp_port,
            on_recv=self._on_recv,
        )

    # --------------------------------------------------------------- lifecycle

    def start(self) -> None:
        self._transport.start()
        self._thread = threading.Thread(
            target=self._run, name="membership-runner", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stopping.set()
        self._transport.stop()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # --------------------------------------------------------------- public API

    def subscribe(self, callback: ObserverCallback) -> None:
        with self._observers_lock:
            self._observers.append(callback)

    def start_load_source(self, fn: Callable[[], LoadReportRecord]) -> None:
        """Register a callable invoked once per outgoing ping-family message
        to produce this node's own load report. Safe to set multiple times;
        the latest wins."""
        self._load_source = fn

    def latest_loads(self) -> dict[str, LoadReportRecord]:
        """Return a snapshot of the most recent load report seen per peer
        shard_id. Caller is responsible for filtering by staleness."""
        with self._peer_loads_lock:
            return dict(self._peer_loads)

    def start_heat_source(self, fn: Callable[[], HeatReportRecord]) -> None:
        """Register a callable invoked once per outgoing ping-family message
        to produce this node's own heat report. Latest registration wins."""
        self._heat_source = fn

    def latest_heat(self) -> dict[str, HeatReportRecord]:
        """Return a snapshot of the most recent heat report seen per peer."""
        with self._peer_heat_lock:
            return dict(self._peer_heat)

    def register_ownership_observer(
        self, callback: Callable[[str, int, int, int, int], None]
    ) -> None:
        """Register a callback invoked for each OwnershipDeltaRecord received
        from a gossip peer (not for self-announced deltas). Signature:
            callback(shard_id, layer_idx, expert_id, action, ts_unix_ms) -> None
        Called from the transport thread; must be fast and non-blocking."""
        with self._ownership_observers_lock:
            self._ownership_observers.append(callback)

    def announce_ownership_add(
        self, layer_idx: int, expert_id: int, ttl: int = _DEFAULT_OWNERSHIP_TTL
    ) -> None:
        """Enqueue an ADD delta about self to piggyback for the next `ttl`
        outbound ping-family messages. Also folds into ``ownership_view``
        immediately so local readers see self-ownership without waiting."""
        rec = OwnershipDeltaRecord(
            shard_id=self._self_spec.shard_id,
            layer_idx=layer_idx,
            expert_id=expert_id,
            action=0,
            ts_unix_ms=int(time.time() * 1000),
        )
        with self._outbound_ownership_lock:
            self._outbound_ownership.append(_OutboundOwnership(record=rec, ttl=ttl))
        with self._ownership_seen_lock:
            key = (rec.shard_id, rec.layer_idx, rec.expert_id)
            existing = self._ownership_view_internal.get(key)
            if existing is None or rec.ts_unix_ms > existing[1]:
                self._ownership_view_internal[key] = (rec.action, rec.ts_unix_ms)

    def announce_ownership_remove(
        self, layer_idx: int, expert_id: int, ttl: int = _DEFAULT_OWNERSHIP_TTL
    ) -> None:
        """Gossip an OwnershipDelta{action=REMOVE}. Symmetric to
        announce_ownership_add. Updates the local view immediately via
        last-writer-wins on ts_unix_ms."""
        with self._ownership_seen_lock:
            key = (self._self_spec.shard_id, layer_idx, expert_id)
            existing = self._ownership_view_internal.get(key)
            # Ensure the REMOVE timestamp strictly exceeds any existing ADD so
            # that same-millisecond add→remove sequences converge correctly.
            existing_ts = existing[1] if existing is not None else 0
            ts = max(int(time.time() * 1000), existing_ts + 1)
            rec = OwnershipDeltaRecord(
                shard_id=self._self_spec.shard_id,
                layer_idx=layer_idx,
                expert_id=expert_id,
                action=1,
                ts_unix_ms=ts,
            )
            self._ownership_view_internal[key] = (rec.action, rec.ts_unix_ms)
        with self._outbound_ownership_lock:
            self._outbound_ownership.append(_OutboundOwnership(record=rec, ttl=ttl))

    def ownership_view(self) -> set[tuple[str, int, int]]:
        """Snapshot of every (shard_id, layer_idx, expert_id) whose latest
        observed action is ADD. REMOVE entries are excluded; stale ADDs
        superseded by a newer REMOVE are also excluded."""
        with self._ownership_seen_lock:
            return {
                key for key, (action, _) in self._ownership_view_internal.items()
                if action == 0
            }

    def _drain_outbound_ownership(self) -> list[OwnershipDeltaRecord]:
        """Return records to piggyback this round and decrement TTLs; evict
        entries whose TTL reaches zero."""
        with self._outbound_ownership_lock:
            to_send = [o.record for o in self._outbound_ownership]
            surviving: list[_OutboundOwnership] = []
            for o in self._outbound_ownership:
                if o.ttl > 1:
                    surviving.append(_OutboundOwnership(record=o.record, ttl=o.ttl - 1))
            self._outbound_ownership = surviving
        return to_send

    @property
    def state(self) -> MembershipState:
        return self._state

    # ---------------------------------------------------------- transport hook

    def _on_recv(self, data: bytes, _addr: tuple[str, int]) -> None:
        decoded = decode_membership_envelope(data)
        if decoded is None:
            return
        self._on_recv_decoded(decoded)

    def _on_recv_decoded(self, decoded: IncomingMessage) -> None:
        """Scrape any load reports carried on the message and post the
        decoded message onto the runner's inbox."""
        loads = getattr(decoded, "loads", None)
        if loads:
            with self._peer_loads_lock:
                for lr in loads:
                    self._peer_loads[lr.shard_id] = lr
        heat = getattr(decoded, "heat", None)
        if heat:
            with self._peer_heat_lock:
                for hr in heat:
                    self._peer_heat[hr.shard_id] = hr
        ownership = getattr(decoded, "ownership", None)
        if ownership:
            applied: list[OwnershipDeltaRecord] = []
            with self._ownership_seen_lock:
                for od in ownership:
                    key = (od.shard_id, od.layer_idx, od.expert_id)
                    existing = self._ownership_view_internal.get(key)
                    if existing is None or od.ts_unix_ms > existing[1]:
                        self._ownership_view_internal[key] = (od.action, od.ts_unix_ms)
                        applied.append(od)
            if applied:
                with self._ownership_observers_lock:
                    observers = list(self._ownership_observers)
                for od in applied:
                    for cb in observers:
                        try:
                            cb(od.shard_id, od.layer_idx, od.expert_id, od.action, od.ts_unix_ms)
                        except Exception:
                            _LOG.exception("ownership observer raised; suppressing")
        try:
            self._inbox.put_nowait(decoded)
        except queue.Full:
            _LOG.warning(
                "membership inbox full; dropping message %s",
                type(decoded).__name__,
            )

    # ---------------------------------------------------------------- run loop

    def _run(self) -> None:
        tick_period_s = self._cfg.t_tick_ms / 1000.0
        while not self._stopping.is_set():
            now = time.monotonic()
            outgoing = self._state.tick(now)

            # Drain any received messages.
            while True:
                try:
                    msg = self._inbox.get_nowait()
                except queue.Empty:
                    break
                outgoing.extend(self._state.recv(msg, time.monotonic()))

            # Fused piggyback pass: compute all three piggybacks up front, then
            # do a single walk over outgoing — one dataclasses.replace per
            # payload instead of up to three.
            my_load: LoadReportRecord | None = None
            if self._load_source is not None:
                try:
                    my_load = self._load_source()
                except Exception:
                    _LOG.exception("load source raised; skipping load piggyback")

            my_heat: HeatReportRecord | None = None
            if self._heat_source is not None:
                try:
                    my_heat = self._heat_source()
                except Exception:
                    _LOG.exception("heat source raised; skipping heat piggyback")

            # Only drain (and decrement TTL) when there are real outgoing
            # messages to carry the piggyback — otherwise TTL burns without
            # any peer ever receiving the delta (t_tick_ms << t_ping_ms).
            owner_batch = self._drain_outbound_ownership() if outgoing else []

            any_piggyback = (
                my_load is not None or my_heat is not None or bool(owner_batch)
            )
            if any_piggyback:
                fused: list[OutgoingMessage] = []
                for o in outgoing:
                    p = o.payload
                    if isinstance(p, (PingMsg, AckMsg, PingReqMsg, PingReqAckMsg)):
                        replace_kwargs: dict[str, Any] = {}
                        if my_load is not None:
                            replace_kwargs["loads"] = [*p.loads, my_load]
                        if my_heat is not None:
                            replace_kwargs["heat"] = [*p.heat, my_heat]
                        if owner_batch:
                            replace_kwargs["ownership"] = [*p.ownership, *owner_batch]
                        new_payload = dataclasses.replace(p, **replace_kwargs)
                        # OutgoingMessage may be frozen; construct a fresh one.
                        fused.append(dataclasses.replace(o, payload=new_payload))
                    else:
                        fused.append(o)
                outgoing = fused

            for o in outgoing:
                addr = self._addr_by_id.get(o.target_shard_id)
                if addr is None:
                    _LOG.warning(
                        "no address for shard_id %r; dropping message",
                        o.target_shard_id,
                    )
                    continue
                self._transport.send_to(addr, encode_membership_envelope(o.payload))

            # Fire observer callbacks for any transitions since last loop.
            new_transitions = self._state.changes_since(self._watermark)
            self._watermark = self._state.transition_watermark
            if new_transitions:
                self._fire_observers(new_transitions)

            self._stopping.wait(tick_period_s)

    def _fire_observers(self, transitions: list[StateTransition]) -> None:
        with self._observers_lock:
            observers = list(self._observers)
        for t in transitions:
            for cb in observers:
                try:
                    cb(t)
                except Exception:
                    _LOG.exception("observer callback raised; suppressing")


__all__ = ["MembershipRunner", "ObserverCallback"]
