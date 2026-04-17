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
from typing import Final

from model_shard.membership.config import SwimConfig
from model_shard.membership.messages import (
    decode_membership_envelope,
    encode_membership_envelope,
)
from model_shard.membership.records import (
    AckMsg,
    IncomingMessage,
    LoadReportRecord,
    PingMsg,
    PingReqAckMsg,
    PingReqMsg,
    StateTransition,
)
from model_shard.membership.state import MembershipState, PeerSpec
from model_shard.membership.transport import UDPTransport

_LOG = logging.getLogger(__name__)


_INTERNAL_QUEUE_MAX: Final[int] = 4096

ObserverCallback = Callable[[StateTransition], None]


class MembershipRunner:
    def __init__(
        self,
        self_spec: PeerSpec,
        peers: list[PeerSpec],
        config: SwimConfig,
        rng_seed: int | None = None,
    ) -> None:
        self._cfg = config
        self._self_spec = self_spec
        self._peers = peers
        self._rng = random.Random(rng_seed)
        self._state = MembershipState(
            self_spec=self_spec,
            peer_specs=peers,
            rng=self._rng,
            config=config,
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

            # Phase 4: piggyback own-load on outgoing ping-family messages.
            if self._load_source is not None:
                try:
                    my_load = self._load_source()
                except Exception:
                    _LOG.exception("load source raised; skipping load piggyback")
                    my_load = None
                if my_load is not None:
                    new_outgoing = []
                    for o in outgoing:
                        p = o.payload
                        if isinstance(p, (PingMsg, AckMsg, PingReqMsg, PingReqAckMsg)):
                            new_payload = dataclasses.replace(
                                p, loads=[*p.loads, my_load]
                            )
                            # OutgoingMessage may be frozen; construct a fresh one.
                            new_outgoing.append(
                                dataclasses.replace(o, payload=new_payload)
                            )
                        else:
                            new_outgoing.append(o)
                    outgoing = new_outgoing

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
