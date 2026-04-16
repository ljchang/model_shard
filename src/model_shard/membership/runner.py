"""Threaded runner that drives `MembershipState` against a real UDP transport.

One thread owns both the tick loop (every T_TICK ms) and the receive callback
plumbing (the transport invokes `_on_recv` from its own thread, which posts
work onto an internal queue the runner thread drains).

Observer callbacks are invoked from the runner thread, after `state.tick` /
`state.recv` returns — never reentrantly. Exceptions are caught and logged.
"""

from __future__ import annotations

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
from model_shard.membership.records import IncomingMessage, StateTransition
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

    @property
    def state(self) -> MembershipState:
        return self._state

    # ---------------------------------------------------------- transport hook

    def _on_recv(self, data: bytes, _addr: tuple[str, int]) -> None:
        decoded = decode_membership_envelope(data)
        if decoded is None:
            return
        try:
            self._inbox.put_nowait(decoded)
        except queue.Full:
            _LOG.warning("membership inbox full; dropping message %s", type(decoded).__name__)

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
