"""UDP sidecar for SWIM messages. Independent of the TCP envelope used for
activations to avoid head-of-line blocking. One bound socket per node."""

from __future__ import annotations

import contextlib
import logging
import socket
import threading
from collections.abc import Callable
from typing import Final

_LOG = logging.getLogger(__name__)
_MTU_GUARD: Final[int] = 1400  # safe single-datagram size on most networks
_RECV_TIMEOUT_S: Final[float] = 0.25  # short timeout so stop() responds quickly
_RECV_BUFSIZE: Final[int] = 65535  # max UDP datagram


class UDPTransport:
    def __init__(
        self,
        host: str,
        port: int,
        on_recv: Callable[[bytes, tuple[str, int]], None],
    ) -> None:
        self._host = host
        self._port = port
        self._on_recv = on_recv
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # AF_INET requires an IPv4 address. With Tailscale (or any host
        # that has both IPv4+IPv6 in DNS) bind/sendto can fail with
        # EINVAL because the system resolver returns an IPv6 address
        # first. Force IPv4 resolution via gethostbyname.
        self._sock.bind((socket.gethostbyname(host), port))
        self._sock.settimeout(_RECV_TIMEOUT_S)
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None
        # Cache: hostname -> IPv4 address. Hostnames don't change at
        # runtime; avoid re-resolving on every gossip send.
        self._ipv4_cache: dict[str, str] = {}

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("UDPTransport already started")
        self._thread = threading.Thread(
            target=self._recv_loop, name="udp-recv", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stopping.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        with contextlib.suppress(OSError):
            self._sock.close()

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def send_to(self, address: tuple[str, int], payload: bytes) -> None:
        if len(payload) > _MTU_GUARD:
            _LOG.error(
                "dropping oversize UDP message (%d bytes > MTU=%d) to %s:%d",
                len(payload),
                _MTU_GUARD,
                address[0],
                address[1],
            )
            return
        try:
            host = address[0]
            ipv4 = self._ipv4_cache.get(host)
            if ipv4 is None:
                ipv4 = socket.gethostbyname(host)
                self._ipv4_cache[host] = ipv4
            self._sock.sendto(payload, (ipv4, address[1]))
        except OSError as exc:
            _LOG.warning("UDP sendto %s:%d failed: %s", address[0], address[1], exc)

    def _recv_loop(self) -> None:
        while not self._stopping.is_set():
            try:
                data, addr = self._sock.recvfrom(_RECV_BUFSIZE)
            except TimeoutError:
                continue
            except OSError:
                # socket closed during shutdown
                return
            try:
                self._on_recv(data, addr)
            except Exception:
                _LOG.exception("UDP on_recv callback raised")


__all__ = ["UDPTransport"]
