import socket
import threading
import time
from typing import Any

from model_shard.membership.transport import UDPTransport


def _free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_send_and_receive_roundtrip() -> None:
    port_a = _free_udp_port()
    port_b = _free_udp_port()
    received: list[tuple[bytes, tuple[str, int]]] = []
    done = threading.Event()

    def on_recv(data: bytes, addr: tuple[str, int]) -> None:
        received.append((data, addr))
        done.set()

    a = UDPTransport(host="127.0.0.1", port=port_a, on_recv=on_recv)
    b = UDPTransport(host="127.0.0.1", port=port_b, on_recv=lambda *_: None)
    a.start()
    b.start()
    try:
        b.send_to(("127.0.0.1", port_a), b"hello")
        assert done.wait(timeout=1.0)
        assert received[0][0] == b"hello"
        assert received[0][1][1] == port_b
    finally:
        a.stop()
        b.stop()


def test_send_oversize_message_is_dropped(caplog: Any) -> None:
    import logging
    port = _free_udp_port()
    t = UDPTransport(host="127.0.0.1", port=port, on_recv=lambda *_: None)
    t.start()
    try:
        with caplog.at_level(logging.ERROR, logger="model_shard.membership.transport"):
            t.send_to(("127.0.0.1", port + 1), b"x" * 2000)  # > 1400 MTU
        assert any("MTU" in r.message for r in caplog.records)
    finally:
        t.stop()


def test_stop_unblocks_recv_loop() -> None:
    port = _free_udp_port()
    t = UDPTransport(host="127.0.0.1", port=port, on_recv=lambda *_: None)
    t.start()
    t.stop()
    # If stop() doesn't unblock, the thread is still alive after a moment.
    time.sleep(0.5)
    assert not t.is_alive()
