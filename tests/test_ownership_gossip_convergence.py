"""3-node cluster: after attach, ownership gossip converges within N rounds."""
from __future__ import annotations

import socket as _sk
import threading
import time

import pytest

from model_shard.node import Node
from model_shard.shard_map import NodeAddress, ShardMap, ShardSpec

pytestmark = pytest.mark.slow


def _free_port() -> int:
    """Return a free TCP port in [1024, 64535] so that udp_port = port+1000
    stays within the valid 0-65535 range (see ShardSpec.udp_port)."""
    for _ in range(100):
        s = _sk.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        if port <= 64535:
            return port
    raise RuntimeError("could not find a port <= 64535 after 100 tries")


@pytest.fixture
def gossip_env(monkeypatch):
    monkeypatch.setenv("ENABLE_PARTIAL_LOAD", "true")
    monkeypatch.setenv("ENABLE_DYNAMIC_MIGRATION", "false")
    monkeypatch.setenv("ENABLE_GOSSIP", "true")


def test_ownership_delta_propagates_within_three_rounds(gossip_env):
    ports = [_free_port() for _ in range(3)]
    specs = [
        ShardSpec(
            shard_id=f"n{i}",
            address=NodeAddress(host="127.0.0.1", port=p),
            start_layer=0, end_layer=30,
            moe_experts={15: (i, 3 + i)},
        )
        for i, p in enumerate(ports)
    ]
    sm = ShardMap({s.shard_id: s for s in specs})
    nodes = [Node(shard=s, shard_map=sm, total_layers=30) for s in specs]
    threads = [threading.Thread(target=n.serve_forever, daemon=True) for n in nodes]
    for t in threads:
        t.start()
    try:
        # Wait for SWIM stabilization.
        time.sleep(2.0)
        # Fake a local attach on n0 for expert 42.
        nodes[0]._live_experts.setdefault(15, set()).add(42)
        with nodes[0]._ownership_seen_lock:
            nodes[0]._ownership_seen.add((nodes[0]._shard.shard_id, 15, 42))
        nodes[0]._membership.announce_ownership_add(15, 42)

        # Wait up to 6s for gossip propagation.
        deadline = time.monotonic() + 6.0
        converged = False
        while time.monotonic() < deadline:
            view_1 = nodes[1]._membership.ownership_view()
            view_2 = nodes[2]._membership.ownership_view()
            if ("n0", 15, 42) in view_1 and ("n0", 15, 42) in view_2:
                converged = True
                break
            time.sleep(0.1)
        assert converged, "ownership ADD did not propagate to all peers"
    finally:
        for n in nodes:
            n.shutdown()
        for t in threads:
            t.join(timeout=3.0)
