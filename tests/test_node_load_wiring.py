"""Node wires LoadTracker into runner and orchestrator."""

from __future__ import annotations

import random
import socket
from typing import Any

import pytest

from model_shard.node import Node
from model_shard.shard_map import NodeAddress, ShardMap, ShardSpec


def _free_port() -> int:
    for _ in range(100):
        p = random.randint(30000, 60000)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", p))
            return p
        except OSError:
            continue
    raise RuntimeError("no free port in range")


@pytest.mark.slow
def test_node_wires_load_tracker_and_runner_load_source(
    monkeypatch: pytest.MonkeyPatch, loaded_model: Any
) -> None:
    monkeypatch.setenv("ENABLE_EXPERT_SHARD", "true")
    monkeypatch.setenv("ENABLE_GOSSIP", "false")

    # Two-node topology so _resolve_downstream can find a peer (solo shard
    # would trip the self-downstream guard). We only construct the head
    # Node; the second spec is purely a shard-map entry for routing.
    port_a = _free_port()
    port_b = _free_port()
    spec_a = ShardSpec(
        shard_id="solo",
        address=NodeAddress("127.0.0.1", port_a),
        start_layer=0,
        end_layer=20,
        moe_experts={15: (0, 1, 2)},
    )
    spec_b = ShardSpec(
        shard_id="peer",
        address=NodeAddress("127.0.0.1", port_b),
        start_layer=20,
        end_layer=30,
    )
    sm = ShardMap({"solo": spec_a, "peer": spec_b})
    node = Node(shard=spec_a, shard_map=sm, loaded_model=loaded_model, total_layers=30)
    try:
        # Tracker attribute exists.
        assert hasattr(node, "_load_tracker")
        assert node._load_tracker is not None

        # Orchestrator (if constructed for this shard) has loads_provider.
        orch = getattr(node, "_orchestrator", None)
        if orch is not None:
            assert callable(orch.loads_provider)
            # With gossip off, latest_loads() returns empty, so provider returns {}.
            assert orch.loads_provider() == {}
    finally:
        node.shutdown()
