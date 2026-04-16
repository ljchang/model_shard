"""Shared fixtures.

Model loading is expensive — we share one instance across the whole session,
and only tests marked `slow` depend on it.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass
from typing import Any

import pytest


@pytest.fixture(scope="session")
def loaded_model() -> Any:
    """Loads Gemma 4 26B A4B (4-bit) once per test session."""
    from model_shard.mlx_engine import load_model

    return load_model("mlx-community/gemma-4-26b-a4b-it-4bit")


def _find_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_for_listening(host: str, port: int, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with closing(socket.create_connection((host, port), timeout=0.5)):
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError(f"node at {host}:{port} never came up")


@dataclass
class DistributedCluster:
    """Handle for an in-process 3-node cluster running in daemon threads."""

    shard_map: Any            # model_shard.shard_map.ShardMap
    nodes_by_id: dict[str, Any]  # shard_id -> model_shard.node.Node


@pytest.fixture(scope="session")
def three_node_pipeline(loaded_model: Any) -> Iterator[DistributedCluster]:
    """Session-scoped 3-node decentralized pipeline. Nodes know about each
    other via a shared ShardMap and forward activations peer-to-peer.

    Yields a DistributedCluster; tests that need per-node state (e.g., Tier 2
    reading debug captures) reach into ``nodes_by_id``.
    """
    from model_shard.node import Node
    from model_shard.shard_map import NodeAddress, ShardMap, ShardSpec

    ports = [_find_free_port() for _ in range(3)]
    specs = [
        ShardSpec(
            shard_id="layer_0-10",
            address=NodeAddress("127.0.0.1", ports[0]),
            start_layer=0,
            end_layer=10,
        ),
        ShardSpec(
            shard_id="layer_10-20",
            address=NodeAddress("127.0.0.1", ports[1]),
            start_layer=10,
            end_layer=20,
        ),
        ShardSpec(
            shard_id="layer_20-30",
            address=NodeAddress("127.0.0.1", ports[2]),
            start_layer=20,
            end_layer=30,
        ),
    ]
    shard_map = ShardMap({s.shard_id: s for s in specs})

    nodes = {
        spec.shard_id: Node(
            shard=spec,
            shard_map=shard_map,
            loaded_model=loaded_model,
            total_layers=loaded_model.num_layers,
        )
        for spec in specs
    }
    threads = [
        threading.Thread(target=n.serve_forever, daemon=True) for n in nodes.values()
    ]
    for t in threads:
        t.start()
    for spec in specs:
        _wait_for_listening(spec.address.host, spec.address.port)

    try:
        yield DistributedCluster(shard_map=shard_map, nodes_by_id=nodes)
    finally:
        for n in nodes.values():
            n.shutdown()
        for t in threads:
            t.join(timeout=2.0)
