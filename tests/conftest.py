"""Shared fixtures.

Model loading is expensive — we share one instance across the whole session,
and only tests marked `slow` depend on it.
"""

from __future__ import annotations

import os
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
    # Phase 2 derives UDP gossip port = tcp_port + 1000. macOS ephemeral
    # ports can land above 64535, making tcp+1000 overflow 65535. Pick a
    # random free port in 30000-60000 to keep derived ports valid.
    import random

    for _ in range(100):
        port = random.randint(30000, 60000)
        try:
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
                s.bind(("127.0.0.1", port))
            return port
        except OSError:
            continue
    raise RuntimeError("could not obtain a free port in 30000-60000 after 100 tries")


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


@pytest.fixture(scope="session")
def three_node_pipeline_expert_split(
    loaded_model: Any,
) -> Iterator[DistributedCluster]:
    """Session-scoped 3-node pipeline with layer 15's 128 experts split
    round-robin across the three shards (Phase 3).

    Sets ``ENABLE_EXPERT_SHARD=true`` in the environment BEFORE constructing
    any ``Node``, so each node picks up the flag in its ``__init__``. Restores
    the prior value on teardown so other fixtures / sessions are unaffected.

    Uses its own set of free ports distinct from ``three_node_pipeline`` so
    both can coexist at session scope without sharing any node state.
    """
    from model_shard.node import Node
    from model_shard.shard_map import NodeAddress, ShardMap, ShardSpec

    ports = [_find_free_port() for _ in range(3)]

    def _ids_mod3(r: int) -> tuple[int, ...]:
        return tuple(e for e in range(128) if e % 3 == r)

    specs = [
        ShardSpec(
            shard_id="layer_0-10",
            address=NodeAddress("127.0.0.1", ports[0]),
            start_layer=0,
            end_layer=10,
            moe_experts={15: _ids_mod3(0)},
        ),
        ShardSpec(
            shard_id="layer_10-20",
            address=NodeAddress("127.0.0.1", ports[1]),
            start_layer=10,
            end_layer=20,
            moe_experts={15: _ids_mod3(1)},
        ),
        ShardSpec(
            shard_id="layer_20-30",
            address=NodeAddress("127.0.0.1", ports[2]),
            start_layer=20,
            end_layer=30,
            moe_experts={15: _ids_mod3(2)},
        ),
    ]
    shard_map = ShardMap({s.shard_id: s for s in specs})

    # Flip the Phase 3 gate BEFORE constructing any Node (the constructor
    # reads this env var to decide whether to build an ExpertOrchestrator).
    prev_flag = os.environ.get("ENABLE_EXPERT_SHARD")
    os.environ["ENABLE_EXPERT_SHARD"] = "true"
    try:
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
            threading.Thread(target=n.serve_forever, daemon=True)
            for n in nodes.values()
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
    finally:
        # Restore the env var so other fixtures / tests see the default.
        if prev_flag is None:
            os.environ.pop("ENABLE_EXPERT_SHARD", None)
        else:
            os.environ["ENABLE_EXPERT_SHARD"] = prev_flag


@pytest.fixture(scope="module")
def partial_load_fixture():
    """Spin up a single Node with partial load at layer 15 = [0,3,6,9].

    Used by Phase 5b source-side and migration TCP tests."""
    import os
    import socket as _sk
    import threading as _th
    import time as _time

    os.environ["ENABLE_PARTIAL_LOAD"] = "true"
    os.environ["ENABLE_GOSSIP"] = "false"

    from model_shard.node import Node
    from model_shard.shard_map import NodeAddress, ShardMap, ShardSpec

    def _free_port() -> int:
        s = _sk.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    port = _free_port()
    # The "src" shard covers all 30 layers. As the tail node it needs a
    # downstream entry in the ShardMap with start_layer=0 that is not itself
    # (for _resolve_downstream). We add a dummy "src-head" entry pointing at
    # the same address; the node will try to connect to it only when forwarding
    # activations, which does not happen in ExpertWeightRequest handling.
    dummy_port = _free_port()
    spec = ShardSpec(
        shard_id="src",
        address=NodeAddress(host="127.0.0.1", port=port),
        start_layer=0,
        end_layer=30,
        moe_experts={15: (0, 3, 6, 9)},
    )
    dummy_spec = ShardSpec(
        shard_id="src-head",
        address=NodeAddress(host="127.0.0.1", port=dummy_port),
        start_layer=0,
        end_layer=30,
    )
    sm = ShardMap({"src": spec, "src-head": dummy_spec})
    node = Node(shard=spec, shard_map=sm, total_layers=30)
    t = _th.Thread(target=node.serve_forever, daemon=True)
    t.start()
    _time.sleep(0.5)
    try:
        yield (node, port)
    finally:
        node.shutdown()
        t.join(timeout=2.0)
        os.environ.pop("ENABLE_PARTIAL_LOAD", None)
        os.environ.pop("ENABLE_GOSSIP", None)
