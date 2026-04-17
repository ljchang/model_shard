"""Node picks load_model vs load_model_partial based on ENABLE_PARTIAL_LOAD."""

from __future__ import annotations

import random
import socket

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
    raise RuntimeError("no free port")


@pytest.mark.slow
def test_node_partial_load_active_when_enabled_and_moe_experts_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENABLE_PARTIAL_LOAD", "true")
    monkeypatch.setenv("ENABLE_GOSSIP", "false")

    port = _free_port()
    spec = ShardSpec(
        shard_id="solo",
        address=NodeAddress("127.0.0.1", port),
        start_layer=0,
        end_layer=30,
        moe_experts={15: (0, 3, 6, 9)},
    )
    peer_port = _free_port()
    # ``solo`` is head+tail (start=0, end=30), so the downstream resolver
    # looks for a peer with start_layer=0. A stub with start=0, end=0 (empty
    # range) satisfies the resolver; its address is never actually dialed in
    # this construction-only test.
    peer = ShardSpec(
        shard_id="peer",
        address=NodeAddress("127.0.0.1", peer_port),
        start_layer=0,
        end_layer=0,
    )
    sm = ShardMap({"solo": spec, "peer": peer})
    node = Node(
        shard=spec, shard_map=sm,
        loaded_model=None,   # force internal partial load
        total_layers=30,
    )
    try:
        lm = node._lm
        assert lm.held_ids_per_layer == {15: (0, 3, 6, 9)}
        layer15 = lm.text_model.layers[15]
        assert layer15.experts.switch_glu.gate_proj.weight.shape[0] == 4
    finally:
        node.shutdown()
