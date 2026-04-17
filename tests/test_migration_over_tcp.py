"""End-to-end target-pull migration between two in-process Nodes."""
from __future__ import annotations

import socket as _sk
import threading
import time

import mlx.core as mx
import pytest

from model_shard.moe import run_selected_experts
from model_shard.node import Node
from model_shard.shard_map import NodeAddress, ShardMap, ShardSpec

pytestmark = pytest.mark.slow


def _free_port() -> int:
    s = _sk.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def migration_env(monkeypatch):
    monkeypatch.setenv("ENABLE_PARTIAL_LOAD", "true")
    monkeypatch.setenv("ENABLE_DYNAMIC_MIGRATION", "false")
    monkeypatch.setenv("ENABLE_GOSSIP", "false")


def test_pull_over_tcp_matches_bit_exact(migration_env):
    port_a = _free_port()
    port_b = _free_port()

    spec_a = ShardSpec(
        shard_id="A", address=NodeAddress(host="127.0.0.1", port=port_a),
        start_layer=0, end_layer=30, moe_experts={15: (0, 3, 6, 9)},
    )
    spec_b = ShardSpec(
        shard_id="B", address=NodeAddress(host="127.0.0.1", port=port_b),
        start_layer=0, end_layer=30, moe_experts={15: (1, 4, 7, 10)},
    )
    sm = ShardMap({"A": spec_a, "B": spec_b})

    node_a = Node(shard=spec_a, shard_map=sm, total_layers=30)
    node_b = Node(shard=spec_b, shard_map=sm, total_layers=30)
    t_a = threading.Thread(target=node_a.serve_forever, daemon=True)
    t_b = threading.Thread(target=node_b.serve_forever, daemon=True)
    t_a.start()
    t_b.start()
    time.sleep(0.5)

    try:
        from model_shard.migration import ExpertWeightPeerRPC
        rpc = ExpertWeightPeerRPC(
            addresses={"A": ("127.0.0.1", port_a)}, timeout_s=60.0
        )
        tensors = rpc.pull(source_shard_id="A", layer_idx=15, expert_id=3)
        node_b.migration_attach(layer_idx=15, expert_id=3, tensors=tensors)
        assert 3 in node_b._live_experts[15]

        # Verify bit-exact post-attach.
        hidden = node_a._lm.text_model.layers[15].pre_feedforward_layernorm_2.weight.shape[0]
        mx.random.seed(7)
        h = mx.random.normal((1, 7, hidden)).astype(mx.bfloat16)
        out_a = run_selected_experts(node_a._lm, h, 15, [3])
        out_b = run_selected_experts(node_b._lm, h, 15, [3])
        assert mx.array_equal(out_a[3], out_b[3]).item()
    finally:
        node_a.shutdown()
        node_b.shutdown()
        t_a.join(timeout=3.0)
        t_b.join(timeout=3.0)
