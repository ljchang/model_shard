"""Phase 6-C slow: eviction correctness under ExpertRequest.

Lock invariant: _MLX_COMPUTE_LOCK serializes detach_expert with compute.
After eviction completes, subsequent ExpertRequest arriving sees the
post-eviction _live_experts and correctly returns ERR_WRONG_SHARD (via
Task 5's authority shift)."""
from __future__ import annotations

import random
import socket as _sk
import threading
import time
from pathlib import Path

import mlx.core as mx
import pytest

from model_shard.migration import ExpertWeightPeerRPC
from model_shard.node import Node
from model_shard.shard_map import NodeAddress, ShardMap, ShardSpec

pytestmark = pytest.mark.slow


def _find_free_port() -> int:
    while True:
        p = random.randint(30000, 60000)
        s = _sk.socket()
        try:
            s.bind(("127.0.0.1", p))
            s.close()
            return p
        except OSError:
            s.close()


def test_evicted_expert_serves_wrong_shard_error(monkeypatch):
    """After eviction, subsequent ExpertRequest for the evicted expert
    returns ERR_WRONG_SHARD (not silent success, not hang)."""
    monkeypatch.setenv("ENABLE_PARTIAL_LOAD", "true")
    monkeypatch.setenv("ENABLE_DYNAMIC_MIGRATION", "false")
    monkeypatch.setenv("ENABLE_GOSSIP", "true")
    monkeypatch.setenv("ENABLE_EVICTION", "true")
    monkeypatch.setenv("MIGRATION_EVICT_COOLDOWN_SECONDS", "0")

    sm_yaml = ShardMap.from_yaml(Path("config/shards.yaml"))
    ids = sm_yaml.all_shards()
    ports = [_find_free_port() for _ in ids]
    specs = []
    for sid, port in zip(ids, ports, strict=True):
        s = sm_yaml.lookup(sid)
        specs.append(
            ShardSpec(
                shard_id=sid, address=NodeAddress(host="127.0.0.1", port=port),
                start_layer=s.start_layer, end_layer=s.end_layer, moe_experts=s.moe_experts,
            )
        )
    sm = ShardMap({s.shard_id: s for s in specs})
    nodes = [Node(shard=s, shard_map=sm, total_layers=30) for s in specs]
    threads = [threading.Thread(target=n.serve_forever, daemon=True) for n in nodes]
    for t in threads:
        t.start()
    time.sleep(3.0)

    try:
        head = nodes[0]
        # Expert 40 is in layer_10-20's base config.
        target_expert = 40
        source_sid = "layer_10-20"
        source_port = specs[1].address.port

        # Migrate expert 40 to head.
        rpc_weight = ExpertWeightPeerRPC(
            addresses={source_sid: ("127.0.0.1", source_port)}, timeout_s=60.0
        )
        tensors = rpc_weight.pull(
            source_shard_id=source_sid, layer_idx=15, expert_id=target_expert
        )
        head.migration_attach(
            layer_idx=15, expert_id=target_expert, tensors=tensors
        )
        assert target_expert in head._live_experts[15]

        # Evict it.
        head.migration_detach(15, target_expert)
        assert target_expert not in head._live_experts[15]

        # Now send an ExpertRequest directly to head for the evicted expert.
        # Head's _handle_expert_request must reject with ERR_WRONG_SHARD per Task 5.
        from model_shard.expert_orchestrator import TcpPeerRPC
        direct_rpc = TcpPeerRPC(
            addresses={head._shard.shard_id: ("127.0.0.1", head._shard.address.port)},
            timeout_s=30.0,
        )
        hidden = 2816  # Gemma 4 26B hidden_size.
        h = mx.zeros((1, 1, hidden), dtype=mx.bfloat16)
        with pytest.raises(RuntimeError, match=r"(WRONG_SHARD|not hosted|wrong shard|not held|does not host)"):
            direct_rpc.call(
                peer_shard_id=head._shard.shard_id,
                request_id="r-post-evict",
                layer_idx=15,
                expert_ids=[target_expert],
                h=h,
            )
    finally:
        for n, th in zip(nodes, threads, strict=True):
            n.shutdown()
            th.join(timeout=3.0)
