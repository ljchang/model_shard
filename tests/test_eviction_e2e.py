"""Phase 6-C slow E2E: 3-node cluster, migrate expert in, force-evict,
verify cluster convergence and continued correctness."""
from __future__ import annotations

import random
import socket as _sk
import threading
import time
from pathlib import Path

import pytest

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


def test_full_attach_evict_cycle_converges(monkeypatch):
    monkeypatch.setenv("ENABLE_PARTIAL_LOAD", "true")
    monkeypatch.setenv("ENABLE_DYNAMIC_MIGRATION", "false")  # drive migration manually
    monkeypatch.setenv("ENABLE_GOSSIP", "true")
    monkeypatch.setenv("ENABLE_EVICTION", "true")
    monkeypatch.setenv("MIGRATION_EVICT_COOLDOWN_SECONDS", "0")

    sm_yaml = ShardMap.from_yaml(Path("config/shards.yaml"))
    ids = sm_yaml.all_shards()
    ports = [_find_free_port() for _ in ids]
    specs = []
    for sid, port in zip(ids, ports):
        s = sm_yaml.lookup(sid)
        specs.append(
            ShardSpec(
                shard_id=sid,
                address=NodeAddress(host="127.0.0.1", port=port),
                start_layer=s.start_layer, end_layer=s.end_layer,
                moe_experts=s.moe_experts,
            )
        )
    sm = ShardMap({s.shard_id: s for s in specs})

    nodes = [Node(shard=s, shard_map=sm, total_layers=30) for s in specs]
    threads = [threading.Thread(target=n.serve_forever, daemon=True) for n in nodes]
    for t in threads: t.start()
    time.sleep(3.0)  # SWIM stabilization

    try:
        head = nodes[0]
        # Expert 40 is in layer_10-20's base config (40 % 3 == 1 per Phase 4 overlap).
        # Pull it to head. After attach, head and layer_10-20 both own 40.
        # Then evict from head; layer_10-20 remains the sole owner.
        target_expert = 40
        source_sid = "layer_10-20"
        source_port = specs[1].address.port

        from model_shard.migration import ExpertWeightPeerRPC
        rpc = ExpertWeightPeerRPC(
            addresses={source_sid: ("127.0.0.1", source_port)}, timeout_s=60.0
        )
        tensors = rpc.pull(source_shard_id=source_sid, layer_idx=15, expert_id=target_expert)
        head.migration_attach(layer_idx=15, expert_id=target_expert, tensors=tensors)
        assert target_expert in head._live_experts[15]

        # Wait for gossip convergence on ADD.
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            if all(
                head._shard.shard_id in n.owners_of(15, target_expert) for n in nodes[1:]
            ):
                break
            time.sleep(0.1)
        assert all(
            head._shard.shard_id in n.owners_of(15, target_expert) for n in nodes[1:]
        ), f"ADD gossip did not converge: {[n.owners_of(15, target_expert) for n in nodes[1:]]}"

        # Evict (cooldown=0, so no wait needed).
        head.migration_detach(15, target_expert)
        assert target_expert not in head._live_experts[15]

        # Wait for gossip convergence on REMOVE.
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            if all(
                head._shard.shard_id not in n.owners_of(15, target_expert)
                for n in nodes[1:]
            ):
                break
            time.sleep(0.1)
        assert all(
            head._shard.shard_id not in n.owners_of(15, target_expert) for n in nodes[1:]
        ), (
            f"REMOVE gossip did not converge: "
            f"{[n.owners_of(15, target_expert) for n in nodes[1:]]}"
        )
    finally:
        for n, th in zip(nodes, threads):
            n.shutdown()
            th.join(timeout=3.0)
