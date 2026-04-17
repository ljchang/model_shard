"""Slow: corrupting one byte of a chain entry causes downstream rejection.

Monkeypatches Node._forward_activation on the mid node to flip one byte
of one entry's hash before sending downstream. The tail should validate
and reject with ERR_INVALID_PROVENANCE; the client should receive a clean
error (not hang)."""
from __future__ import annotations

import random
import socket as _sk
import threading
import time

import pytest

from model_shard.client import Client
from model_shard.node import Node
from model_shard.request import ProvenanceEntry
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


def test_corrupted_chain_gets_rejected(monkeypatch):
    monkeypatch.setenv("ENABLE_PROVENANCE", "true")
    monkeypatch.setenv("ENABLE_GOSSIP", "true")
    monkeypatch.setenv("ENABLE_PARTIAL_LOAD", "false")
    monkeypatch.setenv("ENABLE_DYNAMIC_MIGRATION", "false")

    # Load model once; shared across all three in-process nodes.
    from model_shard.mlx_engine import load_model
    lm = load_model("mlx-community/gemma-4-26b-a4b-it-4bit")

    ports = [_find_free_port() for _ in range(3)]
    specs = [
        ShardSpec(
            shard_id="layer_0-10",
            address=NodeAddress(host="127.0.0.1", port=ports[0]),
            start_layer=0,
            end_layer=10,
        ),
        ShardSpec(
            shard_id="layer_10-20",
            address=NodeAddress(host="127.0.0.1", port=ports[1]),
            start_layer=10,
            end_layer=20,
        ),
        ShardSpec(
            shard_id="layer_20-30",
            address=NodeAddress(host="127.0.0.1", port=ports[2]),
            start_layer=20,
            end_layer=30,
        ),
    ]
    sm = ShardMap({s.shard_id: s for s in specs})

    nodes = [Node(shard=s, shard_map=sm, loaded_model=lm, total_layers=30) for s in specs]

    # Monkeypatch the MID node's _forward_activation to corrupt the last
    # entry's hash before sending downstream.
    mid_node = next(n for n in nodes if n._shard.start_layer == 10)
    orig_forward = mid_node._forward_activation

    def corrupting_forward(request_id, h, provenance_chain=None):
        if provenance_chain:
            last = provenance_chain[-1]
            # Flip one byte of the hash; preserve everything else.
            corrupted_hash = last.hash[:5] + bytes([(last.hash[5] ^ 0xFF)]) + last.hash[6:]
            corrupted = ProvenanceEntry(
                shard_id=last.shard_id, node_id=last.node_id,
                timestamp=last.timestamp,
                hash=corrupted_hash,
                parent_hashes=last.parent_hashes, op=last.op,
            )
            provenance_chain[-1] = corrupted
        return orig_forward(request_id, h, provenance_chain=provenance_chain)

    mid_node._forward_activation = corrupting_forward

    threads = [threading.Thread(target=n.serve_forever, daemon=True) for n in nodes]
    for t in threads:
        t.start()
    time.sleep(3.0)

    head_spec = next(s for s in specs if s.start_layer == 0)
    client = Client(head_address=head_spec.address)

    errors: list[Exception] = []
    done = threading.Event()

    def drive():
        try:
            client.generate(prompt_tokens=[1, 5674, 1], max_new_tokens=8)
        except Exception as e:
            errors.append(e)
        finally:
            done.set()

    t = threading.Thread(target=drive, daemon=True)
    t.start()
    assert done.wait(timeout=15.0), "client hung after corrupted-chain delivery"
    assert errors, "expected client to receive an error"
    # Sanity check: the error message should mention provenance.
    assert any(
        "provenance" in str(e).lower()
        or "INVALID_PROVENANCE" in str(e)
        for e in errors
    ), f"errors did not mention provenance: {errors}"

    for n, th in zip(nodes, threads, strict=False):
        n.shutdown()
        th.join(timeout=3.0)
