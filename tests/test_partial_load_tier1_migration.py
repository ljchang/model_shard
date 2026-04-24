"""Tier 1 E2E with ENABLE_PARTIAL_LOAD=true AND ENABLE_DYNAMIC_MIGRATION=true.

Verifies that the migration scanner running in the background does not
break token correctness. Short prompts (<=8 tokens) stay on the no-sort path
per 5a §7.5 so bit-exact token ids are expected.

No migration actually fires — heat threshold is 50 picks and short prompts
don't accumulate that — but the scanner ticks every 2 s while the test runs.
"""
from __future__ import annotations

import json
import os
import random
import socket
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from model_shard.client import Client
from model_shard.node import Node
from model_shard.shard_map import NodeAddress, ShardMap, ShardSpec

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "artifacts" / "ref" / "manifest.json"
MAX_TOK = 8  # stay on no-sort path (B*Seq < 64)


def _ids_mod3(r: int) -> tuple[int, ...]:
    return tuple(e for e in range(128) if e % 3 == r)


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


def _wait_listening(host: str, port: int, timeout: float = 5.0) -> None:
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError(f"{host}:{port} never came up")


@pytest.fixture(scope="module")
def three_node_pipeline_with_migration(shards_model_id: str) -> Iterator[Any]:
    # Uses bf16 (shards_model_id) because this test compares distributed
    # output against the bf16 Phase 1 oracle in artifacts/ref/. Partial
    # load keeps per-Node memory small enough to fit 3 Nodes on M5.
    os.environ["ENABLE_EXPERT_SHARD"] = "true"
    os.environ["ENABLE_PARTIAL_LOAD"] = "true"
    os.environ["ENABLE_DYNAMIC_MIGRATION"] = "true"
    os.environ["ENABLE_GOSSIP"] = "true"
    os.environ["MIGRATION_SCAN_INTERVAL_SECONDS"] = "2.0"

    ports = [_free_port() for _ in range(3)]
    specs = [
        ShardSpec(
            shard_id="layer_0-10",
            address=NodeAddress("127.0.0.1", ports[0]),
            start_layer=0, end_layer=10,
            moe_experts={15: _ids_mod3(0)},
        ),
        ShardSpec(
            shard_id="layer_10-20",
            address=NodeAddress("127.0.0.1", ports[1]),
            start_layer=10, end_layer=20,
            moe_experts={15: _ids_mod3(1)},
        ),
        ShardSpec(
            shard_id="layer_20-30",
            address=NodeAddress("127.0.0.1", ports[2]),
            start_layer=20, end_layer=30,
            moe_experts={15: _ids_mod3(2)},
        ),
    ]
    shard_map = ShardMap({s.shard_id: s for s in specs}, model_id=shards_model_id)

    nodes = {
        spec.shard_id: Node(
            shard=spec,
            shard_map=shard_map,
            loaded_model=None,
            total_layers=30,
        )
        for spec in specs
    }
    threads = [
        threading.Thread(target=n.serve_forever, daemon=True)
        for n in nodes.values()
    ]
    for t in threads:
        t.start()
    for s in specs:
        _wait_listening(s.address.host, s.address.port)

    try:
        from tests.conftest import DistributedCluster
        yield DistributedCluster(shard_map=shard_map, nodes_by_id=nodes)
    finally:
        for n in nodes.values():
            n.shutdown()
        for t in threads:
            t.join(timeout=3.0)
        # Clean up env vars so other tests are unaffected.
        for key in (
            "ENABLE_EXPERT_SHARD",
            "ENABLE_PARTIAL_LOAD",
            "ENABLE_DYNAMIC_MIGRATION",
            "ENABLE_GOSSIP",
            "MIGRATION_SCAN_INTERVAL_SECONDS",
        ):
            os.environ.pop(key, None)


@pytest.mark.slow
@pytest.mark.parametrize("prompt_idx", range(5))
def test_tier1_with_migration_enabled(
    three_node_pipeline_with_migration: Any,
    prompt_idx: int,
) -> None:
    """Token ids must be bit-exact with Phase 1 reference while migration scanner runs.

    Phase 7-C-3a note: heavy on bf16 (3 partial-bf16 Node instances + migration
    scanner running in one pytest process). Migration's bit-exactness invariant
    is also covered by tests/test_migration_bit_exact_per_expert.py and
    tests/test_migration_over_tcp.py — both fast. Run this E2E test manually
    when needed."""
    if not MANIFEST.exists():
        pytest.skip("reference manifest missing")
    manifest = json.loads(MANIFEST.read_text())
    record = manifest["prompts"][prompt_idx]
    prompt_tokens = list(record["prompt_tokens"])
    expected = list(record["generated_tokens"])[:MAX_TOK]

    head = three_node_pipeline_with_migration.shard_map.lookup("layer_0-10")
    got = Client(head_address=head.address).generate(prompt_tokens, max_new_tokens=MAX_TOK)
    assert got == expected, (
        f"prompt {prompt_idx}: distributed {got[:10]}... != reference {expected[:10]}..."
    )
