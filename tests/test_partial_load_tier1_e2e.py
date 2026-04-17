"""Tier 1 E2E with partial-load enabled: each of 3 in-process nodes runs
with a sliced layer-15 expert subset; tokens must match Phase 1 reference."""

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
MAX_TOK = 32


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
def three_node_pipeline_partial_load() -> Iterator[Any]:
    os.environ["ENABLE_EXPERT_SHARD"] = "true"
    os.environ["ENABLE_PARTIAL_LOAD"] = "true"

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
    shard_map = ShardMap({s.shard_id: s for s in specs})
    # Each node loads its own sliced model via ENABLE_PARTIAL_LOAD.
    nodes = {
        spec.shard_id: Node(
            shard=spec, shard_map=shard_map,
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


@pytest.mark.slow
@pytest.mark.parametrize("prompt_idx", range(5))
def test_tier1_under_partial_load(
    three_node_pipeline_partial_load: Any,
    prompt_idx: int,
) -> None:
    if not MANIFEST.exists():
        pytest.skip("reference manifest missing")
    manifest = json.loads(MANIFEST.read_text())
    record = manifest["prompts"][prompt_idx]
    prompt_tokens = list(record["prompt_tokens"])
    expected = list(record["generated_tokens"])[:MAX_TOK]

    head = three_node_pipeline_partial_load.shard_map.lookup("layer_0-10")
    got = Client(head_address=head.address).generate(prompt_tokens, max_new_tokens=MAX_TOK)
    assert got == expected, (
        f"prompt {prompt_idx}: distributed {got[:10]}... != reference {expected[:10]}..."
    )
