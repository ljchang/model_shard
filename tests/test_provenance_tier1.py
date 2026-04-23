"""Slow: Tier 1 tokens match Phase 1 reference with ENABLE_PROVENANCE=true.

Provenance is pure bookkeeping -- must not affect token output."""
from __future__ import annotations

import json
import random
import socket as _sk
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from model_shard.client import Client
from model_shard.node import Node
from model_shard.shard_map import NodeAddress, ShardMap, ShardSpec

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "artifacts" / "ref" / "manifest.json"
MAX_TOK = 8  # no-sort path: first 8 tokens, B*Seq <= 7 window

pytestmark = pytest.mark.slow


def _find_free_port() -> int:
    for _ in range(100):
        p = random.randint(30000, 60000)
        s = _sk.socket()
        try:
            s.bind(("127.0.0.1", p))
            s.close()
            return p
        except OSError:
            s.close()
    raise RuntimeError("no free port")


def _wait_listening(host: str, port: int, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with _sk.create_connection((host, port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError(f"{host}:{port} never came up")


def test_tier1_tokens_match_with_provenance_on(monkeypatch: Any, shards_model_id: str) -> None:
    if not MANIFEST.exists():
        pytest.skip(
            "reference artifacts missing -- run: "
            "uv run python scripts/run_reference.py "
            "--prompt-set tests/prompts.json --out-dir artifacts/ref"
        )

    monkeypatch.setenv("ENABLE_PROVENANCE", "true")
    monkeypatch.setenv("ENABLE_GOSSIP", "false")
    monkeypatch.setenv("ENABLE_PARTIAL_LOAD", "false")
    monkeypatch.setenv("ENABLE_DYNAMIC_MIGRATION", "false")
    monkeypatch.setenv("ENABLE_EXPERT_RETRY", "false")
    monkeypatch.setenv("ENABLE_EXPERT_SHARD", "false")

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

    from model_shard.mlx_engine import load_model
    lm = load_model(shards_model_id)

    nodes = [
        Node(shard=s, shard_map=sm, loaded_model=lm, total_layers=30)
        for s in specs
    ]
    threads = [
        threading.Thread(target=n.serve_forever, daemon=True) for n in nodes
    ]
    for t in threads:
        t.start()
    for s in specs:
        _wait_listening(s.address.host, s.address.port)

    head_spec = next(s for s in specs if s.start_layer == 0)
    client = Client(head_address=head_spec.address)

    manifest = json.loads(MANIFEST.read_text())
    prompts = manifest["prompts"]

    try:
        for rec in prompts[:2]:  # first 2 prompts to keep test time bounded
            prompt_ids = list(rec["prompt_tokens"])
            expected = list(rec["generated_tokens"])[:MAX_TOK]
            got = client.generate(prompt_tokens=prompt_ids, max_new_tokens=len(expected))
            assert got == expected, (
                f"tokens diverged with provenance on for prompt {rec.get('id', '?')!r}: "
                f"got {got}, want {expected}"
            )
    finally:
        for n in nodes:
            n.shutdown()
        for th in threads:
            th.join(timeout=3.0)
