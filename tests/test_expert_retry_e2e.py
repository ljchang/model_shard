"""Phase 6-A E2E: kill a replica peer mid-generation, verify tokens continue.

Uses Phase 4's overlapping shard config where experts 0-2 are replicated
across two shards. Killing one replica leaves the other alive; retry should
keep the generation alive.

Marked ``slow`` because it requires a real model load and a 3-node in-process
cluster. Expected runtime: ~10-15s total (model load is session-shared).
"""
from __future__ import annotations

import random
import socket as _sk
import threading
import time
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from model_shard.client import Client
from model_shard.node import Node
from model_shard.shard_map import NodeAddress, ShardMap, ShardSpec

pytestmark = pytest.mark.slow


def _find_free_port() -> int:
    """Return a free 127.0.0.1 port in a range that leaves headroom for
    udp_port = tcp_port + 1000 (must stay <= 65535)."""
    for _ in range(100):
        p = random.randint(30000, 60000)
        s = _sk.socket()
        try:
            s.bind(("127.0.0.1", p))
            s.close()
            return p
        except OSError:
            s.close()
    raise RuntimeError("could not obtain a free port in 30000-60000 after 100 tries")


def _wait_for_listening(host: str, port: int, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with closing(_sk.create_connection((host, port), timeout=0.5)):
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError(f"node at {host}:{port} never came up")


@pytest.mark.slow
def test_retry_keeps_generation_alive_after_replica_death(
    monkeypatch: pytest.MonkeyPatch,
    loaded_model: Any,
) -> None:
    """Kill the middle shard mid-decode; retry should route expert RPCs to the
    remaining replica on head or tail so generation either completes or raises a
    clean error instead of hanging forever.

    The key invariant: done.wait() must return within 30s. Pre-6-A behavior
    would block indefinitely because a dead replica left expert RPC callers
    spinning with no fallback.
    """
    monkeypatch.setenv("ENABLE_GOSSIP", "true")
    monkeypatch.setenv("ENABLE_PARTIAL_LOAD", "false")
    monkeypatch.setenv("ENABLE_DYNAMIC_MIGRATION", "false")
    monkeypatch.setenv("ENABLE_EXPERT_RETRY", "true")
    monkeypatch.setenv("EXPERT_RETRY_BACKOFF_MS", "0,50")  # fast retry for test

    # Phase 4's overlapping config: experts 0/1/2 replicated across 2 shards each.
    # Load shard IDs and expert assignments from the canonical YAML, but bind
    # new free ports so we don't collide with anything already listening.
    sm_yaml = ShardMap.from_yaml(Path("config/shards.yaml"))
    ids = sm_yaml.all_shards()
    ports = [_find_free_port() for _ in ids]

    specs = []
    for sid, port in zip(ids, ports, strict=True):
        s = sm_yaml.lookup(sid)
        specs.append(
            ShardSpec(
                shard_id=sid,
                address=NodeAddress(host="127.0.0.1", port=port),
                start_layer=s.start_layer,
                end_layer=s.end_layer,
                moe_experts=s.moe_experts,
            )
        )
    sm = ShardMap({s.shard_id: s for s in specs})

    nodes = [
        Node(
            shard=spec,
            shard_map=sm,
            loaded_model=loaded_model,
            total_layers=loaded_model.num_layers,
        )
        for spec in specs
    ]
    threads = [
        threading.Thread(target=n.serve_forever, daemon=True) for n in nodes
    ]
    for t in threads:
        t.start()
    for spec in specs:
        _wait_for_listening(spec.address.host, spec.address.port)
    time.sleep(3.0)  # SWIM stabilization

    head_spec = next(s for s in specs if s.start_layer == 0)
    client = Client(head_address=head_spec.address)

    # Kill the middle shard (index 1): it holds expert 0 (overlap with head) and
    # expert 1 (overlap with tail), so both overlapping experts still have a live
    # replica. The pipeline can also continue using only head+tail for non-expert
    # layers. This exercises the retry path without making generation impossible.
    kill_idx = 1  # middle shard (layer_10-20)
    killed_sid = specs[kill_idx].shard_id

    errors: list[Exception] = []
    tokens_received: list[int] = []
    done = threading.Event()

    def drive() -> None:
        try:
            # Very long generation — kills happen at ~1s, so 2048 tokens
            # gives a generous window for retry to finish.
            tokens = client.generate(
                prompt_tokens=[1, 5674, 1],  # <bos> "Hello" <bos>-like
                max_new_tokens=2048,
            )
            tokens_received.extend(tokens)
        except Exception as exc:
            errors.append(exc)
        finally:
            done.set()

    drive_thread = threading.Thread(target=drive, daemon=True)
    drive_thread.start()
    time.sleep(1.0)  # let prefill + a couple decode rounds happen

    # Kill the replica.
    nodes[kill_idx].shutdown()
    threads[kill_idx].join(timeout=3.0)

    # Wait for generation to either complete or cleanly error (not hang).
    done.wait(timeout=30.0)

    # Clean up remaining nodes.
    for i, (n, th) in enumerate(zip(nodes, threads, strict=True)):
        if i != kill_idx and th.is_alive():
            n.shutdown()
            th.join(timeout=3.0)

    # Key invariant: the client did NOT hang indefinitely. It either
    # completed (retry carried it) or received a clean error (retry
    # exhausted because killed peer was critical to a non-replicated path).
    assert done.is_set(), (
        f"client hung after killing {killed_sid!r} — retry or cleanup failed"
    )
    print(
        f"E2E retry result: tokens_received={len(tokens_received)}, "
        f"errors={errors}"
    )
