"""3-node Tier 1: kill mid-decode peer, verify head exits cleanly via queue poison.

Phase 5b Task 22: E2E proof that the Phase 3 known issue (decode loop blocked
on token_queue.get() after peer death) is fixed by Task 18's poison branch.

Marked ``slow`` because it requires a real model load and a 3-node in-process
cluster. Expected runtime: ~15s total (model load is session-shared).
"""
from __future__ import annotations

import random
import socket
import threading
import time
from contextlib import closing
from typing import Any

import pytest

from model_shard.client import Client
from model_shard.shard_map import NodeAddress

pytestmark = pytest.mark.slow


def _find_free_port() -> int:
    """Pick a random free TCP port in 30000-60000.

    Phase 2 derives the SWIM UDP port as tcp_port + 1000. macOS ephemeral
    ports can land above 64535, making tcp+1000 overflow 65535. Staying in
    30000-60000 keeps all derived ports valid.
    """
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


@pytest.mark.slow
def test_mid_decode_peer_death_unblocks_head(
    monkeypatch: pytest.MonkeyPatch,
    loaded_model: Any,
) -> None:
    """Kill the tail node mid-decode; assert the head's decode loop exits within
    SWIM's suspect window (default a few seconds) instead of hanging forever.

    The head must raise an error that propagates back through the client.
    """
    monkeypatch.setenv("ENABLE_GOSSIP", "true")
    monkeypatch.setenv("ENABLE_PARTIAL_LOAD", "false")
    monkeypatch.setenv("ENABLE_DYNAMIC_MIGRATION", "false")

    from model_shard.node import Node
    from model_shard.shard_map import ShardMap, ShardSpec

    ports = [_find_free_port() for _ in range(3)]
    specs = [
        ShardSpec(
            shard_id="hang_fix_0-10",
            address=NodeAddress("127.0.0.1", ports[0]),
            start_layer=0,
            end_layer=10,
        ),
        ShardSpec(
            shard_id="hang_fix_10-20",
            address=NodeAddress("127.0.0.1", ports[1]),
            start_layer=10,
            end_layer=20,
        ),
        ShardSpec(
            shard_id="hang_fix_20-30",
            address=NodeAddress("127.0.0.1", ports[2]),
            start_layer=20,
            end_layer=30,
        ),
    ]
    shard_map = ShardMap({s.shard_id: s for s in specs})

    nodes = [
        Node(
            shard=spec,
            shard_map=shard_map,
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

    # Use a short prompt of real token IDs so prefill succeeds without needing
    # a tokenizer. Token 1 = <bos>, then a handful of common sub-word tokens.
    prompt_tokens = [1, 5674, 1]  # <bos> "Hello" <bos>-like — just needs to be valid ints

    errors: list[Exception] = []
    done = threading.Event()

    def drive() -> None:
        try:
            client = Client(head_address=specs[0].address)
            # Very long generation — on this hardware ~85 tok/s in 3-node
            # pipeline, so 2048 tokens takes ~24s; we kill the tail at ~2s
            # to ensure we're firmly in the middle of the decode loop.
            _ = client.generate(prompt_tokens, max_new_tokens=2048)
        except Exception as exc:
            errors.append(exc)
        finally:
            done.set()

    drive_thread = threading.Thread(target=drive, daemon=True)
    drive_thread.start()
    time.sleep(2.0)  # let prefill + several decode rounds run, then kill tail

    # Kill the tail (nodes[2] owns layers 20-30 = is_tail).
    nodes[2].shutdown()
    threads[2].join(timeout=3.0)

    # Head should detect peer death via SWIM and poison the token queue.
    # Default SWIM suspect period is a few seconds; 15s total is generous.
    assert done.wait(timeout=15.0), (
        "decode loop did not exit within 15s after tail peer death — "
        "queue-poison path (Task 18) may not have fired"
    )

    # The client must have received an error propagated from the head.
    assert errors, (
        "expected the client to receive an error after peer death, "
        "but generate() returned without raising"
    )

    # Clean up the remaining nodes.
    for n, th in zip(nodes[:2], threads[:2], strict=True):
        n.shutdown()
        th.join(timeout=3.0)
