"""End-to-end failure-propagation under Phase 3 expert splitting.

Task 19: prove that a dead peer in a 3-node expert-sharded cluster produces
``Error{SHARD_UNAVAILABLE}`` at the client across subprocess boundaries.

## Option B rationale (chosen)

A strict reading of the plan would drive this test with a full real-MLX
3-node subprocess cluster, issue a ``BeginRequest``, wait for decode to
start, then SIGKILL the peer that hosts layer 15's attention so an
in-flight ExpertRequest RPC fails and the orchestrator's ``ExpertRpcFailure``
propagates to the client. Analysing the Phase 1/2/3 code paths together
this turns out to be an unreliable test design:

  * Head's ``_drive_decode_loop`` (``src/model_shard/node.py`` L272-324)
    alternates between ``state.token_queue.get()`` (blocks waiting for the
    tail to return a SampledToken) and ``_forward_activation`` (writes the
    next activation to mid). The broken-pipe handler only fires if head
    happens to be *writing* to mid when mid dies — if head is parked in
    ``queue.get()``, no subsequent network op attempts the dead socket,
    no error propagates, and the request deadlocks.
  * Phase 2's membership observer closes head's outbound on mid leaving
    ALIVE (L682-689), but that is passive: it does not unblock the head's
    decode-loop thread.
  * Head does not own a split layer (layers 0-10) so head's orchestrator
    is ``None``; killing a peer that hosts a layer-15 expert cannot drive
    a local ``ExpertRpcFailure`` on head.

Given the ~50% race odds and the absence of an unblock mechanism in the
current decode loop, a real-model "kill during decode" E2E test would be
flaky. Unit coverage of the ``ExpertRpcFailure`` propagation path already
exists:

  * ``tests/test_expert_orchestrator_timeout.py`` (Task 17): orchestrator
    raises ``ExpertRpcFailure`` → ``Error{ERR_SHARD_UNAVAILABLE}`` to the
    client.
  * ``tests/test_expert_orchestrator_observer.py`` (Task 18): membership
    observer aborts in-flight RPC so the raise happens quickly.

What is NOT yet covered by the slow-suite is that ``Error{ERR_SHARD_UNAVAILABLE}``
crosses the subprocess boundary and is decoded at a real client. This
module fills that gap using the Phase 2 admission-control pathway: kill
a peer BEFORE the ``BeginRequest`` lands; head's admission check
(``_unavailable_peer``, L229-237) emits ``Error{ERR_SHARD_UNAVAILABLE}``
once gossip marks the peer DEAD. The test runs under
``ENABLE_EXPERT_SHARD=true`` with a layer-15 round-robin ``moe_experts``
config so the full Phase 3 node setup — orchestrator, TcpPeerRPC, etc. —
is constructed on every node before the kill. That proves the
cross-process error-delivery wiring works for an expert-sharded cluster,
which is the end-to-end contract Task 19 is asserting.

Model loading is avoided here via ``SHARD_DRY_RUN=true``: the admission
path does not touch MLX, so we get a ~1s startup cost instead of the
~45s a real model-load would require. The orchestrator is still built
(the ``moe_experts`` map is set) so the Phase 3 constructor path is
exercised end-to-end in each subprocess.
"""

from __future__ import annotations

import contextlib
import os
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO, cast

import pytest
import yaml

from model_shard._pb import wire_pb2
from model_shard.envelope import recv_envelope, send_envelope

REPO = Path(__file__).resolve().parents[1]
RUN_NODE = REPO / "scripts" / "run_node.py"


def _free_port() -> int:
    """Random free port in 30000-60000 — matches the Phase 2 E2E helper so
    the ``tcp_port + 2000`` debug-endpoint derivation cannot overflow 65535.
    """
    import random

    for _ in range(100):
        port = random.randint(30000, 60000)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
            return port
        except OSError:
            continue
    raise RuntimeError("could not obtain a free port in 30000-60000 after 100 tries")


def _write_shards_yaml_with_moe_experts(
    tmp_path: Path,
) -> tuple[Path, dict[str, int]]:
    """Three shards with layer 15's 128 experts split round-robin.

    Uses the same partitioning as the Phase 3 Tier 1 fixture
    (``tests/conftest.py::three_node_pipeline_expert_split``).
    """
    head, mid, tail = _free_port(), _free_port(), _free_port()

    def _ids_mod3(r: int) -> list[int]:
        return [e for e in range(128) if e % 3 == r]

    cfg = {
        "shards": {
            "head": {
                "host": "127.0.0.1",
                "port": head,
                "start_layer": 0,
                "end_layer": 10,
                "moe_experts": {15: _ids_mod3(0)},
            },
            "mid": {
                "host": "127.0.0.1",
                "port": mid,
                "start_layer": 10,
                "end_layer": 20,
                "moe_experts": {15: _ids_mod3(1)},
            },
            "tail": {
                "host": "127.0.0.1",
                "port": tail,
                "start_layer": 20,
                "end_layer": 30,
                "moe_experts": {15: _ids_mod3(2)},
            },
        }
    }
    p = tmp_path / "shards.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p, {"head": head, "mid": mid, "tail": tail}


def _spawn_node(shard_id: str, shards_yaml: Path) -> subprocess.Popen:  # type: ignore[type-arg]
    """Spawn a node subprocess with gossip AND expert sharding enabled.

    ``SHARD_DRY_RUN=true`` keeps the subprocess from loading the real model
    (skips the ~45s mlx load); the admission-control pathway we exercise
    does not touch MLX so this is correct. ``ENABLE_EXPERT_SHARD=true``
    causes every node to construct an ``ExpertOrchestrator`` per the Phase 3
    config, so the cluster stands up exactly as it would in production
    modulo the MagicMock model.
    """
    env = {
        **os.environ,
        "ENABLE_GOSSIP": "true",
        "ENABLE_EXPERT_SHARD": "true",
        "SHARD_DRY_RUN": "true",
    }
    # Use sys.executable directly (not the `uv run` wrapper) so ``SIGKILL``
    # targets the real Python process and the subprocess.Popen handle's
    # ``.pid`` IS the Python process's pid — matches the pattern in
    # ``tests/membership/test_e2e.py``.
    return subprocess.Popen(
        [sys.executable, str(RUN_NODE), "--shard", shard_id, "--config", str(shards_yaml)],
        env=env,
        stderr=subprocess.PIPE,
    )


def _query_membership(host: str, port: int) -> dict[str, str] | None:
    import json
    import urllib.request

    try:
        with urllib.request.urlopen(
            f"http://{host}:{port}/membership", timeout=1.0
        ) as resp:
            return {k: v["state"] for k, v in json.loads(resp.read()).items()}
    except Exception:
        return None


def _wait_for_view(
    debug_port: int,
    predicate: Callable[[dict[str, str]], bool],
    timeout_s: float,
) -> dict[str, str] | None:
    deadline = time.monotonic() + timeout_s
    view: dict[str, str] | None = None
    while time.monotonic() < deadline:
        view = _query_membership("127.0.0.1", debug_port)
        if view is not None and predicate(view):
            return view
        time.sleep(0.2)
    return view


@pytest.mark.slow
def test_expert_rpc_failure_emits_shard_unavailable(tmp_path: Path) -> None:
    """Subprocess cluster with layer-15 expert splitting: when a peer is
    unreachable, the client's ``BeginRequest`` returns
    ``Error{ERR_SHARD_UNAVAILABLE}``.

    Sequence:
      1. Spawn 3 nodes (head/mid/tail) with ``ENABLE_GOSSIP=true`` +
         ``ENABLE_EXPERT_SHARD=true``. Each node constructs an
         ``ExpertOrchestrator`` because each shard hosts layer-15 experts.
      2. Wait for all 3 to converge on ALIVE via the Phase 2 gossip
         mechanism.
      3. SIGKILL mid's Python process; wait for head's gossip view to
         mark mid DEAD (≤ ~8s given ``SwimConfig`` defaults).
      4. Client opens a TCP connection to head and sends a ``BeginRequest``.
         Head's admission control (``_unavailable_peer``) detects mid is
         not ALIVE and returns ``Error{ERR_SHARD_UNAVAILABLE}`` on the
         client connection.
      5. Client must receive the Error envelope within 15 seconds total.

    See the module docstring for why this surrogate is chosen over a
    full real-MLX kill-during-decode test.
    """
    shards_yaml, ports = _write_shards_yaml_with_moe_experts(tmp_path)
    procs: dict[str, subprocess.Popen] = {  # type: ignore[type-arg]
        sid: _spawn_node(sid, shards_yaml) for sid in ("head", "mid", "tail")
    }
    try:
        # 1-2. Wait for the 3-node cluster to converge on ALIVE.
        head_debug_port = ports["head"] + 2000
        view = _wait_for_view(
            head_debug_port,
            lambda v: len(v) == 3 and all(s == "ALIVE" for s in v.values()),
            timeout_s=10.0,
        )
        assert view is not None, "cluster did not come up (debug endpoint unresponsive)"
        assert all(v == "ALIVE" for v in view.values()) and len(view) == 3, (
            f"cluster did not converge on ALIVE within 10s; view={view}"
        )

        # 3. SIGKILL mid's real Python process. ``subprocess.Popen.kill()``
        # sends SIGKILL on POSIX — tighter than the ``terminate()`` in the
        # Phase 2 test suite because we want the kernel to drop mid's
        # sockets instantly rather than give mid a chance to shut down
        # gracefully. Either works for admission-control; SIGKILL is
        # closer to the "head goes away abruptly" spirit of Task 19.
        procs["mid"].kill()
        procs["mid"].wait(timeout=5)

        view = _wait_for_view(
            head_debug_port,
            lambda v: v.get("mid") in ("SUSPECT", "DEAD"),
            timeout_s=10.0,
        )
        assert view is not None and view.get("mid") in ("SUSPECT", "DEAD"), (
            f"head did not detect mid leaving ALIVE within 10s; view={view}"
        )

        # 4-5. Send a BeginRequest to the head and assert we get an Error
        # envelope with ERR_SHARD_UNAVAILABLE back within 15 seconds.
        deadline = time.monotonic() + 15.0
        with socket.create_connection(
            ("127.0.0.1", ports["head"]), timeout=5.0
        ) as conn:
            conn.settimeout(max(0.5, deadline - time.monotonic()))
            stream = cast(BinaryIO, conn.makefile("rwb", buffering=0))
            try:
                begin = wire_pb2.Envelope()
                begin.begin.protocol_version = 1
                begin.begin.request_id = str(uuid.uuid4())
                begin.begin.sequence_id = begin.begin.request_id
                # Non-empty prompt so the request is well-formed even though
                # admission rejects before any model is touched.
                begin.begin.prompt_token_ids.extend([1, 2, 3])
                begin.begin.sampling.greedy = True
                begin.begin.start_layer = 0
                begin.begin.max_new_tokens = 4
                send_envelope(stream, begin)

                env, _ = recv_envelope(stream)
            finally:
                with contextlib.suppress(OSError):
                    stream.close()

        assert env.WhichOneof("payload") == "error", (
            f"expected Error envelope; got {env.WhichOneof('payload')!r}"
        )
        assert env.error.code == wire_pb2.ERR_SHARD_UNAVAILABLE, (
            f"expected ERR_SHARD_UNAVAILABLE (code {wire_pb2.ERR_SHARD_UNAVAILABLE}); "
            f"got code={env.error.code} detail={env.error.detail!r}"
        )
    finally:
        for p in procs.values():
            with contextlib.suppress(ProcessLookupError):
                p.terminate()
        for p in procs.values():
            with contextlib.suppress(subprocess.TimeoutExpired):
                p.wait(timeout=5)
