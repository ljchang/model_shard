"""Behavioral end-to-end tests for the membership layer.

Marked `slow` because each test starts/stops 3 real Python subprocesses
(via `scripts/run_node.py`) and the model is mocked out via the
SHARD_DRY_RUN env var.
"""

from __future__ import annotations

import contextlib
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
RUN_NODE = REPO / "scripts" / "run_node.py"


def _free_port() -> int:
    # Each node needs tcp_port, tcp_port+1000 (UDP gossip), and
    # tcp_port+2000 (HTTP debug) to all fit in 1..65535. The macOS
    # ephemeral range (49152-65535) can hand back ports whose +2000
    # derivation overflows, so we pick a random free port in 30000-60000.
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


def _write_shards_yaml(tmp_path: Path) -> tuple[Path, dict[str, int]]:
    """Write a temporary shards.yaml with random ports; return path and tcp ports."""
    head, mid, tail = _free_port(), _free_port(), _free_port()
    cfg = {
        "shards": {
            "head": {"host": "127.0.0.1", "port": head, "start_layer": 0, "end_layer": 10},
            "mid": {"host": "127.0.0.1", "port": mid, "start_layer": 10, "end_layer": 20},
            "tail": {"host": "127.0.0.1", "port": tail, "start_layer": 20, "end_layer": 30},
        }
    }
    p = tmp_path / "shards.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p, {"head": head, "mid": mid, "tail": tail}


def _spawn_node(shard_id: str, shards_yaml: Path) -> subprocess.Popen:  # type: ignore[type-arg]
    env = {**os.environ, "ENABLE_GOSSIP": "true", "SHARD_DRY_RUN": "true"}
    return subprocess.Popen(
        [sys.executable, str(RUN_NODE), "--shard", shard_id, "--config", str(shards_yaml)],
        env=env,
        stderr=subprocess.PIPE,
    )


@contextlib.contextmanager
def _cluster(
    tmp_path: Path,
) -> Iterator[tuple[Path, dict[str, int], dict[str, subprocess.Popen]]]:  # type: ignore[type-arg]
    shards_yaml, ports = _write_shards_yaml(tmp_path)
    procs = {sid: _spawn_node(sid, shards_yaml) for sid in ("head", "mid", "tail")}
    try:
        yield shards_yaml, ports, procs
    finally:
        for p in procs.values():
            with contextlib.suppress(ProcessLookupError):
                p.terminate()
        for p in procs.values():
            with contextlib.suppress(subprocess.TimeoutExpired):
                p.wait(timeout=5)


def _query_view(host: str, port: int) -> dict[str, str] | None:
    """Reach into the head's debug HTTP endpoint."""
    import json
    import urllib.request

    try:
        with urllib.request.urlopen(
            f"http://{host}:{port}/membership", timeout=1.0
        ) as resp:
            return {k: v["state"] for k, v in json.loads(resp.read()).items()}
    except Exception:
        return None


@pytest.mark.slow
def test_three_nodes_converge_on_alive(tmp_path: Path) -> None:
    with _cluster(tmp_path) as (_, ports, _):
        debug_port = ports["head"] + 2000  # convention: tcp_port + 2000
        deadline = time.monotonic() + 5.0
        view = None
        while time.monotonic() < deadline:
            view = _query_view("127.0.0.1", debug_port)
            if view and all(v == "ALIVE" for v in view.values()) and len(view) == 3:
                return
            time.sleep(0.2)
        pytest.fail(f"cluster did not converge within 5s; final view={view}")


@pytest.mark.slow
def test_kill_one_node_others_detect_dead(tmp_path: Path) -> None:
    with _cluster(tmp_path) as (_, ports, procs):
        debug_port = ports["head"] + 2000
        # Wait for convergence first.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            view = _query_view("127.0.0.1", debug_port)
            if view and all(v == "ALIVE" for v in view.values()) and len(view) == 3:
                break
            time.sleep(0.2)
        else:
            pytest.fail("did not converge before kill")

        procs["mid"].terminate()
        procs["mid"].wait(timeout=3)

        # Within ~7s the head should mark mid dead.
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            view = _query_view("127.0.0.1", debug_port)
            if view and view.get("mid") == "DEAD":
                return
            time.sleep(0.2)
        pytest.fail(f"head did not detect mid dead; final view={view}")


@pytest.mark.slow
def test_killed_node_rejoins_returns_to_alive(tmp_path: Path) -> None:
    with _cluster(tmp_path) as (shards_yaml, ports, procs):
        debug_port = ports["head"] + 2000
        # Converge.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            view = _query_view("127.0.0.1", debug_port)
            if view and all(v == "ALIVE" for v in view.values()):
                break
            time.sleep(0.2)
        # Kill mid.
        procs["mid"].terminate()
        procs["mid"].wait(timeout=3)
        # Wait for dead.
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            view = _query_view("127.0.0.1", debug_port)
            if view and view.get("mid") == "DEAD":
                break
            time.sleep(0.2)
        # Restart mid.
        procs["mid"] = _spawn_node("mid", shards_yaml)
        # Within ~5s mid should be alive again.
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            view = _query_view("127.0.0.1", debug_port)
            if view and view.get("mid") == "ALIVE":
                return
            time.sleep(0.2)
        pytest.fail(f"mid did not rejoin to alive; final view={view}")


@pytest.mark.slow
def test_bootstrap_with_unreachable_seeds_starts_in_single_node_view(
    tmp_path: Path,
) -> None:
    """Spawn ONE node whose shards.yaml lists two non-running peers.
    The node must boot and report itself alive; the missing peers should be
    detected as suspect/dead within T_SUSPECT."""
    shards_yaml, ports = _write_shards_yaml(tmp_path)
    debug_port = ports["head"] + 2000
    head_proc = _spawn_node("head", shards_yaml)
    try:
        deadline = time.monotonic() + 8.0
        view: dict[str, str] | None = None
        while time.monotonic() < deadline:
            view = _query_view("127.0.0.1", debug_port)
            if (
                view
                and view.get("head") == "ALIVE"
                and view.get("mid") in ("SUSPECT", "DEAD")
            ):
                return
            time.sleep(0.2)
        pytest.fail(f"single-node bootstrap fallback failed; final view={view}")
    finally:
        head_proc.terminate()
        head_proc.wait(timeout=3)
