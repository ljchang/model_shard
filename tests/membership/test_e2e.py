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
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


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
