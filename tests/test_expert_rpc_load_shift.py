"""E2E: gossip-delivered peer loads are observable via /loads endpoint."""

from __future__ import annotations

import contextlib
import json
import os
import random
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
RUN_NODE = REPO / "scripts" / "run_node.py"


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


def _write_shards(tmp_path: Path) -> tuple[Path, dict[str, int]]:
    head, mid, tail = _free_port(), _free_port(), _free_port()
    cfg = {
        "shards": {
            "head": {
                "host": "127.0.0.1", "port": head,
                "start_layer": 0, "end_layer": 10,
                "moe_experts": {15: [0, 3]},
            },
            "mid": {
                "host": "127.0.0.1", "port": mid,
                "start_layer": 10, "end_layer": 20,
                "moe_experts": {15: [0, 1]},
            },
            "tail": {
                "host": "127.0.0.1", "port": tail,
                "start_layer": 20, "end_layer": 30,
                "moe_experts": {15: [2]},
            },
        }
    }
    p = tmp_path / "shards.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p, {"head": head, "mid": mid, "tail": tail}


def _spawn(shard_id: str, cfg: Path) -> subprocess.Popen:  # type: ignore[type-arg]
    env = {
        **os.environ,
        "ENABLE_GOSSIP": "true",
        "ENABLE_EXPERT_SHARD": "true",
        "SHARD_DRY_RUN": "true",
    }
    return subprocess.Popen(
        [sys.executable, str(RUN_NODE), "--shard", shard_id, "--config", str(cfg)],
        env=env,
        stderr=subprocess.PIPE,
    )


def _get_loads(debug_port: int) -> dict[str, int] | None:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{debug_port}/loads", timeout=1.0
        ) as resp:
            data = json.loads(resp.read())
            return {k: v["queue_depth_ema"] for k, v in data.items()}
    except Exception:
        return None


@pytest.mark.slow
def test_gossip_delivers_peer_loads_within_twenty_seconds(tmp_path: Path) -> None:
    cfg, ports = _write_shards(tmp_path)
    procs = {sid: _spawn(sid, cfg) for sid in ("head", "mid", "tail")}
    try:
        head_debug = ports["head"] + 2000
        deadline = time.monotonic() + 20.0
        last: dict[str, int] | None = None
        while time.monotonic() < deadline:
            view = _get_loads(head_debug)
            last = view
            if view is not None and set(view.keys()) >= {"head", "mid", "tail"}:
                return
            time.sleep(0.5)
        pytest.fail(f"head did not see all peer loads within 20s; final={last}")
    finally:
        for p in procs.values():
            with contextlib.suppress(ProcessLookupError):
                p.terminate()
        for p in procs.values():
            with contextlib.suppress(subprocess.TimeoutExpired):
                p.wait(timeout=5)
