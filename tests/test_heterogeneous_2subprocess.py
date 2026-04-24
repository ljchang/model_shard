"""Phase 7-C-3b Task 7: heterogeneous Mac MLX + Mac PyTorch CPU pipeline.

Spawns two subprocesses on localhost using the existing
``scripts/run_node.py``, one with ``MODEL_SHARD_BACKEND=mlx`` and one
with ``MODEL_SHARD_BACKEND=pytorch``. Then drives a Tier 1 prompt
through the pipeline using the existing ``Client``. Asserts token-exact
match against the Phase 1 oracle.

Memory requirement: >=80 GB unified (Mac M5 default config). PyTorch
on Mac CPU is slow on Gemma 4 26B (~minutes per token), so this test
limits to 1 prompt and a short ``max_new_tokens``. The point is
protocol correctness, not throughput.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import tempfile
import time
from contextlib import closing
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MANIFEST = _REPO_ROOT / "artifacts" / "ref" / "manifest.json"
_SHARDS_YAML_TMPL = """
model_id: "google/gemma-4-26B-A4B-it"
shards:
  head:
    host: 127.0.0.1
    port: {port_head}
    start_layer: 0
    end_layer: 15
  tail:
    host: 127.0.0.1
    port: {port_tail}
    start_layer: 15
    end_layer: 30
"""

# How many tokens to generate. Lower than the standard Tier 1 (64) because
# PyTorch on Mac CPU is glacial on Gemma 4 26B.
_MAX_NEW_TOKENS = 4

# Which prompt index from artifacts/ref/manifest.json to test.
_PROMPT_IDX = 0


def _free_port() -> int:
    # Membership UDP is tcp+1000; cap below 64535.
    import random

    for _ in range(100):
        port = random.randint(30000, 60000)
        try:
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
                s.bind(("127.0.0.1", port))
            return port
        except OSError:
            continue
    raise RuntimeError("could not obtain free port")


def _wait_listening(host: str, port: int, timeout_s: float = 60.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with closing(socket.create_connection((host, port), timeout=1.0)):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"{host}:{port} never came up")


@pytest.mark.slow
def test_heterogeneous_mlx_head_pytorch_tail_tier1() -> None:
    """Mac MLX head (layers 0-14) + Mac PyTorch CPU tail (layers 15-29)
    produce the same token sequence as the bf16 single-backend oracle."""
    if not _MANIFEST.exists():
        pytest.skip("reference manifest missing; run scripts/run_reference.py first")

    manifest = json.loads(_MANIFEST.read_text())
    record = manifest["prompts"][_PROMPT_IDX]
    prompt_tokens = list(record["prompt_tokens"])
    expected = list(record["generated_tokens"])[:_MAX_NEW_TOKENS]

    port_head = _free_port()
    port_tail = _free_port()
    cfg_text = _SHARDS_YAML_TMPL.format(port_head=port_head, port_tail=port_tail)

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = Path(tmpdir) / "shards.yaml"
        cfg_path.write_text(cfg_text)

        # Capture subprocess output to files for debugging if either fails.
        head_log = Path(tmpdir) / "head.log"
        tail_log = Path(tmpdir) / "tail.log"

        env_head = {**os.environ, "MODEL_SHARD_BACKEND": "mlx"}
        env_tail = {**os.environ, "MODEL_SHARD_BACKEND": "pytorch"}

        cmd_head = [
            "uv",
            "run",
            "python",
            "scripts/run_node.py",
            "--config",
            str(cfg_path),
            "--shard",
            "head",
        ]
        cmd_tail = [
            "uv",
            "run",
            "python",
            "scripts/run_node.py",
            "--config",
            str(cfg_path),
            "--shard",
            "tail",
        ]

        proc_head: subprocess.Popen[bytes]
        proc_tail: subprocess.Popen[bytes]
        with head_log.open("w") as fh, tail_log.open("w") as ft:
            proc_head = subprocess.Popen(
                cmd_head,
                env=env_head,
                cwd=str(_REPO_ROOT),
                stdout=fh,
                stderr=subprocess.STDOUT,
            )
            proc_tail = subprocess.Popen(
                cmd_tail,
                env=env_tail,
                cwd=str(_REPO_ROOT),
                stdout=ft,
                stderr=subprocess.STDOUT,
            )

        try:
            try:
                _wait_listening("127.0.0.1", port_head, timeout_s=300.0)
            except TimeoutError:
                pytest.fail(
                    f"head (MLX) never came up. Log:\n{head_log.read_text()}"
                )
            try:
                _wait_listening("127.0.0.1", port_tail, timeout_s=600.0)
            except TimeoutError:
                pytest.fail(
                    f"tail (PyTorch CPU) never came up. Log:\n{tail_log.read_text()}"
                )
            # Allow SWIM membership to stabilize.
            time.sleep(5.0)

            from model_shard.client import Client
            from model_shard.shard_map import NodeAddress

            client = Client(
                head_address=NodeAddress(host="127.0.0.1", port=port_head)
            )
            got = client.generate(prompt_tokens, max_new_tokens=_MAX_NEW_TOKENS)
            assert got == expected, (
                f"heterogeneous pipeline output {got!r} != "
                f"reference {expected!r} (prompt {_PROMPT_IDX})\n"
                f"--- head log ---\n{head_log.read_text()}\n"
                f"--- tail log ---\n{tail_log.read_text()}"
            )
        finally:
            for proc in (proc_head, proc_tail):
                proc.terminate()
                try:
                    proc.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5.0)
