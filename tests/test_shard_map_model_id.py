"""Phase 7-C-3a Task 2: ShardMap.model_id field tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from model_shard.shard_map import ShardMap


def _write_yaml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "shards.yaml"
    p.write_text(body)
    return p


def test_shard_map_exposes_model_id(tmp_path: Path) -> None:
    cfg = _write_yaml(
        tmp_path,
        """
model_id: "/tmp/fake-bf16-model"
shards:
  head:
    host: 127.0.0.1
    port: 9001
    start_layer: 0
    end_layer: 30
""",
    )
    sm = ShardMap.from_yaml(cfg)
    assert sm.model_id == "/tmp/fake-bf16-model"


def test_shard_map_model_id_defaults_to_empty_when_absent(tmp_path: Path) -> None:
    """Backwards compat during migration: missing model_id parses but
    yields empty string. Task 12 will flip this to required."""
    cfg = _write_yaml(
        tmp_path,
        """
shards:
  head:
    host: 127.0.0.1
    port: 9001
    start_layer: 0
    end_layer: 30
""",
    )
    sm = ShardMap.from_yaml(cfg)
    assert sm.model_id == ""


def test_shard_map_model_id_must_be_string(tmp_path: Path) -> None:
    cfg = _write_yaml(
        tmp_path,
        """
model_id: 42
shards:
  head:
    host: 127.0.0.1
    port: 9001
    start_layer: 0
    end_layer: 30
""",
    )
    with pytest.raises(ValueError, match="model_id"):
        ShardMap.from_yaml(cfg)
