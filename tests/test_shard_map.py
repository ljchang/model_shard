"""Tests for the static YAML-backed ShardMap.

In later phases this becomes gossip-driven. Phase 1's contract: load a YAML
file, look up shard_id -> (host, port), iterate all shards.
"""

from pathlib import Path

import pytest

from model_shard.shard_map import NodeAddress, ShardMap


def test_shard_map_from_dict_lookup() -> None:
    sm = ShardMap(
        {
            "layer_0-10": NodeAddress(host="127.0.0.1", port=9001),
            "layer_10-20": NodeAddress(host="127.0.0.1", port=9002),
        }
    )
    assert sm.lookup("layer_0-10") == NodeAddress(host="127.0.0.1", port=9001)
    assert sm.lookup("layer_10-20") == NodeAddress(host="127.0.0.1", port=9002)


def test_shard_map_missing_shard_raises_keyerror() -> None:
    sm = ShardMap({"only": NodeAddress(host="h", port=1)})
    with pytest.raises(KeyError, match="missing"):
        sm.lookup("missing")


def test_shard_map_all_shards_returns_all_ids() -> None:
    sm = ShardMap(
        {
            "a": NodeAddress(host="h", port=1),
            "b": NodeAddress(host="h", port=2),
            "c": NodeAddress(host="h", port=3),
        }
    )
    assert sorted(sm.all_shards()) == ["a", "b", "c"]


def test_shard_map_load_yaml(tmp_path: Path) -> None:
    cfg = tmp_path / "shards.yaml"
    cfg.write_text(
        """
shards:
  layer_0-12:
    host: 127.0.0.1
    port: 9001
  layer_12-24:
    host: 127.0.0.1
    port: 9002
  layer_24-36:
    host: 127.0.0.1
    port: 9003
""".strip()
    )
    sm = ShardMap.from_yaml(cfg)
    assert sm.lookup("layer_0-12") == NodeAddress(host="127.0.0.1", port=9001)
    assert sm.lookup("layer_12-24") == NodeAddress(host="127.0.0.1", port=9002)
    assert sm.lookup("layer_24-36") == NodeAddress(host="127.0.0.1", port=9003)
    assert sorted(sm.all_shards()) == ["layer_0-12", "layer_12-24", "layer_24-36"]


def test_shard_map_yaml_requires_shards_key(tmp_path: Path) -> None:
    cfg = tmp_path / "shards.yaml"
    cfg.write_text("nodes: []\n")
    with pytest.raises(ValueError, match="shards"):
        ShardMap.from_yaml(cfg)


def test_shard_map_yaml_rejects_missing_host_or_port(tmp_path: Path) -> None:
    cfg = tmp_path / "shards.yaml"
    cfg.write_text(
        """
shards:
  broken:
    host: 127.0.0.1
""".strip()
    )
    with pytest.raises(ValueError, match="port"):
        ShardMap.from_yaml(cfg)


def test_shard_map_yaml_rejects_bad_port_type(tmp_path: Path) -> None:
    cfg = tmp_path / "shards.yaml"
    cfg.write_text(
        """
shards:
  bad:
    host: 127.0.0.1
    port: not-a-number
""".strip()
    )
    with pytest.raises(ValueError):
        ShardMap.from_yaml(cfg)


def test_node_address_equality() -> None:
    a = NodeAddress(host="h", port=1)
    b = NodeAddress(host="h", port=1)
    c = NodeAddress(host="h", port=2)
    assert a == b
    assert a != c
