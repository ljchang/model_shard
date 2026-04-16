"""Tests for the static YAML-backed ShardMap.

In later phases this becomes gossip-driven. Phase 1's contract: load a YAML
file, look up shard_id -> ShardSpec (address + layer range), iterate shards.
"""

from pathlib import Path

import pytest

from model_shard.shard_map import NodeAddress, ShardMap, ShardSpec


def test_shard_map_from_dict_lookup_returns_full_spec() -> None:
    sm = ShardMap(
        {
            "layer_0-10": ShardSpec(
                shard_id="layer_0-10",
                address=NodeAddress(host="127.0.0.1", port=9001),
                start_layer=0,
                end_layer=10,
            ),
            "layer_10-20": ShardSpec(
                shard_id="layer_10-20",
                address=NodeAddress(host="127.0.0.1", port=9002),
                start_layer=10,
                end_layer=20,
            ),
        }
    )
    spec = sm.lookup("layer_0-10")
    assert spec.shard_id == "layer_0-10"
    assert spec.address == NodeAddress(host="127.0.0.1", port=9001)
    assert spec.start_layer == 0
    assert spec.end_layer == 10


def test_shard_map_missing_shard_raises_keyerror() -> None:
    sm = ShardMap(
        {
            "only": ShardSpec(
                shard_id="only",
                address=NodeAddress(host="h", port=1),
                start_layer=0,
                end_layer=5,
            )
        }
    )
    with pytest.raises(KeyError, match="missing"):
        sm.lookup("missing")


def test_shard_map_all_shards_returns_all_ids() -> None:
    def mk(sid: str, port: int, s: int, e: int) -> ShardSpec:
        return ShardSpec(
            shard_id=sid,
            address=NodeAddress(host="h", port=port),
            start_layer=s,
            end_layer=e,
        )

    sm = ShardMap(
        {
            "a": mk("a", 1, 0, 5),
            "b": mk("b", 2, 5, 10),
            "c": mk("c", 3, 10, 15),
        }
    )
    assert sorted(sm.all_shards()) == ["a", "b", "c"]


def test_shard_map_load_yaml_with_layer_ranges(tmp_path: Path) -> None:
    cfg = tmp_path / "shards.yaml"
    cfg.write_text(
        """
shards:
  layer_0-10:
    host: 127.0.0.1
    port: 9001
    start_layer: 0
    end_layer: 10
  layer_10-20:
    host: 127.0.0.1
    port: 9002
    start_layer: 10
    end_layer: 20
  layer_20-30:
    host: 127.0.0.1
    port: 9003
    start_layer: 20
    end_layer: 30
""".strip()
    )
    sm = ShardMap.from_yaml(cfg)

    head = sm.lookup("layer_0-10")
    mid = sm.lookup("layer_10-20")
    tail = sm.lookup("layer_20-30")

    assert head.start_layer == 0 and head.end_layer == 10
    assert mid.start_layer == 10 and mid.end_layer == 20
    assert tail.start_layer == 20 and tail.end_layer == 30
    assert head.address.port == 9001
    assert tail.address.port == 9003


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
    start_layer: 0
    end_layer: 5
""".strip()
    )
    with pytest.raises(ValueError, match="port"):
        ShardMap.from_yaml(cfg)


def test_shard_map_yaml_rejects_missing_layer_range(tmp_path: Path) -> None:
    cfg = tmp_path / "shards.yaml"
    cfg.write_text(
        """
shards:
  broken:
    host: 127.0.0.1
    port: 9001
""".strip()
    )
    with pytest.raises(ValueError, match="start_layer"):
        ShardMap.from_yaml(cfg)


def test_shard_map_yaml_rejects_bad_port_type(tmp_path: Path) -> None:
    cfg = tmp_path / "shards.yaml"
    cfg.write_text(
        """
shards:
  bad:
    host: 127.0.0.1
    port: not-a-number
    start_layer: 0
    end_layer: 5
""".strip()
    )
    with pytest.raises(ValueError):
        ShardMap.from_yaml(cfg)


def test_shard_map_yaml_rejects_inverted_layer_range(tmp_path: Path) -> None:
    cfg = tmp_path / "shards.yaml"
    cfg.write_text(
        """
shards:
  bad:
    host: 127.0.0.1
    port: 9001
    start_layer: 10
    end_layer: 5
""".strip()
    )
    with pytest.raises(ValueError, match="end_layer"):
        ShardMap.from_yaml(cfg)


def test_node_address_equality() -> None:
    a = NodeAddress(host="h", port=1)
    b = NodeAddress(host="h", port=1)
    c = NodeAddress(host="h", port=2)
    assert a == b
    assert a != c


def test_shard_spec_udp_port_is_tcp_port_plus_1000() -> None:
    from model_shard.shard_map import NodeAddress, ShardSpec
    spec = ShardSpec(
        shard_id="x",
        address=NodeAddress(host="127.0.0.1", port=9001),
        start_layer=0,
        end_layer=10,
    )
    assert spec.udp_port == 10001
