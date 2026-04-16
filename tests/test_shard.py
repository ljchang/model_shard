"""Tests for LayerGroupShard metadata (no MLX involvement yet).

Run() lands in Week 2 with the MLX engine. Week 1 establishes only the
identity and range contract.
"""

import pytest

from model_shard.shard import LayerGroupShard


def test_layer_group_shard_basic_fields() -> None:
    shard = LayerGroupShard(start_layer=0, end_layer=10)
    assert shard.start_layer == 0
    assert shard.end_layer == 10


def test_layer_group_shard_id_convention() -> None:
    """Shard IDs use end-exclusive range notation: 'layer_{start}-{end}'."""
    assert LayerGroupShard(start_layer=0, end_layer=10).id == "layer_0-10"
    assert LayerGroupShard(start_layer=12, end_layer=24).id == "layer_12-24"
    assert LayerGroupShard(start_layer=24, end_layer=36).id == "layer_24-36"


def test_layer_group_shard_layer_count() -> None:
    assert LayerGroupShard(start_layer=0, end_layer=10).layer_count == 10
    assert LayerGroupShard(start_layer=12, end_layer=24).layer_count == 12


def test_layer_group_shard_contains() -> None:
    """A shard 'contains' the layers in its half-open [start, end) range."""
    shard = LayerGroupShard(start_layer=10, end_layer=20)
    assert not shard.contains(9)
    assert shard.contains(10)
    assert shard.contains(15)
    assert shard.contains(19)
    assert not shard.contains(20)


def test_layer_group_shard_rejects_inverted_range() -> None:
    with pytest.raises(ValueError, match="end_layer"):
        LayerGroupShard(start_layer=10, end_layer=5)


def test_layer_group_shard_rejects_empty_range() -> None:
    with pytest.raises(ValueError, match="end_layer"):
        LayerGroupShard(start_layer=10, end_layer=10)


def test_layer_group_shard_rejects_negative_start() -> None:
    with pytest.raises(ValueError, match="start_layer"):
        LayerGroupShard(start_layer=-1, end_layer=5)
