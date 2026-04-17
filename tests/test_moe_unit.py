"""Fast unit tests for pure MoE helper functions."""

from __future__ import annotations

from model_shard.moe import group_expert_ids_by_owner


def test_group_expert_ids_by_owner_round_robin_mod3() -> None:
    owners = {
        "head": {0, 3, 6, 9, 126},
        "mid":  {1, 4, 7, 127},
        "tail": {2, 5, 8, 125},
    }
    top_k = [3, 7, 5, 1, 126, 2, 9, 127]

    got = group_expert_ids_by_owner(top_k, owners)

    assert got["head"] == [3, 126, 9]   # order preserves appearance in top_k
    assert got["mid"]  == [7, 1, 127]
    assert got["tail"] == [5, 2]


def test_group_expert_ids_by_owner_empty_owner_absent() -> None:
    owners = {"head": {0}, "mid": {1}, "tail": {2}}
    got = group_expert_ids_by_owner([0, 0], owners)
    assert got == {"head": [0, 0]}  # mid and tail absent, not empty lists


def test_group_expert_ids_by_owner_unknown_id_raises() -> None:
    owners = {"head": {0}, "mid": {1}}
    import pytest
    with pytest.raises(KeyError, match="expert_id 99"):
        group_expert_ids_by_owner([0, 99], owners)
