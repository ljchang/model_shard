"""Fast unit tests for pure MoE helper functions."""

from __future__ import annotations

import random

import pytest

from model_shard.moe import group_expert_ids_by_owner, group_expert_ids_by_owner_loaded


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
    with pytest.raises(KeyError, match="expert_id 99"):
        group_expert_ids_by_owner([0, 99], owners)


def test_group_loaded_single_candidate_uses_sole_owner() -> None:
    owners = {"head": {0}, "mid": {1}, "tail": {2}}
    got = group_expert_ids_by_owner_loaded(
        top_k_ids=[0, 1, 2],
        owners=owners,
        peer_loads={"mid": 100, "tail": 100},
        self_shard_id="head",
        self_load=50,
        rng=random.Random(0),
    )
    assert got == {"head": [0], "mid": [1], "tail": [2]}


def test_group_loaded_two_candidates_picks_less_loaded() -> None:
    owners = {"head": {0, 1}, "mid": {0, 1}, "tail": {2}}
    got = group_expert_ids_by_owner_loaded(
        top_k_ids=[0, 1, 2],
        owners=owners,
        peer_loads={"mid": 1000},
        self_shard_id="head",
        self_load=10,
        rng=random.Random(0),
    )
    assert got["head"] == [0, 1]
    assert got["tail"] == [2]
    assert "mid" not in got


def test_group_loaded_three_candidates_samples_two_then_picks_less_loaded() -> None:
    owners = {"a": {0}, "b": {0}, "c": {0}}
    rng = random.Random(0)
    got = group_expert_ids_by_owner_loaded(
        top_k_ids=[0],
        owners=owners,
        peer_loads={"a": 100, "b": 50, "c": 10},
        self_shard_id="self_not_in_owners",
        self_load=0,
        rng=rng,
    )
    assert sum(1 for v in got.values() if v == [0]) == 1
    assert all(v == [0] for v in got.values())


def test_group_loaded_unknown_peer_treated_as_max_load() -> None:
    """When a candidate has no entry in peer_loads, it's treated as effectively
    infinite so the known candidate wins."""
    owners = {"head": {0}, "mid": {0}}
    got = group_expert_ids_by_owner_loaded(
        top_k_ids=[0],
        owners=owners,
        peer_loads={},
        self_shard_id="head",
        self_load=42,
        rng=random.Random(0),
    )
    assert got == {"head": [0]}


def test_group_loaded_unknown_self_and_peer_falls_back_to_rng() -> None:
    owners = {"a": {0}, "b": {0}}
    rng = random.Random(99)
    got = group_expert_ids_by_owner_loaded(
        top_k_ids=[0],
        owners=owners,
        peer_loads={},
        self_shard_id="not-a-candidate",
        self_load=0,
        rng=rng,
    )
    winners = [k for k, v in got.items() if v == [0]]
    assert len(winners) == 1


def test_group_loaded_raises_on_unknown_id() -> None:
    owners = {"head": {0}, "mid": {1}}
    with pytest.raises(KeyError, match="expert_id 99"):
        group_expert_ids_by_owner_loaded(
            top_k_ids=[99], owners=owners, peer_loads={},
            self_shard_id="head", self_load=0, rng=random.Random(0),
        )
