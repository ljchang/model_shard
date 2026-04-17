"""Tests for live-owners resolution via callback."""
from __future__ import annotations

import random
from unittest.mock import MagicMock

from model_shard.expert_orchestrator import ExpertOrchestrator
from model_shard.moe import group_expert_ids_by_owner_loaded


def test_live_owners_provider_augments_static_owners():
    static = {"A": {3}, "B": {7}}  # A owns 3, B owns 7 at this layer
    # C now also owns 7 (newly migrated replica).
    def provider(eid: int) -> set[str]:
        if eid == 3:
            return {"A"}
        if eid == 7:
            return {"B", "C"}
        return set()
    rng = random.Random(0)
    result = group_expert_ids_by_owner_loaded(
        top_k_ids=[3, 7],
        owners=static,
        peer_loads={"A": 100, "B": 500, "C": 10},
        self_shard_id="A",
        self_load=100,
        rng=rng,
        live_owners_provider=provider,
    )
    # Expert 7's P2C should pick C (lower load).
    assert "A" in result and result["A"] == [3]
    assert "C" in result and result["C"] == [7]


def test_live_owners_provider_default_preserves_phase4():
    static = {"A": {3}, "B": {7}}
    rng = random.Random(0)
    result = group_expert_ids_by_owner_loaded(
        top_k_ids=[3, 7],
        owners=static,
        peer_loads={"A": 100, "B": 500},
        self_shard_id="A",
        self_load=100,
        rng=rng,
    )
    # Single-owner per id: A gets 3, B gets 7.
    assert result == {"A": [3], "B": [7]}


def test_live_owners_provider_dedupes_overlap():
    # Both static owners and live provider return overlapping sets.
    static = {"A": {7}, "B": {7}}  # overlap — static already has 7 on both
    def provider(eid: int) -> set[str]:
        if eid == 7:
            return {"A"}  # already in static
        return set()
    rng = random.Random(0)
    result = group_expert_ids_by_owner_loaded(
        top_k_ids=[7],
        owners=static,
        peer_loads={"A": 100, "B": 500},
        self_shard_id="A",
        self_load=100,
        rng=rng,
        live_owners_provider=provider,
    )
    # Still only 2 candidates (A and B); should not double-count A.
    # P2C picks A (lower load).
    assert result == {"A": [7]}


def test_orchestrator_accepts_and_invokes_live_owners_provider():
    calls: list[int] = []
    def provider(eid: int) -> set[str]:
        calls.append(eid)
        return set()  # no extra owners
    orch = ExpertOrchestrator(
        self_shard_id="A",
        owners={"A": {3}, "B": {7}},
        peer_rpc=MagicMock(),
        rpc_timeout_s=1.0,
        live_owners_provider=provider,
    )
    assert orch.live_owners_provider is provider
    # Directly drive the grouping step with the orchestrator's provider.
    result = group_expert_ids_by_owner_loaded(
        top_k_ids=[3, 7],
        owners=orch.owners,
        peer_loads={},
        self_shard_id="A",
        self_load=0,
        rng=random.Random(0),
        live_owners_provider=orch.live_owners_provider,
    )
    assert sorted(calls) == [3, 7]
    assert result == {"A": [3], "B": [7]}
