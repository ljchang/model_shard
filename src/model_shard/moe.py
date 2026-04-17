"""Pure MoE helpers for expert-level sharding (Phase 3).

All functions in this module are pure — no threading, no I/O, no mlx evaluation
side effects beyond graph construction. They are composed by
ExpertOrchestrator for the network path and called directly by the split-
equivalence test for the correctness proof.
"""

from __future__ import annotations

from collections.abc import Mapping


def group_expert_ids_by_owner(
    top_k_ids: list[int],
    owners: Mapping[str, set[int]],
) -> dict[str, list[int]]:
    """Partition `top_k_ids` by which shard hosts each expert.

    Preserves per-shard order as ids appear in `top_k_ids`. Shards that own
    none of the ids are absent from the result (not empty-listed), so callers
    can iterate the dict without sending no-op RPCs.

    Raises KeyError if any id has no owner in `owners`.
    """
    id_to_owner: dict[int, str] = {}
    for owner, ids in owners.items():
        for i in ids:
            id_to_owner[i] = owner

    by_owner: dict[str, list[int]] = {}
    for eid in top_k_ids:
        try:
            owner = id_to_owner[eid]
        except KeyError as e:
            raise KeyError(f"expert_id {eid} has no owner in {list(owners)}") from e
        by_owner.setdefault(owner, []).append(eid)
    return by_owner


__all__ = ["group_expert_ids_by_owner"]
