"""Pure MoE helpers for expert-level sharding (Phase 3).

All functions in this module are pure — no threading, no I/O, no mlx evaluation
side effects beyond graph construction. They are composed by
ExpertOrchestrator for the network path and called directly by the split-
equivalence test for the correctness proof.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import mlx.core as mx


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


def run_attention_and_route(
    lm: Any,
    h: mx.array,
    layer_idx: int,
    cache: list[Any],
    masks: tuple[Any, Any],
) -> tuple[mx.array, mx.array, mx.array]:
    """Run attention + LN + router for one Gemma4 decoder layer.

    Returns the post-attention hidden state (input to the MoE block's two
    parallel branches) and the router's top-k expert ids / weights. Does not
    run any experts — the caller feeds ids/weights into fan-out and
    ``aggregate_experts``.

    Mirrors the first half of mlx-vlm ``DecoderLayer.__call__`` (gemma4
    language.py): ``residual + post_attention_layernorm(self_attn(
    input_layernorm(x), mask, cache))``. The router is then called on this
    tensor directly — it has its own internal RMSNorm, so we do not pre-norm.

    The returned ``top_k_indices`` / ``top_k_weights`` are whatever the
    ``Router`` module produces: weights are already L1-renormalized and
    multiplied by ``per_expert_scale[top_k_indices]``. Downstream code must
    not re-softmax or re-normalize them.
    """
    tm = lm.text_model
    layer = tm.layers[layer_idx]
    global_mask, sliding_mask = masks
    mask = global_mask if layer.layer_type == "full_attention" else sliding_mask
    c = cache[tm.layer_idx_to_cache_idx[layer_idx]]

    # Attention sub-block (verified against mlx_vlm.models.gemma4.language.
    # DecoderLayer.__call__): input_layernorm -> self_attn -> post_attention_layernorm
    # with a residual from x.
    residual = h
    x = layer.input_layernorm(h)
    x = layer.self_attn(x, mask, c)
    x = layer.post_attention_layernorm(x)
    post_attn = residual + x

    # Router lives directly on the DecoderLayer (layer.router), not under
    # layer.mlp. It returns (top_k_indices, top_k_weights) already scaled.
    top_k_ids, top_k_weights = layer.router(post_attn)
    return post_attn, top_k_ids, top_k_weights


__all__ = ["group_expert_ids_by_owner", "run_attention_and_route"]
