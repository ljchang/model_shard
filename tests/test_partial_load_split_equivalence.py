"""Load-bearing Phase 5a proof:

Three sliced LoadedModels (mod-3 at layer 15) + one full LoadedModel.
For each token, top-k ids are partitioned by mod-3 owner, each sliced
LM computes its share of expert outputs, aggregation runs, outer ops run.
Result must match atomic layer 15 on the full model bit-for-bit.
"""

from __future__ import annotations

from typing import Any, cast

import mlx.core as mx
import pytest

from model_shard.mlx_engine import embed_tokens, load_model_partial, make_cache, make_masks
from model_shard.moe import (
    aggregate_experts,
    run_attention_and_route,
    run_selected_experts,
    run_shared_expert,
)


def _ids_mod3(r: int) -> list[int]:
    return [e for e in range(128) if e % 3 == r]


@pytest.mark.slow
def test_three_sliced_shards_compose_bit_exact(loaded_model: Any, shards_model_id: str) -> None:
    lm_full = loaded_model
    lm_shards = [
        load_model_partial(shards_model_id, {15: _ids_mod3(0)}),
        load_model_partial(shards_model_id, {15: _ids_mod3(1)}),
        load_model_partial(shards_model_id, {15: _ids_mod3(2)}),
    ]
    try:
        layer_idx = 15
        tokens = mx.array([[1, 42, 99, 7, 13, 256, 500]])

        # Atomic on the full model: replay layers 0..14, then layer 15 atomically.
        h_atom = embed_tokens(lm_full, tokens)
        cache_atom = make_cache(lm_full)
        gm, sm = make_masks(lm_full, h_atom, cache_atom)
        tm = lm_full.text_model
        for i in range(layer_idx):
            layer = tm.layers[i]
            c = cache_atom[tm.layer_idx_to_cache_idx[i]]
            mask = gm if layer.layer_type == "full_attention" else sm
            h_atom = layer(h_atom, mask, c, per_layer_input=None)
        layer15 = tm.layers[layer_idx]
        c15 = cache_atom[tm.layer_idx_to_cache_idx[layer_idx]]
        mask15 = gm if layer15.layer_type == "full_attention" else sm
        out_atomic = layer15(h_atom, mask15, c15, per_layer_input=None)

        # Split across 3 sliced shards. The "router shard" holds the attention
        # state for this layer; use lm_shards[1] as the router-equivalent.
        lm_router = lm_shards[1]
        h_split = embed_tokens(lm_router, tokens)
        cache_split = make_cache(lm_router)
        gm2, sm2 = make_masks(lm_router, h_split, cache_split)
        for i in range(layer_idx):
            layer = lm_router.text_model.layers[i]
            c = cache_split[lm_router.text_model.layer_idx_to_cache_idx[i]]
            mask = gm2 if layer.layer_type == "full_attention" else sm2
            h_split = layer(h_split, mask, c, per_layer_input=None)

        post_attn, top_k_ids, top_k_weights = run_attention_and_route(
            lm_router, h_split, layer_idx, cache_split, (gm2, sm2)
        )
        mx.eval(top_k_ids)

        # Collect expert outputs across the 3 shards.
        all_ids = sorted(
            {int(eid) for eid in cast(list[int], top_k_ids.reshape(-1).tolist())}
        )
        expert_outputs: dict[int, mx.array] = {}
        for shard_lm in lm_shards:
            held = set(shard_lm.held_ids_per_layer.get(layer_idx, ()))
            mine = [e for e in all_ids if e in held]
            if not mine:
                continue
            contribution = run_selected_experts(shard_lm, post_attn, layer_idx, mine)
            expert_outputs.update(contribution)

        shared_out = run_shared_expert(lm_router, post_attn, layer_idx)
        post_ffn_ln_2 = lm_router.text_model.layers[layer_idx].post_feedforward_layernorm_2

        # Per-position aggregation — matches Phase 3 Task 9's pattern.
        h1_plus_h2 = mx.zeros_like(post_attn)
        for b in range(top_k_ids.shape[0]):
            for ll in range(top_k_ids.shape[1]):
                ids_l = [
                    int(x) for x in cast(list[int], top_k_ids[b, ll].tolist())
                ]
                weights = top_k_weights[b : b + 1, ll : ll + 1, :]
                per_pos = {
                    eid: expert_outputs[eid][b : b + 1, ll : ll + 1, :] for eid in ids_l
                }
                per_pos_shared = shared_out[b : b + 1, ll : ll + 1, :]
                agg = aggregate_experts(
                    per_pos, ids_l, weights, per_pos_shared, post_ffn_ln_2
                )
                h1_plus_h2 = mx.concatenate(
                    [h1_plus_h2[:, :ll, :], agg, h1_plus_h2[:, ll + 1 :, :]],
                    axis=1,
                ) if h1_plus_h2.shape[1] > 1 else agg

        # Outer layer ops from DecoderLayer.__call__
        layer_router = lm_router.text_model.layers[layer_idx]
        out_split = layer_router.post_feedforward_layernorm(h1_plus_h2)
        out_split = post_attn + out_split
        if layer_router.layer_scalar is not None:
            out_split = out_split * layer_router.layer_scalar

        mx.eval(out_atomic, out_split)
        assert mx.array_equal(out_atomic, out_split), (
            f"sliced-split != atomic; max abs diff = "
            f"{mx.max(mx.abs(out_atomic - out_split)).item()}"
        )
    finally:
        for lm in lm_shards:
            del lm
        mx.metal.clear_cache()
