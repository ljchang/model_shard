"""Load-bearing correctness proof for Phase 3.

Runs layer 15 two ways on the same input and asserts bit-equality:
  (a) atomic:  layer(h, mask, cache)  — as Phase 1 does
  (b) split:   run_attention_and_route
               -> run_selected_experts on every expert in top-k
               -> run_shared_expert
               -> aggregate_experts per-position
               -> outer post_feedforward_layernorm + residual + layer_scalar

If (a) != (b), Phase 3 cannot reproduce Tier 1. Fix before proceeding.
"""

from __future__ import annotations

from typing import Any, cast

import mlx.core as mx
import pytest

from model_shard.mlx_engine import embed_tokens, make_cache, make_masks
from model_shard.moe import (
    aggregate_experts,
    run_attention_and_route,
    run_selected_experts,
    run_shared_expert,
)


@pytest.mark.slow
def test_layer15_split_equivalent_to_atomic(loaded_model: Any) -> None:
    lm = loaded_model
    layer_idx = 15
    tokens = mx.array([[1, 42, 99, 7, 13, 256, 500]])  # B=1, L=7

    # Atomic path (replay layers 0..14 to set up the right input + cache state).
    h_atom = embed_tokens(lm, tokens)
    cache_atom = make_cache(lm)
    gm, sm = make_masks(lm, h_atom, cache_atom)
    tm = lm.text_model
    for i in range(layer_idx):
        layer = tm.layers[i]
        c = cache_atom[tm.layer_idx_to_cache_idx[i]]
        mask = gm if layer.layer_type == "full_attention" else sm
        h_atom = layer(h_atom, mask, c, per_layer_input=None)
    layer15 = tm.layers[layer_idx]
    c15 = cache_atom[tm.layer_idx_to_cache_idx[layer_idx]]
    mask15 = gm if layer15.layer_type == "full_attention" else sm
    out_atomic = layer15(h_atom, mask15, c15, per_layer_input=None)

    # Split path — same tokens, fresh cache, replay layers 0..14 so the
    # input to layer 15 matches.
    h_split = embed_tokens(lm, tokens)
    cache_split = make_cache(lm)
    gm2, sm2 = make_masks(lm, h_split, cache_split)
    for i in range(layer_idx):
        layer = tm.layers[i]
        c = cache_split[tm.layer_idx_to_cache_idx[i]]
        mask = gm2 if layer.layer_type == "full_attention" else sm2
        h_split = layer(h_split, mask, c, per_layer_input=None)

    post_attn, top_k_ids, top_k_weights = run_attention_and_route(
        lm, h_split, layer_idx, cache_split, (gm2, sm2)
    )
    mx.eval(top_k_ids)
    all_ids = sorted(
        {int(eid) for eid in cast(list[int], top_k_ids.reshape(-1).tolist())}
    )
    expert_outputs = run_selected_experts(lm, post_attn, layer_idx, all_ids)
    shared_out = run_shared_expert(lm, post_attn, layer_idx)
    post_ffn_ln_2 = tm.layers[layer_idx].post_feedforward_layernorm_2

    # Per-position aggregation — matches orchestrator's per-position loop.
    h1_plus_h2 = mx.zeros_like(post_attn)
    for b in range(top_k_ids.shape[0]):
        for ll in range(top_k_ids.shape[1]):
            ids = [int(x) for x in cast(list[int], top_k_ids[b, ll].tolist())]
            weights = top_k_weights[b : b + 1, ll : ll + 1, :]
            per_pos_outs = {
                eid: expert_outputs[eid][b : b + 1, ll : ll + 1, :] for eid in ids
            }
            per_pos_shared = shared_out[b : b + 1, ll : ll + 1, :]
            agg = aggregate_experts(
                per_pos_outs, ids, weights, per_pos_shared, post_ffn_ln_2
            )
            h1_plus_h2 = (
                mx.concatenate(
                    [h1_plus_h2[:, :ll, :], agg, h1_plus_h2[:, ll + 1 :, :]], axis=1
                )
                if h1_plus_h2.shape[1] > 1
                else agg
            )

    # Outer ops from DecoderLayer.__call__: post_feedforward_layernorm,
    # second residual, layer_scalar. Per-layer-input gating skipped for 26B.
    layer = tm.layers[layer_idx]
    out_split = layer.post_feedforward_layernorm(h1_plus_h2)
    out_split = post_attn + out_split
    if layer.layer_scalar is not None:
        out_split = out_split * layer.layer_scalar

    mx.eval(out_atomic, out_split)
    max_diff = mx.max(mx.abs(out_atomic - out_split)).item()
    assert mx.array_equal(out_atomic, out_split), (
        f"split != atomic; max abs diff = {max_diff}"
    )
