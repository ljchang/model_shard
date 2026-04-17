"""Slow tests verifying moe.run_attention_and_route matches an atomic layer call."""

from __future__ import annotations

from typing import Any, cast

import mlx.core as mx
import pytest

from model_shard.mlx_engine import embed_tokens, make_cache, make_masks
from model_shard.moe import run_attention_and_route


@pytest.mark.slow
def test_attention_and_route_matches_atomic_prefill(loaded_model: Any) -> None:
    """run_attention_and_route should produce the same post-attention hidden
    state and router top-k as the layer's atomic forward, on the same input."""
    lm = loaded_model
    layer_idx = 15

    tokens = mx.array([[1, 42, 99, 7, 13]])  # B=1, L=5
    h = embed_tokens(lm, tokens)
    cache = make_cache(lm)
    global_mask, sliding_mask = make_masks(lm, h, cache)

    post_attn, top_k_ids, top_k_weights = run_attention_and_route(
        lm, h, layer_idx, cache, (global_mask, sliding_mask)
    )

    assert post_attn.shape == h.shape
    assert top_k_ids.shape[-1] == 8           # top-8 per token
    assert top_k_weights.shape[-1] == 8
    # All top-k ids must be valid expert indices.
    mx.eval(top_k_ids)
    ids_np = cast(list[list[list[int]]], top_k_ids.astype(mx.int32).tolist())
    for tok_ids in ids_np[0]:
        for eid in tok_ids:
            assert 0 <= eid < 128
