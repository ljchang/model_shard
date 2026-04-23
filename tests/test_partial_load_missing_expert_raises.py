"""run_selected_experts must raise KeyError when given a global id
not in the shard's held_ids_per_layer[layer_idx]."""

from __future__ import annotations

import mlx.core as mx
import pytest

from model_shard.mlx_engine import load_model_partial
from model_shard.moe import run_selected_experts


@pytest.mark.slow
def test_run_selected_experts_unknown_global_id_raises(shards_model_id: str) -> None:
    held = {15: [0, 3, 6]}
    lm = load_model_partial(shards_model_id, held)
    try:
        h = mx.random.normal((1, 2, lm.text_model.config.hidden_size)).astype(mx.bfloat16)
        with pytest.raises(KeyError, match="expert 42 not held on this shard"):
            run_selected_experts(lm, h, layer_idx=15, expert_ids=[42])
    finally:
        del lm
        mx.metal.clear_cache()


@pytest.mark.slow
def test_run_selected_experts_held_layer_unaffected_elsewhere(shards_model_id: str) -> None:
    """If layer 15 is subset-loaded but layer 20 is not, requests for experts
    on layer 20 still work with any global id."""
    held = {15: [0, 3]}
    lm = load_model_partial(shards_model_id, held)
    try:
        h = mx.random.normal((1, 2, lm.text_model.config.hidden_size)).astype(mx.bfloat16)
        # Layer 20 has no slice; global id 99 is still valid (full stack).
        out = run_selected_experts(lm, h, layer_idx=20, expert_ids=[99])
        assert 99 in out
    finally:
        del lm
        mx.metal.clear_cache()
