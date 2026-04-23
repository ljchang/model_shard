"""Unit test that run_selected_experts translates global ids to local slots
when lm.held_ids_per_layer[layer_idx] is non-empty."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import pytest

from model_shard.mlx_engine import load_model_partial
from model_shard.moe import run_selected_experts


@pytest.mark.slow
def test_run_selected_experts_sliced_lm_returns_correct_outputs(loaded_model: Any, shards_model_id: str) -> None:
    """Given the same input h, run_selected_experts on a sliced model for a
    held id returns the same tensor as on the full model for that id."""
    lm_full = loaded_model
    held_ids = [0, 3, 6, 9]
    lm_part = load_model_partial(
        shards_model_id,
        {15: held_ids},
    )
    try:
        h = mx.random.normal((1, 3, lm_full.text_model.config.hidden_size)).astype(mx.bfloat16)
        out_full = run_selected_experts(lm_full, h, layer_idx=15, expert_ids=[3])
        out_part = run_selected_experts(lm_part, h, layer_idx=15, expert_ids=[3])
        mx.eval(out_full[3], out_part[3])
        assert mx.array_equal(out_full[3], out_part[3]), (
            f"bit-exact failure for expert 3; max abs diff = "
            f"{mx.max(mx.abs(out_full[3] - out_part[3])).item()}"
        )
    finally:
        del lm_part
        mx.metal.clear_cache()
