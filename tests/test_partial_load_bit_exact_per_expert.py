"""Bit-exact per-expert equivalence between full-loaded and sliced model."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import pytest

from model_shard.mlx_engine import load_model_partial
from model_shard.moe import run_selected_experts


@pytest.mark.slow
def test_every_held_expert_matches_full_model(loaded_model: Any, shards_model_id: str) -> None:
    lm_full = loaded_model
    held_ids = [0, 3, 6, 9, 12, 15, 42, 127]
    lm_part = load_model_partial(
        shards_model_id,
        {15: held_ids},
    )
    try:
        h = mx.random.normal(
            (1, 5, lm_full.text_model.config.hidden_size)
        ).astype(mx.bfloat16)

        out_full = run_selected_experts(lm_full, h, layer_idx=15, expert_ids=held_ids)
        out_part = run_selected_experts(lm_part, h, layer_idx=15, expert_ids=held_ids)

        for eid in held_ids:
            mx.eval(out_full[eid], out_part[eid])
            assert mx.array_equal(out_full[eid], out_part[eid]), (
                f"expert {eid}: bit-exact failure; max abs diff = "
                f"{mx.max(mx.abs(out_full[eid] - out_part[eid])).item()}"
            )
    finally:
        del lm_part
        mx.metal.clear_cache()
