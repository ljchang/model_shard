from __future__ import annotations

from typing import Any

import mlx.core as mx
import pytest

from model_shard.moe import run_selected_experts


@pytest.mark.slow
def test_run_selected_experts_output_keyed_by_id(loaded_model: Any) -> None:
    lm = loaded_model
    h = mx.random.normal((1, 3, lm.text_model.config.hidden_size)).astype(mx.bfloat16)
    want = [3, 6, 126]
    out = run_selected_experts(lm, h, layer_idx=15, expert_ids=want)
    assert set(out.keys()) == set(want)
    for eid, tensor in out.items():
        mx.eval(tensor)
        assert tensor.shape == h.shape, f"expert {eid} shape mismatch"


@pytest.mark.slow
def test_run_selected_experts_empty_returns_empty(loaded_model: Any) -> None:
    lm = loaded_model
    h = mx.random.normal((1, 3, lm.text_model.config.hidden_size)).astype(mx.bfloat16)
    out = run_selected_experts(lm, h, layer_idx=15, expert_ids=[])
    assert out == {}
