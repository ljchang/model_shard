from __future__ import annotations

import mlx.core as mx
import pytest

from model_shard.moe import run_shared_expert


@pytest.mark.slow
def test_shared_expert_output_has_correct_shape(loaded_model) -> None:
    lm = loaded_model
    h = mx.random.normal((1, 3, lm.text_model.config.hidden_size))
    out = run_shared_expert(lm, h, layer_idx=15)
    mx.eval(out)
    assert out.shape == h.shape


@pytest.mark.slow
def test_shared_expert_deterministic(loaded_model) -> None:
    lm = loaded_model
    h = mx.random.normal((1, 3, lm.text_model.config.hidden_size))
    out1 = run_shared_expert(lm, h, layer_idx=15)
    out2 = run_shared_expert(lm, h, layer_idx=15)
    mx.eval(out1, out2)
    assert mx.all(out1 == out2).item()
