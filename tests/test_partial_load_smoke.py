"""Smoke test: load_model_partial returns a LoadedModel with correct shape."""

from __future__ import annotations

import pytest

from model_shard.mlx_engine import load_model_partial


@pytest.mark.slow
def test_partial_load_slices_layer_experts(shards_model_id: str) -> None:
    held = {15: [0, 3, 6, 9]}
    lm = load_model_partial(shards_model_id, held)

    assert lm.num_layers == 30
    assert lm.held_ids_per_layer == {15: (0, 3, 6, 9)}

    layer15 = lm.text_model.layers[15]
    w = layer15.experts.switch_glu.gate_proj.weight
    # Held-layer weight has leading dim == len(held_ids).
    assert w.shape[0] == 4

    # A non-held layer retains full 128.
    layer0 = lm.text_model.layers[0]
    w0 = layer0.experts.switch_glu.gate_proj.weight
    assert w0.shape[0] == 128
