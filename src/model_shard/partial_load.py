"""Partial expert-weight loading for Phase 5a.

A shard can declare which routed experts it holds per layer (via
ShardSpec.moe_experts). This module provides a custom safetensors reader
that slices the stacked (128, out, in) expert projection tensors at load
time so the shard's resident memory contains only the held experts'
weights.

Chassis weights (attention, dense mlp, norms, embeddings, LM head, router)
load unchanged on every node.
"""

from __future__ import annotations

import logging

import mlx.core as mx
import numpy as np

from model_shard.mlx_engine import LoadedModel

_LOG = logging.getLogger(__name__)


def _slice_stacked_by_axis0(
    arr: np.ndarray, ids: list[int]
) -> np.ndarray:
    """Return the rows of `arr` at positions `ids` along axis 0.

    Order is preserved: the returned array's row `i` is `arr[ids[i]]`.
    Raises IndexError or ValueError if any id is out of bounds.
    """
    if not ids:
        return arr[0:0]
    return arr[ids]


def load_model_partial(
    hf_id: str,
    held_experts_per_layer: dict[int, list[int]],
) -> LoadedModel:
    """Load Gemma 4 26B with routed-expert weights restricted to held subset per layer.

    Layers absent from `held_experts_per_layer` load the full 128-expert stack
    (same as `load_model`). Chassis weights (attention, dense mlp, norms,
    embeddings, LM head, router) always load fully.

    Strategy: use mlx-vlm's standard `load()` to construct the full model
    normally (peak memory blip ~14 GB), then iterate the held layers and
    replace each layer's `experts.switch_glu.<proj>.{weight, scales, biases}`
    with a compact (k, ...) tensor sliced along axis 0. Calls
    `mx.metal.clear_cache()` at the end so the full stacked tensors are
    eligible for release.
    """
    from mlx_vlm import load as _mlx_vlm_load

    model, processor = _mlx_vlm_load(hf_id)
    language_model = model.language_model
    text_model = language_model.model
    num_layers = len(text_model.layers)

    for layer_idx, ids in held_experts_per_layer.items():
        if not ids:
            continue
        layer = text_model.layers[layer_idx]
        switch_glu = layer.experts.switch_glu
        held_arr = mx.array(list(ids))
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            proj = getattr(switch_glu, proj_name)
            for attr in ("weight", "scales", "biases"):
                if not hasattr(proj, attr):
                    continue
                full = getattr(proj, attr)
                if full is None:
                    continue
                held = mx.take(full, held_arr, axis=0)
                setattr(proj, attr, held)
        _LOG.info(
            "partial_load: layer %d sliced to %d experts (from 128)",
            layer_idx,
            len(ids),
        )

    # Release the full-stacked tensors that are no longer referenced.
    mx.metal.clear_cache()

    held_ids_norm: dict[int, tuple[int, ...]] = {
        k: tuple(v) for k, v in held_experts_per_layer.items()
    }

    return LoadedModel(
        mlx_model=model,
        language_model=language_model,
        text_model=text_model,
        processor=processor,
        num_layers=num_layers,
        held_ids_per_layer=held_ids_norm,
    )


__all__ = ["_slice_stacked_by_axis0", "load_model_partial"]
