"""MLX forward-pass building blocks for Gemma 4 26B A4B.

This module is the single source of truth for how a forward pass is composed.
Both the reference oracle and each sharded node run the same functions —
they only differ in which layer range they execute.

Gemma 4 26B A4B specifics baked in:
  * num_layers = 30
  * No per-layer embeddings (hidden_size_per_layer_input = 0), so
    per_layer_input is always None.
  * num_kv_shared_layers = 0, so cache has one slot per layer.
  * tie_word_embeddings = True, so LM head reuses embed_tokens.as_linear.
  * final_logit_softcapping = 30.0.
"""

from dataclasses import dataclass
from functools import partial
from typing import Any

import mlx.core as mx


@dataclass
class LoadedModel:
    """Thin handle over mlx-vlm's loaded Gemma 4 model."""

    mlx_model: Any           # top-level mlx_vlm Model
    language_model: Any      # LanguageModel wrapper (exposes make_cache, softcap)
    text_model: Any          # Gemma4TextModel (exposes layers, embed_tokens, norm)
    processor: Any           # tokenizer/processor
    num_layers: int


@partial(mx.compile, shapeless=True)
def _softcap(softcap: float, x: mx.array) -> mx.array:
    return mx.tanh(x / softcap) * softcap


def load_model(hf_id: str) -> LoadedModel:
    from mlx_vlm import load

    model, processor = load(hf_id)
    language_model = model.language_model
    text_model = language_model.model
    return LoadedModel(
        mlx_model=model,
        language_model=language_model,
        text_model=text_model,
        processor=processor,
        num_layers=len(text_model.layers),
    )


def embed_tokens(lm: LoadedModel, token_ids: mx.array) -> mx.array:
    """Embedding lookup + Gemma's sqrt(hidden_size) scale."""
    h = lm.text_model.embed_tokens(token_ids)
    return h * lm.text_model.embed_scale  # type: ignore[no-any-return]


def make_cache(lm: LoadedModel) -> list[Any]:
    """Per-layer KV caches (KVCache for full-attention, RotatingKVCache for sliding)."""
    return list(lm.language_model.make_cache())


def make_masks(lm: LoadedModel, h: mx.array, cache: list[Any]) -> tuple[Any, Any]:
    """Reconstruct the global and sliding masks for the current step.

    Masks are deterministic given the hidden-state shape and the cache offsets;
    each shard can rebuild them from shared state without cross-shard traffic.
    """
    from mlx_vlm.models.base import create_attention_mask

    tm = lm.text_model
    first_full = tm.first_full_cache_idx
    first_sliding = tm.first_sliding_cache_idx

    global_mask = create_attention_mask(
        h,
        cache[first_full] if first_full < len(cache) else None,
    )
    sliding_mask = create_attention_mask(
        h,
        cache[first_sliding] if first_sliding < len(cache) else None,
        window_size=tm.window_size,
    )
    return global_mask, sliding_mask


def run_layers(
    lm: LoadedModel,
    h: mx.array,
    start_layer: int,
    end_layer: int,
    cache: list[Any],
    global_mask: Any,
    sliding_mask: Any,
) -> mx.array:
    """Run transformer layers in the half-open range [start_layer, end_layer).

    Mutates `cache` in place (MLX KV caches update on each call). The caller
    owns cache lifetime — in Phase 1 that's the node hosting the layer.
    """
    tm = lm.text_model
    for i in range(start_layer, end_layer):
        layer = tm.layers[i]
        c = cache[tm.layer_idx_to_cache_idx[i]]
        mask = global_mask if layer.layer_type == "full_attention" else sliding_mask
        h = layer(h, mask, c, per_layer_input=None)
    return h


def finalize(lm: LoadedModel, h: mx.array) -> mx.array:
    """Final RMSNorm + tied LM head + logit softcap. Produces [B, L, vocab_size]."""
    h = lm.text_model.norm(h)
    logits = lm.text_model.embed_tokens.as_linear(h)
    softcap = lm.language_model.final_logit_softcapping
    if softcap is not None:
        logits = _softcap(softcap, logits)
    return logits  # type: ignore[no-any-return]
