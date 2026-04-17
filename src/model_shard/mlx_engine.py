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

from dataclasses import dataclass, field
from functools import partial
from typing import Any

import mlx.core as mx
import numpy as np

from model_shard._pb import wire_pb2


@dataclass
class LoadedModel:
    """Thin handle over mlx-vlm's loaded Gemma 4 model."""

    mlx_model: Any           # top-level mlx_vlm Model
    language_model: Any      # LanguageModel wrapper (exposes make_cache, softcap)
    text_model: Any          # Gemma4TextModel (exposes layers, embed_tokens, norm)
    processor: Any           # tokenizer/processor
    num_layers: int
    # Phase 5a: per-layer tuple of held routed-expert global ids. Empty dict
    # (or an absent layer_idx key) means that layer holds all 128 experts.
    held_ids_per_layer: dict[int, tuple[int, ...]] = field(default_factory=dict)


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
    split_layers: set[int] | None = None,
    orchestrator: Any = None,
    request_id: str = "",
    provenance_chain: list[Any] | None = None,
    node_id: str = "",
) -> mx.array:
    """Run transformer layers in the half-open range [start_layer, end_layer).

    Mutates `cache` in place (MLX KV caches update on each call). The caller
    owns cache lifetime — in Phase 1 that's the node hosting the layer.

    For ``i in split_layers``, delegate to ``orchestrator.run_split_layer``.
    All other layers run atomically (Phase 1 behavior). ``split_layers=None``
    is equivalent to an empty set.

    ``provenance_chain`` and ``node_id`` are optional Phase 6-B kwargs.
    When ``provenance_chain`` is not None, an OP_LAYER_ATOMIC entry is
    appended for each atomic (non-split) layer after execution. Existing
    callers that omit these kwargs are unaffected.
    """
    tm = lm.text_model
    split = split_layers or set()
    for i in range(start_layer, end_layer):
        if i in split:
            if orchestrator is None:
                raise ValueError(
                    f"layer {i} is split but no orchestrator given"
                )
            h = orchestrator.run_split_layer(
                lm,
                h=h,
                layer_idx=i,
                cache=cache,
                masks=(global_mask, sliding_mask),
                request_id=request_id,
                provenance_chain=provenance_chain,
            )
        else:
            layer = tm.layers[i]
            c = cache[tm.layer_idx_to_cache_idx[i]]
            mask = global_mask if layer.layer_type == "full_attention" else sliding_mask
            h = layer(h, mask, c, per_layer_input=None)
            if provenance_chain is not None:
                from model_shard.provenance import build_entry
                from model_shard.request import OpDescriptor, OpType
                prev_hash = provenance_chain[-1].hash if provenance_chain else b""
                provenance_chain.append(
                    build_entry(
                        node_id=node_id,
                        op=OpDescriptor(op_type=OpType.OP_LAYER_ATOMIC, layer_idx=i),
                        output_tensor=h,
                        parent_hashes=(prev_hash,) if prev_hash else (),
                    )
                )
    return h


def finalize(lm: LoadedModel, h: mx.array) -> mx.array:
    """Final RMSNorm + tied LM head + logit softcap. Produces [B, L, vocab_size]."""
    h = lm.text_model.norm(h)
    logits = lm.text_model.embed_tokens.as_linear(h)
    softcap = lm.language_model.final_logit_softcapping
    if softcap is not None:
        logits = _softcap(softcap, logits)
    return logits  # type: ignore[no-any-return]


# dtype enum (proto) → (mx dtype, numpy staging dtype, bytes-per-element)
_DTYPE_MAP: dict[int, tuple[mx.Dtype, np.dtype, int]] = {
    wire_pb2.DTYPE_FLOAT32: (mx.float32, np.dtype("float32"), 4),
    wire_pb2.DTYPE_FLOAT16: (mx.float16, np.dtype("float16"), 2),
    wire_pb2.DTYPE_BFLOAT16: (mx.bfloat16, np.dtype("uint16"), 2),  # staged as uint16
    wire_pb2.DTYPE_INT32: (mx.int32, np.dtype("int32"), 4),
    wire_pb2.DTYPE_INT8: (mx.int8, np.dtype("int8"), 1),
    wire_pb2.DTYPE_UINT8: (mx.uint8, np.dtype("uint8"), 1),
    wire_pb2.DTYPE_UINT32: (mx.uint32, np.dtype("uint32"), 4),  # NEW
}


def _mx_to_wire_dtype(dtype: mx.Dtype) -> int:
    for wire, (mxt, _, _) in _DTYPE_MAP.items():
        if mxt == dtype:
            return wire
    raise ValueError(f"unsupported mx dtype for wire: {dtype}")


def tensor_to_bytes(arr: mx.array) -> bytes:
    """Serialize an mx.array to raw bytes (no shape/dtype metadata — those go
    in the accompanying TensorDescriptor).

    bf16 is staged through uint16 because numpy's buffer protocol doesn't
    handle bf16 directly.
    """
    staged = np.array(arr.view(mx.uint16)) if arr.dtype == mx.bfloat16 else np.array(arr)
    return staged.tobytes()


def bytes_to_tensor(raw: bytes, shape: list[int], dtype: int) -> mx.array:
    """Rehydrate an mx.array from raw bytes + the TensorDescriptor metadata."""
    if dtype not in _DTYPE_MAP:
        raise ValueError(f"unsupported wire dtype: {dtype}")
    mx_dtype, np_stage, _ = _DTYPE_MAP[dtype]
    staged = np.frombuffer(raw, dtype=np_stage).reshape(shape)
    arr = mx.array(staged)
    if mx_dtype == mx.bfloat16:
        arr = arr.view(mx.bfloat16)
    return arr


def load_model_partial(
    hf_id: str,
    held_experts_per_layer: dict[int, list[int]],
) -> LoadedModel:
    """Phase 5a wrapper. See partial_load.load_model_partial for semantics."""
    from model_shard.partial_load import load_model_partial as _impl
    return _impl(hf_id, held_experts_per_layer)
