"""Phase 7-B: PyTorch engine primitives for Gemma 4 26B A4B (Mixture-of-Experts).

Mirror of mlx_engine.py. Each function takes the HF model as first arg rather
than a LoadedModel struct — HF models carry their own state.

The ``run_layer_atomic`` / ``run_layers`` path is the non-split
atomic-layer forward. Split-layer MoE fan-out lives in ``pt_moe.py``
(analog of ``moe.py``).
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from model_shard._pb import wire_pb2

# ---- dtype mapping -----------------------------------------------------

_TORCH_TO_WIRE: dict[torch.dtype, int] = {
    torch.bfloat16: wire_pb2.DTYPE_BFLOAT16,
    torch.float16: wire_pb2.DTYPE_FLOAT16,
    torch.float32: wire_pb2.DTYPE_FLOAT32,
}

_WIRE_TO_TORCH: dict[int, torch.dtype] = {v: k for k, v in _TORCH_TO_WIRE.items()}


def torch_to_wire_dtype(dtype: torch.dtype) -> int:
    try:
        return _TORCH_TO_WIRE[dtype]
    except KeyError:
        raise ValueError(f"unsupported torch dtype for wire: {dtype}") from None


def _wire_to_torch_dtype(wire: int) -> torch.dtype:
    try:
        return _WIRE_TO_TORCH[wire]
    except KeyError:
        raise ValueError(f"unsupported wire dtype: {wire}") from None


def _text_model(model: Any) -> Any:
    """Return the ``Gemma4TextModel`` sub-component of a loaded HF model.

    Handles both topology variants:

    - **Text-only** (tiny synthetic test with ``Gemma4TextConfig``):
      ``model.model`` IS the ``Gemma4TextModel`` directly.
    - **Multimodal wrapper** (real ``google/gemma-4-26B-A4B-it`` loaded via
      ``AutoModelForCausalLM.from_pretrained`` with a ``Gemma4Config``):
      ``model.model`` is a ``Gemma4Model`` that nests the text model at
      ``.language_model``.

    Using ``getattr(inner, "language_model", inner)`` returns the wrapper's
    inner text model when present, otherwise the already-text-model node.
    """
    inner = model.model
    return getattr(inner, "language_model", inner)


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


# ---- model loading -----------------------------------------------------

def load_model(
    hf_id: str,
    device: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> Any:
    """Load a Gemma 4 HF model. bf16 on CUDA/CPU, fp16 on MPS."""
    from transformers import AutoModelForCausalLM
    device = device or _default_device()
    if device == "mps" and dtype == torch.bfloat16:
        dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=dtype, device_map=device,
    )
    model.eval()  # type: ignore[no-untyped-call]
    return model


# ---- primitives --------------------------------------------------------

def embed_tokens(model: Any, token_ids: list[int]) -> torch.Tensor:
    """Return [1, L, H] hidden states from token embeddings."""
    device = next(model.parameters()).device
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out: torch.Tensor = _text_model(model).embed_tokens(input_ids)
        return out


def make_cache(model: Any) -> Any:
    """Construct a fresh DynamicCache for one request."""
    from transformers import DynamicCache
    return DynamicCache()


def _resolve_layer_type(model: Any, layer_idx: int) -> str:
    """Resolve the layer type string for ``layer_idx``.

    HF's ``Gemma4TextDecoderLayer`` does NOT expose ``layer_type`` as an
    attribute — only the inner ``self_attn`` does. The authoritative
    source is ``config.layer_types[layer_idx]``. For ``Gemma4Config`` that
    wraps ``Gemma4TextConfig`` we call ``get_text_config()`` to unwrap.
    Falls back to ``"full_attention"`` if the config has no ``layer_types``.
    """
    layer = _text_model(model).layers[layer_idx]
    # Synthetic test stubs set layer.layer_type directly — honor that first.
    lt = getattr(layer, "layer_type", None)
    if isinstance(lt, str):
        return lt
    # Real HF layers expose layer_type on self_attn (Gemma4TextAttention).
    lt = getattr(getattr(layer, "self_attn", None), "layer_type", None)
    if isinstance(lt, str):
        return lt
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "get_text_config"):
        config = config.get_text_config()
    layer_types = list(getattr(config, "layer_types", [])) if config is not None else []
    if 0 <= layer_idx < len(layer_types):
        return str(layer_types[layer_idx])
    return "full_attention"


def make_masks(model: Any, h: torch.Tensor, cache: Any) -> tuple[Any, Any]:
    """Compute per-layer-type rotary position embeddings for the forward pass.

    Phase 7-C-1: Gemma 4's rotary embedding is per-layer-type (different
    inv_freq for "full_attention" vs "sliding_attention"). We compute once
    per unique layer type and return a dict keyed by layer_type string.
    The Backend protocol's ``masks: tuple[Mask, Mask]`` tuple becomes:

    - slot 0: ``rotary_dict[layer_type] -> (cos, sin)``
    - slot 1: ``attention_mask_dict`` or ``None`` (HF derives causal when None)
    """
    cache_len = cache.get_seq_length() if cache is not None else 0
    seq_len = h.shape[1]
    device = h.device
    position_ids = torch.arange(
        cache_len, cache_len + seq_len, dtype=torch.long, device=device,
    ).unsqueeze(0)
    # Config may be nested (AutoModelForCausalLM wraps text config)
    config = getattr(model, "config", None)
    # Gemma4Config wraps Gemma4TextConfig; real 26B loads via AutoModelForCausalLM
    # produce a Gemma4Config whose layer_types lives on .text_config. Call
    # get_text_config() when available to unwrap. Tiny synthetic configs
    # expose layer_types directly at the top level (no get_text_config method),
    # so the hasattr guard preserves test behavior.
    if config is not None and hasattr(config, "get_text_config"):
        config = config.get_text_config()
    layer_types = list(getattr(config, "layer_types", [])) if config is not None else []
    unique_types = sorted(set(layer_types))
    if not unique_types:
        # Fallback: single "full_attention" entry (tests with minimal models).
        unique_types = ["full_attention"]
    rotary_dict: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    rotary_emb = _text_model(model).rotary_emb
    with torch.no_grad():
        for layer_type in unique_types:
            cos, sin = rotary_emb(h, position_ids, layer_type)
            rotary_dict[layer_type] = (cos, sin)
    return rotary_dict, None


def run_layer_atomic(
    model: Any,
    layer_idx: int,
    h: torch.Tensor,
    cache: Any,
    global_mask: Any,
    sliding_mask: Any,
) -> torch.Tensor:
    """Run one decoder layer against the real HF Gemma4TextDecoderLayer.

    Phase 7-C-1: ``global_mask`` = rotary_dict (keyed by layer_type);
    ``sliding_mask`` = attention_mask_dict or None (HF derives causal).
    ``position_ids`` / ``cache_position`` derived from cache state.
    """
    rotary_dict = global_mask
    attn_mask_dict = sliding_mask
    layer = _text_model(model).layers[layer_idx]
    layer_type = _resolve_layer_type(model, layer_idx)
    cos, sin = rotary_dict[layer_type]
    attention_mask = (
        attn_mask_dict.get(layer_type) if attn_mask_dict is not None else None
    )
    cache_len = cache.get_seq_length() if cache is not None else 0
    seq_len = h.shape[1]
    device = h.device
    position_ids = torch.arange(
        cache_len, cache_len + seq_len, dtype=torch.long, device=device,
    ).unsqueeze(0)
    # Gemma4TextAttention.forward requires shared_kv_states positionally
    # (no default). For non-kv-shared models (num_kv_shared_layers=0) an
    # empty dict is correct — layers with store_full_length_kv may still
    # stash into it but nothing reads back. Mirror HF's model.forward,
    # which always threads `shared_kv_states = {}` through the layers.
    with torch.no_grad():
        out = layer(
            hidden_states=h,
            position_embeddings=(cos, sin),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            shared_kv_states={},
        )
    # HF Gemma4TextDecoderLayer returns a plain torch.Tensor (not tuple).
    # Defensive: if some config variant returns tuple, unpack.
    if isinstance(out, tuple):
        return out[0]  # type: ignore[no-any-return]
    return out  # type: ignore[no-any-return]


def run_layers(
    model: Any,
    start_layer: int,
    end_layer: int,
    h: torch.Tensor,
    cache: Any,
    masks: tuple[Any, Any],
    is_split_layer: Callable[[int], bool],
) -> torch.Tensor:
    """Loop over [start_layer, end_layer) calling run_layer_atomic on each
    non-split layer. Split layers raise — the orchestrator is supposed to
    intercept before run_layers is called.

    Phase 6-B provenance append does NOT happen here (that is a ``node.py``
    concern, outside the engine primitives)."""
    global_mask, sliding_mask = masks
    for i in range(start_layer, end_layer):
        if is_split_layer(i):
            raise RuntimeError(
                f"run_layers called over a split layer (layer_idx={i}); "
                f"split layers must be handled by the ExpertOrchestrator"
            )
        h = run_layer_atomic(model, i, h, cache, global_mask, sliding_mask)
    return h


def finalize(model: Any, h: torch.Tensor) -> torch.Tensor:
    """Apply the final RMSNorm + lm_head + optional logit softcapping.

    Gemma 4 26B sets ``config.final_logit_softcapping = 30.0`` — HF's
    ``Gemma4ForCausalLM.forward`` divides logits by the softcap, passes
    through ``torch.tanh``, then re-multiplies. Skipping this produces
    logits with different magnitudes; argmax matches at obvious positions
    but diverges when runner-up logits are close.
    """
    with torch.no_grad():
        h = _text_model(model).norm(h)
        logits: torch.Tensor = model.lm_head(h)
        config = getattr(model, "config", None)
        if config is not None and hasattr(config, "get_text_config"):
            config = config.get_text_config()
        softcap = getattr(config, "final_logit_softcapping", None)
        if softcap is not None:
            logits = logits / softcap
            logits = torch.tanh(logits)
            logits = logits * softcap
        return logits


# ---- wire serialization ------------------------------------------------

def tensor_to_bytes(t: torch.Tensor) -> bytes:
    """Contiguous CPU bytes. bf16 is 2 bytes/element; matches MLX wire layout
    (both are IEEE 754 bfloat16)."""
    return t.contiguous().cpu().view(torch.uint8).numpy().tobytes()


def bytes_to_tensor(raw: bytes, shape: list[int], dtype: int) -> torch.Tensor:
    torch_dt = _wire_to_torch_dtype(dtype)
    buf = bytearray(raw)
    flat = torch.frombuffer(buf, dtype=torch_dt)
    return flat.reshape(shape)
