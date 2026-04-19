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
        out: torch.Tensor = model.model.embed_tokens(input_ids)
        return out


def make_cache(model: Any) -> Any:
    """Construct a fresh DynamicCache for one request."""
    from transformers import DynamicCache
    return DynamicCache()


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
    layer_types = list(getattr(config, "layer_types", [])) if config is not None else []
    unique_types = sorted(set(layer_types))
    if not unique_types:
        # Fallback: single "full_attention" entry (tests with minimal models).
        unique_types = ["full_attention"]
    rotary_dict: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    rotary_emb = model.model.rotary_emb
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
    layer = model.model.layers[layer_idx]
    layer_type = layer.layer_type
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
    with torch.no_grad():
        out = layer(
            hidden_states=h,
            position_embeddings=(cos, sin),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
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
    """Apply the final RMSNorm + lm_head; return logits [1, L, V]."""
    with torch.no_grad():
        h = model.model.norm(h)
        logits: torch.Tensor = model.lm_head(h)
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
