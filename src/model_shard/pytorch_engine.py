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
    """HF computes masks internally on layer.forward; we return placeholders.

    The returned tuple is passed through the Backend.run_layer_atomic /
    run_attention_and_route signatures unchanged — concrete backends decide
    how to use them. On the PyTorch side, None / None is safe because the
    layer builds its own causal mask from position_ids + sliding_window.
    """
    return (None, None)


def run_layer_atomic(
    model: Any,
    layer_idx: int,
    h: torch.Tensor,
    cache: Any,
    global_mask: Any,
    sliding_mask: Any,
) -> torch.Tensor:
    """Run one decoder layer atomically.

    HF ``Gemma4DecoderLayer.forward`` builds its own attention mask; we
    pass the hidden states through and let the layer consume / update the
    cache in-place. ``use_cache=True`` is always set (works around
    transformers bug #45242)."""
    layer = model.model.layers[layer_idx]
    with torch.no_grad():
        out = layer(h)
    # HF layer can return a tuple (hidden, attn_weights, past_kv) or just hidden.
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
