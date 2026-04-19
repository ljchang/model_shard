"""Phase 7-B: PyTorch per-expert tensor slicing (Phase 5a + 5b + 6-C).

Mirror of partial_load.py. HF Gemma4TextExperts uses stacked tensors
identical in shape semantics to the MLX port, so the algorithm is a
direct translation.
"""
from __future__ import annotations

import threading
from typing import Any

import torch


def _experts(model: Any, layer_idx: int) -> Any:
    return model.model.layers[layer_idx].experts


def slice_expert(
    model: Any, layer_idx: int, expert_id: int, lock: threading.Lock,
) -> list[torch.Tensor]:
    """Return [gate_up_proj[k].detach().cpu(), down_proj[k].detach().cpu()].

    Held under ``lock`` so a concurrent forward pass doesn't observe a
    torn state if the tensor is being written to."""
    e = _experts(model, layer_idx)
    with lock:
        return [
            e.gate_up_proj[expert_id].detach().cpu().clone(),
            e.down_proj[expert_id].detach().cpu().clone(),
        ]


def attach_expert(
    model: Any,
    layer_idx: int,
    expert_id: int,
    tensors: list[torch.Tensor],
    lock: threading.Lock,
) -> None:
    """Write tensors into the model's stacked expert slots in-place.

    Validates shape before acquiring the lock so a caller-side bug doesn't
    corrupt the live model. Moves tensors to the model's device under lock.
    """
    if len(tensors) != 2:
        raise ValueError(f"expected [gate_up, down] tensors, got {len(tensors)}")
    gate_up, down = tensors
    e = _experts(model, layer_idx)
    expected_gate_up = e.gate_up_proj[expert_id].shape
    expected_down = e.down_proj[expert_id].shape
    if tuple(gate_up.shape) != tuple(expected_gate_up):
        raise ValueError(
            f"gate_up shape mismatch: got {tuple(gate_up.shape)}, "
            f"expected {tuple(expected_gate_up)}"
        )
    if tuple(down.shape) != tuple(expected_down):
        raise ValueError(
            f"down shape mismatch: got {tuple(down.shape)}, "
            f"expected {tuple(expected_down)}"
        )
    device = e.gate_up_proj.device
    dtype = e.gate_up_proj.dtype
    with lock:  # noqa: SIM117
        with torch.no_grad():
            e.gate_up_proj[expert_id].copy_(gate_up.to(device=device, dtype=dtype))
            e.down_proj[expert_id].copy_(down.to(device=device, dtype=dtype))


def detach_expert(
    model: Any, layer_idx: int, expert_id: int, lock: threading.Lock,
) -> None:
    """Zero out the expert's slots in-place. Caller tracks live-expert state
    in _live_experts on the Node / MLXBackend-equivalent side."""
    e = _experts(model, layer_idx)
    with lock:  # noqa: SIM117
        with torch.no_grad():
            e.gate_up_proj[expert_id].zero_()
            e.down_proj[expert_id].zero_()


def load_model_partial(
    hf_id: str,
    held_experts_per_layer: dict[int, list[int]],
    device: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> Any:
    """Load the full model, then zero out experts not in held_experts_per_layer.

    MVP behavior: full load, then defensive zero — same memory footprint
    at steady state as held-only would give, just slower to warm up. A
    sparse-load refinement (skip reading non-held expert weights from disk)
    is a Phase 7-C optimization.
    """
    from model_shard import pytorch_engine
    model = pytorch_engine.load_model(hf_id, device=device, dtype=dtype)
    lock = threading.Lock()
    text_layers = model.model.layers
    for layer_idx, layer in enumerate(text_layers):
        experts = getattr(layer, "experts", None)
        if experts is None:
            continue
        num_experts = int(experts.num_experts)
        held = set(held_experts_per_layer.get(layer_idx, []))
        for k in range(num_experts):
            if k not in held:
                detach_expert(model, layer_idx, k, lock)
    return model
