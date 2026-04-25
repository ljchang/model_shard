"""Phase 7-B: PyTorch per-expert tensor slicing (Phase 5a + 5b + 6-C).

Mirror of partial_load.py. HF Gemma4TextExperts uses stacked tensors
identical in shape semantics to the MLX port, so the algorithm is a
direct translation.
"""
from __future__ import annotations

import threading
from typing import Any

import torch

from model_shard.pytorch_engine import _text_model


def _experts(model: Any, layer_idx: int) -> Any:
    return _text_model(model).layers[layer_idx].experts


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
    """Load only the held experts via streaming safetensors.

    Reads tensors one-at-a-time from disk via ``safetensors.safe_open``,
    slices each expert tensor at read time to ``[len(held), ...]``, then
    materializes the result on ``device``. Peak memory ≈ one expert
    tensor (~1 GB) staging + the growing target model.

    Replaces the prior MVP behavior (full load on device, then defensive
    zero) which OOM'd on 24 GB GPUs because peak VRAM = full ~52 GB
    regardless of how few experts the shard kept.
    """
    import json
    import re
    from pathlib import Path

    from accelerate import init_empty_weights  # type: ignore[import-untyped]
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM

    from model_shard.pytorch_engine import _default_device

    device = device or _default_device()
    if device == "mps" and dtype == torch.bfloat16:
        dtype = torch.float16

    snapshot_path = Path(snapshot_download(hf_id))
    config = AutoConfig.from_pretrained(hf_id)
    with (snapshot_path / "model.safetensors.index.json").open() as fh:
        index = json.load(fh)
    weight_map: dict[str, str] = index["weight_map"]

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(  # type: ignore[no-untyped-call]
            config, torch_dtype=dtype,
        )

    handles = {
        f: safe_open(  # type: ignore[no-untyped-call]
            snapshot_path / f, framework="pt", device="cpu",
        )
        for f in sorted(set(weight_map.values()))
    }

    # Any tensor under model.language_model.layers.X.* where X is OUTSIDE
    # this range is skipped — left as a meta tensor that the runtime never
    # touches because the shard's pipeline only invokes its own layer slice.
    # Without this, layers outside the shard's range would still load full
    # [128, ...] expert tensors and OOM a 24 GB GPU.
    if held_experts_per_layer:
        shard_layers: set[int] = set(held_experts_per_layer.keys())
    else:
        shard_layers = set()

    layer_re = re.compile(r"^model\.language_model\.layers\.(\d+)\.")

    # Keep expert tensors at full [num_experts, ...] shape. HF's MoE forward
    # (transformers/integrations/moe.py:_grouped_mm) indexes the weight by
    # the router's output expert IDs — sliced [k, ...] shapes produce
    # "matrix batch sizes have to match" RuntimeErrors. The held_experts_per_
    # layer mapping is now informational metadata for the orchestrator's
    # global->local dispatch; weights stay full-shape.
    skipped = 0
    for tensor_name, shard_file in weight_map.items():
        if shard_layers:
            lm = layer_re.match(tensor_name)
            if lm is not None and int(lm.group(1)) not in shard_layers:
                skipped += 1
                continue
        full_tensor = handles[shard_file].get_tensor(  # type: ignore[no-untyped-call]
            tensor_name,
        )
        device_tensor = full_tensor.to(device=device, dtype=dtype)
        del full_tensor
        _set_module_attr(model, tensor_name, device_tensor)

    # lm_head ↔ embed_tokens are tied — re-link after meta replacement.
    model.tie_weights()
    model.eval()
    return model


def _set_module_attr(model: Any, dotted_name: str, tensor: torch.Tensor) -> None:
    """Replace ``model.<dotted_name>`` with the loaded tensor.

    Walks the dotted path, then either reassigns an ``nn.Parameter`` (for
    parameter slots — the common case) or re-registers a buffer (for
    non-parameter tensors). Handles shape mismatch transparently because
    the slot is fully replaced rather than data-copied.
    """
    parts = dotted_name.split(".")
    obj: Any = model
    for p in parts[:-1]:
        obj = getattr(obj, p)
    leaf = parts[-1]
    current = getattr(obj, leaf, None)
    if isinstance(current, torch.nn.Parameter):
        new_param = torch.nn.Parameter(tensor, requires_grad=False)
        # Bypass setattr's nn.Module logic that would re-register; assign
        # directly to _parameters dict to avoid validation against the
        # previous (meta) parameter's shape.
        if isinstance(obj, torch.nn.Module):
            obj._parameters[leaf] = new_param
        else:
            setattr(obj, leaf, new_param)
    elif isinstance(current, torch.Tensor):
        # Buffer slot — re-register with the new tensor.
        if isinstance(obj, torch.nn.Module) and leaf in obj._buffers:
            obj._buffers[leaf] = tensor
        else:
            setattr(obj, leaf, tensor)
    else:
        raise ValueError(
            f"unexpected attribute type at {dotted_name}: {type(current).__name__}"
        )
