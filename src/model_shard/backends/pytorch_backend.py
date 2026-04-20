"""Phase 7-B PyTorchBackend: Backend protocol implementation over the
existing pytorch_engine / pt_moe / pt_partial_load modules. Thin
delegation layer — zero logic duplication."""

from __future__ import annotations

import threading
from typing import Any

import torch

from model_shard import pt_moe, pt_partial_load, pytorch_engine


class PyTorchBackend:
    """PyTorch implementation of the Backend protocol.

    Each instance owns one HF ``Gemma4ForCausalLM`` (or mock) as
    ``self._model``. The optional ``torch_lock`` is used to serialize
    slice/attach/detach with concurrent forward passes (Node passes its
    process-wide ``_COMPUTE_LOCK`` here in production)."""

    name: str = "pytorch"

    def __init__(
        self,
        device: str | None = None,
        torch_lock: threading.Lock | None = None,
    ) -> None:
        self._device: str = device or pytorch_engine._default_device()
        self._dtype: torch.dtype = (
            torch.float16 if self._device == "mps" else torch.bfloat16
        )
        self._model: Any = None
        self._torch_lock: threading.Lock = torch_lock or threading.Lock()
        self._held_experts_per_layer: dict[int, tuple[int, ...]] = {}

    @classmethod
    def from_loaded_model(
        cls,
        model: Any,
        device: str | None = None,
        torch_lock: threading.Lock | None = None,
    ) -> "PyTorchBackend":  # noqa: UP037
        b = cls(device=device, torch_lock=torch_lock)
        b._model = model
        return b

    # --- Loading -------------------------------------------------------------

    def load(self, hf_id: str) -> None:
        self._model = pytorch_engine.load_model(
            hf_id, device=self._device, dtype=self._dtype,
        )

    def load_partial(
        self, hf_id: str, held_experts_per_layer: dict[int, list[int]],
    ) -> None:
        self._model = pt_partial_load.load_model_partial(
            hf_id, held_experts_per_layer,
            device=self._device, dtype=self._dtype,
        )
        self._held_experts_per_layer = {
            L: tuple(ids) for L, ids in held_experts_per_layer.items()
        }

    def num_layers(self) -> int:
        assert self._model is not None
        # Count via the text model directly. `self._model.config` is the
        # top-level Gemma4Config (multimodal) which lacks num_hidden_layers;
        # the field lives on config.get_text_config(). Going via the module
        # layer list is equivalent and avoids the config unwrap.
        return len(pytorch_engine._text_model(self._model).layers)

    def held_ids(self, layer_idx: int) -> tuple[int, ...]:
        return self._held_experts_per_layer.get(layer_idx, ())

    def is_split_layer(self, layer_idx: int) -> bool:
        # Phase 7-B: always False; ShardSpec.moe_experts is authoritative.
        return False

    # --- Forward pass primitives --------------------------------------------

    def embed(self, token_ids: list[int]) -> torch.Tensor:
        assert self._model is not None
        return pytorch_engine.embed_tokens(self._model, token_ids)

    def make_cache(self) -> Any:
        assert self._model is not None
        return pytorch_engine.make_cache(self._model)

    def make_masks(self, h: torch.Tensor, cache: Any) -> tuple[Any, Any]:
        assert self._model is not None
        return pytorch_engine.make_masks(self._model, h, cache)

    def run_layer_atomic(
        self, layer_idx: int, h: torch.Tensor, cache: Any,
        masks: tuple[Any, Any],
    ) -> torch.Tensor:
        assert self._model is not None
        global_mask, sliding_mask = masks
        return pytorch_engine.run_layer_atomic(
            self._model, layer_idx, h, cache, global_mask, sliding_mask,
        )

    def run_attention_and_route(
        self, layer_idx: int, h: torch.Tensor, cache: Any,
        masks: tuple[Any, Any], heat_observer: Any = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        assert self._model is not None
        post_attn, top_k_ids, top_k_weights = pt_moe.run_attention_and_route(
            self._model, h, layer_idx, cache, masks,
            heat_observer=heat_observer,
        )
        return post_attn, (top_k_ids, top_k_weights)

    def run_shared_expert(self, layer_idx: int, h: torch.Tensor) -> torch.Tensor:
        assert self._model is not None
        return pt_moe.run_shared_expert(self._model, h, layer_idx)

    def run_selected_experts(
        self, layer_idx: int, h: torch.Tensor, expert_ids: list[int],
    ) -> dict[int, torch.Tensor]:
        assert self._model is not None
        return pt_moe.run_selected_experts(
            self._model, h, layer_idx, expert_ids,
        )

    def aggregate_experts(
        self, layer_idx: int,
        expert_outputs: dict[int, torch.Tensor],
        top_k_ids: list[int],
        top_k_weights: torch.Tensor,
        shared_out: torch.Tensor,
    ) -> torch.Tensor:
        assert self._model is not None
        return pt_moe.aggregate_experts(
            self._model, layer_idx, expert_outputs, top_k_ids,
            top_k_weights, shared_out,
        )

    def finalize(self, h: torch.Tensor) -> torch.Tensor:
        assert self._model is not None
        return pytorch_engine.finalize(self._model, h)

    def argmax_last(self, logits: torch.Tensor) -> int:
        return int(torch.argmax(logits[0, -1, :]).item())

    # --- Wire serialization -------------------------------------------------

    def tensor_to_bytes(self, h: torch.Tensor) -> bytes:
        return pytorch_engine.tensor_to_bytes(h)

    def bytes_to_tensor(
        self, raw: bytes, shape: list[int], dtype: int,
    ) -> torch.Tensor:
        t = pytorch_engine.bytes_to_tensor(raw, shape, dtype)
        return t.to(self._device)

    def dtype_to_wire(self, h: torch.Tensor) -> int:
        return pytorch_engine.torch_to_wire_dtype(h.dtype)

    # --- Partial-load / migration -------------------------------------------

    def slice_expert(
        self, layer_idx: int, expert_id: int,
    ) -> list[torch.Tensor]:
        assert self._model is not None
        return pt_partial_load.slice_expert(
            self._model, layer_idx, expert_id, self._torch_lock,
        )

    def attach_expert(
        self, layer_idx: int, expert_id: int, tensors: list[torch.Tensor],
    ) -> None:
        assert self._model is not None
        pt_partial_load.attach_expert(
            self._model, layer_idx, expert_id, tensors, self._torch_lock,
        )
        held = set(self._held_experts_per_layer.get(layer_idx, ()))
        held.add(expert_id)
        self._held_experts_per_layer[layer_idx] = tuple(sorted(held))

    def detach_expert(self, layer_idx: int, expert_id: int) -> None:
        assert self._model is not None
        pt_partial_load.detach_expert(
            self._model, layer_idx, expert_id, self._torch_lock,
        )
        held = set(self._held_experts_per_layer.get(layer_idx, ()))
        held.discard(expert_id)
        self._held_experts_per_layer[layer_idx] = tuple(sorted(held))


__all__ = ["PyTorchBackend"]
