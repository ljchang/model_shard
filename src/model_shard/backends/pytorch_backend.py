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
        self,
        layer_idx: int,
        expert_outputs: dict[int, Any],   # {eid: [B, S, H] torch.Tensor}
        top_k_ids: Any,                   # [B, S, K] torch.Tensor
        top_k_weights: Any,               # [B, S, K] torch.Tensor
        shared_out: Any,                  # [B, S, H] torch.Tensor
    ) -> Any:
        """Per-position aggregation on PyTorch.

        Phase 7-C-4: this method now owns the per-position loop that
        ExpertOrchestrator.run_split_layer used to drive. Pure helper
        ``pt_moe.aggregate_experts`` is still per-position and is called
        once per (b, l) here; final shape is built via torch.cat per row
        and across rows.

        Note on `shared_out`: unlike MLX (where `run_shared_expert`
        already applies post_feedforward_layernorm_1), the PyTorch path's
        `run_shared_expert` is pre-LN_1 and `pt_moe.aggregate_experts`
        applies LN_1 to `shared_out` itself. The MLX/PyTorch asymmetry
        is preserved here for parity with the HF reference forward."""
        assert self._model is not None
        n_batch, n_seq, _n_k = top_k_ids.shape
        rows: list[Any] = []
        for b in range(n_batch):
            cells: list[Any] = []
            for ll in range(n_seq):
                ids_pos = [int(x) for x in top_k_ids[b, ll].reshape(-1).tolist()]
                per_pos = {
                    eid: expert_outputs[eid][b : b + 1, ll : ll + 1, :]
                    for eid in ids_pos
                }
                weights_pos = top_k_weights[b : b + 1, ll : ll + 1, :]
                shared_pos = shared_out[b : b + 1, ll : ll + 1, :]
                cells.append(
                    pt_moe.aggregate_experts(
                        self._model, layer_idx,
                        per_pos, ids_pos, weights_pos, shared_pos,
                    )
                )
            rows.append(torch.cat(cells, dim=1) if n_seq > 1 else cells[0])
        return torch.cat(rows, dim=0) if n_batch > 1 else rows[0]

    def apply_outer_decoder_ops(
        self,
        layer_idx: int,
        block_in: Any,  # torch.Tensor
        residual: Any,  # torch.Tensor
    ) -> Any:
        """Apply the outer post-MoE ops on the PyTorch path. See the
        MLXBackend docstring for what these ops are.

        Uses pytorch_engine._text_model to unwrap Gemma4Model ->
        language_model when the loaded model is the multimodal wrapper."""
        from model_shard.pytorch_engine import _text_model
        assert self._model is not None
        layer = _text_model(self._model).layers[layer_idx]
        with torch.no_grad():
            out = layer.post_feedforward_layernorm(block_in)
            out = residual + out
            if layer.layer_scalar is not None:
                out = out * layer.layer_scalar
        return out

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
