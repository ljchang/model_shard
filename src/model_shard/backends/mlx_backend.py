"""Phase 7-A MLXBackend: implementation of the Backend protocol over
the existing mlx_engine / moe / partial_load modules. Thin delegation
layer — zero logic duplication."""

from __future__ import annotations

import threading
from typing import Any

import mlx.core as mx

from model_shard import mlx_engine, moe, partial_load


class MLXBackend:
    """MLX implementation of the Backend protocol.

    Each instance owns one ``LoadedModel`` as ``self._lm``. The optional
    ``mlx_lock`` is used to serialize ``slice_expert`` / ``attach_expert``
    / ``detach_expert`` with concurrent MLX compute (Node passes its
    process-wide ``_MLX_COMPUTE_LOCK`` here in production; unit tests may
    leave it unset and a backend-private lock is created).
    """

    name: str = "mlx"

    def __init__(self, mlx_lock: threading.Lock | None = None) -> None:
        self._lm: mlx_engine.LoadedModel | None = None
        self._mlx_lock: threading.Lock = mlx_lock or threading.Lock()

    @classmethod
    def from_loaded_model(
        cls, lm: mlx_engine.LoadedModel,
        mlx_lock: threading.Lock | None = None,
    ) -> MLXBackend:
        """Construct an MLXBackend wrapping an existing LoadedModel.
        Used by tests that inject a MagicMock or a real LoadedModel via
        the ``loaded_model=`` Node kwarg."""
        b = cls(mlx_lock=mlx_lock)
        b._lm = lm
        return b

    # --- Loading ---------------------------------------------------------

    def load(self, hf_id: str) -> None:
        self._lm = mlx_engine.load_model(hf_id)

    def load_partial(
        self, hf_id: str, held_experts_per_layer: dict[int, list[int]],
    ) -> None:
        self._lm = mlx_engine.load_model_partial(hf_id, held_experts_per_layer)

    def num_layers(self) -> int:
        assert self._lm is not None
        return int(self._lm.num_layers)

    def held_ids(self, layer_idx: int) -> tuple[int, ...]:
        assert self._lm is not None
        return self._lm.held_ids_per_layer.get(layer_idx, ())

    def is_split_layer(self, layer_idx: int) -> bool:
        # MLXBackend doesn't know which layers are split for a given shard.
        # Phase 7-A: always False; callers consult ShardSpec.moe_experts.
        return False

    # --- Forward pass primitives -----------------------------------------

    def embed(self, token_ids: list[int]) -> mx.array:
        assert self._lm is not None
        return mlx_engine.embed_tokens(self._lm, mx.array([token_ids]))

    def make_cache(self) -> list[Any]:
        assert self._lm is not None
        return mlx_engine.make_cache(self._lm)

    def make_masks(
        self, h: mx.array, cache: list[Any],
    ) -> tuple[Any, Any]:
        assert self._lm is not None
        return mlx_engine.make_masks(self._lm, h, cache)

    def run_layer_atomic(
        self, layer_idx: int, h: mx.array, cache: list[Any],
        masks: tuple[Any, Any],
    ) -> mx.array:
        assert self._lm is not None
        global_mask, sliding_mask = masks
        return mlx_engine.run_layer_atomic(
            self._lm, layer_idx, h, cache, global_mask, sliding_mask,
        )

    def run_attention_and_route(
        self, layer_idx: int, h: mx.array, cache: list[Any],
        masks: tuple[Any, Any], heat_observer: Any = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        assert self._lm is not None
        post_attn, top_k_ids, top_k_weights = moe.run_attention_and_route(
            self._lm, h, layer_idx, cache, masks, heat_observer=heat_observer,
        )
        return post_attn, (top_k_ids, top_k_weights)

    def run_shared_expert(self, layer_idx: int, h: mx.array) -> mx.array:
        assert self._lm is not None
        return moe.run_shared_expert(self._lm, h, layer_idx)

    def run_selected_experts(
        self, layer_idx: int, h: mx.array, expert_ids: list[int],
    ) -> dict[int, mx.array]:
        assert self._lm is not None
        return moe.run_selected_experts(self._lm, h, layer_idx, expert_ids)

    def aggregate_experts(
        self, layer_idx: int,
        expert_outputs: dict[int, mx.array],
        top_k_ids: list[int],
        top_k_weights: mx.array,
        shared_out: mx.array,
    ) -> mx.array:
        assert self._lm is not None
        layer = self._lm.text_model.layers[layer_idx]
        return moe.aggregate_experts(
            expert_outputs, top_k_ids, top_k_weights, shared_out,
            layer.post_feedforward_layernorm_2,
        )

    def finalize(self, h: mx.array) -> mx.array:
        assert self._lm is not None
        return mlx_engine.finalize(self._lm, h)

    def argmax_last(self, logits: mx.array) -> int:
        return int(mx.argmax(logits[0, -1, :]).item())

    # --- Wire serialization ----------------------------------------------

    def tensor_to_bytes(self, h: mx.array) -> bytes:
        return mlx_engine.tensor_to_bytes(h)

    def bytes_to_tensor(
        self, raw: bytes, shape: list[int], dtype: int,
    ) -> mx.array:
        return mlx_engine.bytes_to_tensor(raw, shape, dtype)

    def dtype_to_wire(self, h: mx.array) -> int:
        return mlx_engine.mx_to_wire_dtype(h.dtype)

    # --- Partial-load / migration ----------------------------------------

    def slice_expert(
        self, layer_idx: int, expert_id: int,
    ) -> list[mx.array]:
        assert self._lm is not None
        return partial_load.slice_expert(
            self._lm, layer_idx, expert_id, self._mlx_lock,
        )

    def attach_expert(
        self, layer_idx: int, expert_id: int, tensors: list[mx.array],
    ) -> None:
        assert self._lm is not None
        partial_load.attach_expert(
            self._lm, layer_idx, expert_id, tensors, self._mlx_lock,
        )

    def detach_expert(self, layer_idx: int, expert_id: int) -> None:
        assert self._lm is not None
        partial_load.detach_expert(
            self._lm, layer_idx, expert_id, self._mlx_lock,
        )


__all__ = ["MLXBackend"]
