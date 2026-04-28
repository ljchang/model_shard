"""Phase 7-A Backend protocol.

Each Backend instance owns one LoadedModel-equivalent and exposes the
narrow tensor-level API the distributed engine calls. Consumers (Node,
ExpertOrchestrator) pass opaque handles between method calls and
serialize them at wire boundaries via tensor_to_bytes / bytes_to_tensor.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

# Opaque per-backend handle types. Typed as Any at runtime to keep the
# protocol structural; concrete backends use their native types
# (mx.array, torch.Tensor, etc.) internally.
Activation = Any
Cache = Any
Mask = Any
TopK = tuple[Activation, Activation]  # (top_k_indices, top_k_weights)


@runtime_checkable
class Backend(Protocol):
    """Tensor-level operations a Node / ExpertOrchestrator calls.

    Each Backend instance owns exactly one loaded model. All methods are
    thread-safe provided the caller holds the Node's _COMPUTE_LOCK
    (or the backend's own equivalent serialization primitive)."""

    name: str  # "mlx" | "pytorch" | "executorch" ...

    # --- Loading ---------------------------------------------------------

    def load(self, hf_id: str) -> None: ...
    def load_partial(
        self, hf_id: str, held_experts_per_layer: dict[int, list[int]]
    ) -> None: ...
    def num_layers(self) -> int: ...
    def held_ids(self, layer_idx: int) -> tuple[int, ...]: ...
    def is_split_layer(self, layer_idx: int) -> bool: ...

    # --- Forward pass primitives -----------------------------------------

    def embed(self, token_ids: list[int]) -> Activation: ...
    def make_cache(self) -> Cache: ...
    def make_masks(self, h: Activation, cache: Cache) -> tuple[Mask, Mask]: ...
    def run_layer_atomic(
        self, layer_idx: int, h: Activation, cache: Cache,
        masks: tuple[Mask, Mask],
    ) -> Activation: ...
    def run_attention_and_route(
        self, layer_idx: int, h: Activation, cache: Cache,
        masks: tuple[Mask, Mask],
        heat_observer: Any = None,
    ) -> tuple[Activation, TopK]: ...
    def run_shared_expert(self, layer_idx: int, h: Activation) -> Activation: ...
    def run_selected_experts(
        self, layer_idx: int, h: Activation, expert_ids: list[int],
    ) -> dict[int, Activation]: ...
    def aggregate_experts(
        self, layer_idx: int,
        expert_outputs: dict[int, Activation],
        top_k_ids: list[int],
        top_k_weights: Activation,
        shared_out: Activation,
    ) -> Activation: ...
    def finalize(self, h: Activation) -> Activation: ...
    def argmax_last(self, logits: Activation) -> int: ...

    # --- Wire serialization ----------------------------------------------

    def tensor_to_bytes(self, h: Activation) -> bytes: ...
    def bytes_to_tensor(
        self, raw: bytes, shape: list[int], dtype: int,
    ) -> Activation: ...
    def dtype_to_wire(self, h: Activation) -> int: ...

    # --- Partial-load / migration ----------------------------------------

    def slice_expert(
        self, layer_idx: int, expert_id: int,
    ) -> list[Activation]: ...
    def attach_expert(
        self, layer_idx: int, expert_id: int, tensors: list[Activation],
    ) -> None: ...
    def detach_expert(
        self, layer_idx: int, expert_id: int,
    ) -> None: ...
