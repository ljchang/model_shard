"""Phase 7-A: Backend protocol shape + runtime_checkable behavior."""
from __future__ import annotations

from model_shard.backends import Backend


def test_backend_is_runtime_checkable_protocol():
    """Protocol declared with @runtime_checkable; isinstance() works on it."""
    # The Backend object exposes _is_runtime_protocol internally when
    # decorated with @runtime_checkable.
    assert getattr(Backend, "_is_runtime_protocol", False) is True


def test_backend_declares_required_methods():
    """Protocol must declare every method Node/ExpertOrchestrator will call."""
    required = {
        "load", "load_partial", "num_layers", "held_ids", "is_split_layer",
        "embed", "make_cache", "make_masks",
        "run_layer_atomic", "run_attention_and_route",
        "run_shared_expert", "run_selected_experts",
        "aggregate_experts", "finalize", "argmax_last",
        "tensor_to_bytes", "bytes_to_tensor", "dtype_to_wire",
        "slice_expert", "attach_expert", "detach_expert",
    }
    declared = {
        name for name in dir(Backend)
        if not name.startswith("_") and callable(getattr(Backend, name, None))
    }
    missing = required - declared
    assert not missing, f"Backend protocol missing methods: {missing}"


def test_backend_has_name_class_attr():
    """Backend declares `name: str` so consumers can log which backend is active."""
    assert "name" in getattr(Backend, "__annotations__", {})


def test_activation_cache_mask_topk_type_aliases_exist():
    """Opaque handle types for tensor/cache/mask/topk results."""
    from model_shard.backends.base import Activation, Cache, Mask, TopK
    # Just verify they are importable. Their actual typing is Any at runtime.
    _ = (Activation, Cache, Mask, TopK)
