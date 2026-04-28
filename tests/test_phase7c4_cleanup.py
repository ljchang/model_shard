"""Phase 7-C-4 cleanup regression tests.

Verifies cleanup invariants that the rest of the suite doesn't cover.
Subsequent tasks in this phase will add tests to this file as the
corresponding cleanups land.
"""

from __future__ import annotations


def test_mlx_compute_lock_alias_removed() -> None:
    """The Phase 7-B `_MLX_COMPUTE_LOCK` alias must be retired by 7-C-4.

    Only `_COMPUTE_LOCK` should exist as a module attribute on node.py.
    Any external consumer that imported the old name has had a release
    cycle to migrate.
    """
    from model_shard import node

    assert hasattr(node, "_COMPUTE_LOCK"), "_COMPUTE_LOCK must exist"
    assert not hasattr(node, "_MLX_COMPUTE_LOCK"), (
        "_MLX_COMPUTE_LOCK alias must be removed in Phase 7-C-4"
    )


def test_phase_b_with_retry_no_lm_param() -> None:
    """`_phase_b_with_retry` had `lm: Any` only for signature stability
    in Phase 7-B. With the fallback removed in 7-B Task 6 and the
    `del lm` dead-code line shipped since, 7-C-4 retires the parameter."""
    import inspect

    from model_shard.expert_orchestrator import ExpertOrchestrator

    sig = inspect.signature(ExpertOrchestrator._phase_b_with_retry)
    assert "lm" not in sig.parameters, (
        f"_phase_b_with_retry must not take `lm`; got params {list(sig.parameters)}"
    )


def test_backend_protocol_has_apply_outer_decoder_ops() -> None:
    """Phase 7-C-4 adds apply_outer_decoder_ops so Backend owns the
    layer accessor that previously leaked via the `lm` parameter."""
    import inspect

    from model_shard.backends.base import Backend

    method = getattr(Backend, "apply_outer_decoder_ops", None)
    assert method is not None, (
        "Backend protocol must declare apply_outer_decoder_ops"
    )
    sig = inspect.signature(method)
    expected = {"self", "layer_idx", "block_in", "residual"}
    assert set(sig.parameters) == expected, (
        f"apply_outer_decoder_ops params {set(sig.parameters)} != {expected}"
    )


def test_run_split_layer_no_lm_param() -> None:
    """7-C-4 finishes the lm-removal job: run_split_layer no longer
    takes the lm handle. Backend.apply_outer_decoder_ops absorbs the
    layer accessor."""
    import inspect

    from model_shard.expert_orchestrator import ExpertOrchestrator

    sig = inspect.signature(ExpertOrchestrator.run_split_layer)
    assert "lm" not in sig.parameters, (
        f"run_split_layer must not take `lm`; got {list(sig.parameters)}"
    )


def test_aggregate_experts_batched_signature() -> None:
    """Backend.aggregate_experts now takes a top_k_ids ARRAY ([B, S, K])
    instead of a list[int] — the per-position loop moves into the
    backend so run_split_layer can stop slicing/concating.

    With `from __future__ import annotations` in base.py annotations
    are strings; `get_annotations(eval_str=True)` evaluates them under
    the module's globals so we can compare against the resolved type.
    """
    import inspect

    from model_shard.backends.base import Backend

    annotations = inspect.get_annotations(
        Backend.aggregate_experts, eval_str=True,
    )
    top_k_ids_annotation = annotations["top_k_ids"]
    # In base.py, `Activation = Any`. We don't compare against Any
    # directly (every annotation would match); instead we assert the
    # old per-position list[int] annotation is gone.
    assert top_k_ids_annotation is not list, (
        f"top_k_ids must not be `list`; got {top_k_ids_annotation!r}"
    )
    assert top_k_ids_annotation != list[int], (
        f"top_k_ids must not be `list[int]`; got {top_k_ids_annotation!r}"
    )
