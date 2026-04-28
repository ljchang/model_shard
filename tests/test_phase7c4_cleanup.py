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
