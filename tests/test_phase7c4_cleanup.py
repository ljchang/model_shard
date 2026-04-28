"""Phase 7-C-4 cleanup regression tests.

These tests verify the cleanup invariants that the rest of the suite
doesn't already cover:
  - _MLX_COMPUTE_LOCK alias is gone (only _COMPUTE_LOCK remains)
  - lm parameter is gone from ExpertOrchestrator.run_split_layer and
    _phase_b_with_retry signatures
  - Backend protocol has apply_outer_decoder_ops
  - Backend.aggregate_experts accepts the batched [B, S, K] signature
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
