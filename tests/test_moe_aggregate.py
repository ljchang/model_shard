from __future__ import annotations

import mlx.core as mx
import pytest

from model_shard.moe import aggregate_experts


def _identity(x: mx.array) -> mx.array:
    """Substitute for post_feedforward_layernorm_2 that isolates aggregation
    logic from LN numerics."""
    return x


def test_aggregate_pairs_weight_to_slot_not_id() -> None:
    """aggregate_experts must pair top_k_weights[..., j] with
    expert_outputs[top_k_ids[j]] (slot order), matching mlx-vlm."""
    out_3 = mx.array([[[1.0, 0.0]]])
    out_7 = mx.array([[[0.0, 1.0]]])
    shared = mx.array([[[0.0, 0.0]]])

    # Slot 0 carries weight 0.9, slot 1 carries weight 0.1.
    weights = mx.array([[[0.9, 0.1]]])

    r_a = aggregate_experts({3: out_3, 7: out_7}, [3, 7], weights, shared, _identity)
    r_b = aggregate_experts({3: out_3, 7: out_7}, [7, 3], weights, shared, _identity)
    mx.eval(r_a, r_b)
    assert not mx.all(r_a == r_b).item(), (
        "Different id orderings with the same weights should produce "
        "different results — weight pairing follows slot, not id."
    )


def test_aggregate_shared_branch_is_added_linearly() -> None:
    """shared_out is added via a linear residual connection — aggregate's
    output depends linearly on shared_out (because h1 is added after
    post_ffn_ln_2 on the routed branch, not passed through it)."""
    shared = mx.array([[[10.0, 20.0]]])
    out = mx.array([[[1.0, 2.0]]])
    r = aggregate_experts({4: out}, [4], mx.array([[[1.0]]]), shared, _identity)
    r_shifted = aggregate_experts(
        {4: out}, [4], mx.array([[[1.0]]]), shared + 5.0, _identity
    )
    mx.eval(r, r_shifted)
    assert mx.all(r_shifted - r == mx.array([[[5.0, 5.0]]])).item()


def test_aggregate_missing_id_raises() -> None:
    with pytest.raises(KeyError, match="expert 5 output missing"):
        aggregate_experts(
            {}, [5], mx.array([[[1.0]]]), mx.array([[[0.0]]]), _identity
        )


def test_aggregate_applies_post_ffn_ln_2_to_routed_only() -> None:
    """post_ffn_ln_2 is applied to the gated sum, not the shared branch."""
    shared = mx.array([[[5.0, 5.0]]])
    out = mx.array([[[1.0, 1.0]]])

    def scale_2x(x: mx.array) -> mx.array:
        return x * 2.0

    # weight 1.0 → routed = 1.0 * out = [1, 1]; post_ffn_ln_2 → [2, 2]
    # shared stays [5, 5]. Expected [7, 7].
    r = aggregate_experts({4: out}, [4], mx.array([[[1.0]]]), shared, scale_2x)
    mx.eval(r)
    expected = mx.array([[[7.0, 7.0]]])
    assert mx.all(r == expected).item()
