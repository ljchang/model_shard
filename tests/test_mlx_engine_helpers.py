"""Phase 7-A Task 2: mlx_engine helper additions.

Verifies:
  * `run_layer_atomic` is a module-level function with the right signature.
  * `mx_to_wire_dtype` is a public alias for the pre-existing
    `_mx_to_wire_dtype`.
"""
from __future__ import annotations

import inspect

import mlx.core as mx

from model_shard import mlx_engine


def test_run_layer_atomic_exists_as_public_callable():
    assert callable(getattr(mlx_engine, "run_layer_atomic", None))


def test_run_layer_atomic_signature():
    sig = inspect.signature(mlx_engine.run_layer_atomic)
    params = list(sig.parameters.keys())
    # (lm, layer_idx, h, cache, global_mask, sliding_mask) — 6 positional.
    assert params == ["lm", "layer_idx", "h", "cache", "global_mask", "sliding_mask"]


def test_mx_to_wire_dtype_public_alias():
    # The underscore-prefixed original still exists.
    assert callable(getattr(mlx_engine, "_mx_to_wire_dtype", None))
    # The public alias exists and is the same object.
    assert callable(getattr(mlx_engine, "mx_to_wire_dtype", None))
    assert mlx_engine.mx_to_wire_dtype is mlx_engine._mx_to_wire_dtype


def test_mx_to_wire_dtype_returns_int_for_bfloat16():
    from model_shard._pb import wire_pb2
    assert mlx_engine.mx_to_wire_dtype(mx.bfloat16) == wire_pb2.DTYPE_BFLOAT16
