"""Phase 7-C-2 Task 1: MLX top_k_ids_and_weights helper."""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from model_shard import mlx_engine  # noqa: E402


def test_top_k_ids_and_weights_returns_python_lists():
    logits = mx.zeros((1, 1, 10), dtype=mx.float32)
    logits[0, -1, 3] = 5.0
    logits[0, -1, 7] = 3.0
    logits[0, -1, 1] = 1.0
    ids, weights = mlx_engine.top_k_ids_and_weights(logits, k=3)
    assert isinstance(ids, list)
    assert isinstance(weights, list)
    assert all(isinstance(i, int) for i in ids)
    assert all(isinstance(w, float) for w in weights)


def test_top_k_ids_and_weights_correct_order():
    logits = mx.zeros((1, 1, 10), dtype=mx.float32)
    logits[0, -1, 3] = 5.0
    logits[0, -1, 7] = 3.0
    logits[0, -1, 1] = 1.0
    ids, _ = mlx_engine.top_k_ids_and_weights(logits, k=3)
    assert ids == [3, 7, 1]


def test_top_k_ids_and_weights_returns_softmax_probs():
    logits = mx.zeros((1, 1, 4), dtype=mx.float32)
    logits[0, -1, 0] = 10.0
    _, weights = mlx_engine.top_k_ids_and_weights(logits, k=4)
    assert weights[0] > 0.99
    assert sum(weights) == pytest.approx(1.0, abs=1e-5)


def test_top_k_ids_and_weights_uses_last_position():
    logits = mx.zeros((1, 3, 5), dtype=mx.float32)
    logits[0, 0, 4] = 100.0
    logits[0, -1, 2] = 10.0
    ids, _ = mlx_engine.top_k_ids_and_weights(logits, k=1)
    assert ids == [2]


def test_top_k_ids_and_weights_k_larger_than_vocab():
    logits = mx.zeros((1, 1, 3), dtype=mx.float32)
    ids, weights = mlx_engine.top_k_ids_and_weights(logits, k=5)
    assert len(ids) == 3
    assert len(weights) == 3
