"""Phase 7-C-2 Task 1: PyTorch top_k_ids_and_weights helper."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from model_shard import pytorch_engine  # noqa: E402


def test_top_k_ids_and_weights_returns_python_lists():
    """Helper output must be JSON-serializable (list[int], list[float])."""
    logits = torch.zeros((1, 1, 10), dtype=torch.float32)
    logits[0, -1, 3] = 5.0
    logits[0, -1, 7] = 3.0
    logits[0, -1, 1] = 1.0
    ids, weights = pytorch_engine.top_k_ids_and_weights(logits, k=3)
    assert isinstance(ids, list)
    assert isinstance(weights, list)
    assert all(isinstance(i, int) for i in ids)
    assert all(isinstance(w, float) for w in weights)


def test_top_k_ids_and_weights_correct_order():
    """Highest-probability token first."""
    logits = torch.zeros((1, 1, 10), dtype=torch.float32)
    logits[0, -1, 3] = 5.0
    logits[0, -1, 7] = 3.0
    logits[0, -1, 1] = 1.0
    ids, _ = pytorch_engine.top_k_ids_and_weights(logits, k=3)
    assert ids == [3, 7, 1]


def test_top_k_ids_and_weights_returns_softmax_probs():
    """Weights are softmax probabilities — sum to 1 over full vocab,
    top-k slice sums to <=1 but top-1 dominates."""
    logits = torch.zeros((1, 1, 4), dtype=torch.float32)
    logits[0, -1, 0] = 10.0  # winner
    _, weights = pytorch_engine.top_k_ids_and_weights(logits, k=4)
    assert weights[0] > 0.99
    assert sum(weights) == pytest.approx(1.0, abs=1e-5)


def test_top_k_ids_and_weights_uses_last_position():
    """For a [B, L, V] tensor with L>1, take the LAST position only."""
    logits = torch.zeros((1, 3, 5), dtype=torch.float32)
    logits[0, 0, 4] = 100.0  # would win if we looked at position 0
    logits[0, -1, 2] = 10.0  # actual winner at last position
    ids, _ = pytorch_engine.top_k_ids_and_weights(logits, k=1)
    assert ids == [2]


def test_top_k_ids_and_weights_k_larger_than_vocab():
    """Gracefully handle k > vocab (truncate to vocab size)."""
    logits = torch.zeros((1, 1, 3), dtype=torch.float32)
    ids, weights = pytorch_engine.top_k_ids_and_weights(logits, k=5)
    assert len(ids) == 3
    assert len(weights) == 3
