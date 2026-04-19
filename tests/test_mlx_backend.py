"""Phase 7-A Task 3: MLXBackend state handling + protocol conformance."""
from __future__ import annotations

import threading
from unittest.mock import MagicMock

import mlx.core as mx

from model_shard.backends import Backend, MLXBackend


def test_mlx_backend_implements_backend_protocol():
    """runtime_checkable Protocol check at instance level."""
    b = MLXBackend()
    assert isinstance(b, Backend)


def test_mlx_backend_name_is_mlx():
    assert MLXBackend.name == "mlx"


def test_mlx_backend_from_loaded_model_wraps_existing_lm():
    """MLXBackend.from_loaded_model(lm) is the test-fixture escape hatch
    that lets callers inject a pre-loaded (or mocked) LoadedModel."""
    lm = MagicMock()
    lm.num_layers = 30
    b = MLXBackend.from_loaded_model(lm)
    assert b._lm is lm
    assert b.num_layers() == 30


def test_mlx_backend_held_ids_delegates_to_lm():
    lm = MagicMock()
    lm.held_ids_per_layer = {15: (0, 3, 6)}
    b = MLXBackend.from_loaded_model(lm)
    assert b.held_ids(15) == (0, 3, 6)
    assert b.held_ids(99) == ()  # absent layer → empty tuple


def test_mlx_backend_is_split_layer_returns_false_by_default():
    """MLXBackend itself doesn't know which layers are split — that's a
    ShardSpec concern. Always returns False; callers consult ShardSpec."""
    lm = MagicMock()
    b = MLXBackend.from_loaded_model(lm)
    assert b.is_split_layer(0) is False
    assert b.is_split_layer(15) is False


def test_mlx_backend_tensor_to_bytes_roundtrips_bfloat16():
    b = MLXBackend()  # No model needed for this method.
    tensor = mx.full((2, 4), 1.5, dtype=mx.bfloat16)
    raw = b.tensor_to_bytes(tensor)
    recovered = b.bytes_to_tensor(raw, shape=[2, 4], dtype=b.dtype_to_wire(tensor))
    assert mx.array_equal(recovered, tensor).item()


def test_mlx_backend_argmax_last_returns_int():
    b = MLXBackend()
    logits = mx.array([[[1.0, 2.0, 3.0]]], dtype=mx.float32)
    assert b.argmax_last(logits) == 2


def test_mlx_backend_accepts_optional_lock():
    """MLXBackend(mlx_lock=existing_lock) uses the caller's lock for
    slice/attach/detach serialization. Default: backend-private lock."""
    lock = threading.Lock()
    b = MLXBackend(mlx_lock=lock)
    assert b._mlx_lock is lock


def test_mlx_backend_creates_private_lock_when_none():
    b = MLXBackend()
    assert isinstance(b._mlx_lock, type(threading.Lock()))
