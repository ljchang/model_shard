"""Phase 7-B Task 5: PyTorchBackend state handling + protocol conformance."""
from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")

from model_shard.backends import Backend, PyTorchBackend  # noqa: E402


def test_pytorch_backend_implements_backend_protocol():
    b = PyTorchBackend()
    assert isinstance(b, Backend)


def test_pytorch_backend_name_is_pytorch():
    assert PyTorchBackend.name == "pytorch"


def test_pytorch_backend_default_device_auto_selects():
    b = PyTorchBackend()
    assert b._device in {"cuda", "mps", "cpu"}


def test_pytorch_backend_explicit_device_cpu():
    b = PyTorchBackend(device="cpu")
    assert b._device == "cpu"
    assert b._dtype == torch.bfloat16


def test_pytorch_backend_mps_uses_fp16():
    """MPS doesn't support bf16; we fall back to fp16."""
    b = PyTorchBackend(device="mps")
    assert b._dtype == torch.float16


def test_pytorch_backend_from_loaded_model_wraps_existing():
    model = MagicMock()
    model.config.num_hidden_layers = 30
    b = PyTorchBackend.from_loaded_model(model, device="cpu")
    assert b._model is model
    assert b.num_layers() == 30


def test_pytorch_backend_held_ids_reads_internal_registry():
    b = PyTorchBackend(device="cpu")
    b._held_experts_per_layer = {15: (0, 3, 6)}
    assert b.held_ids(15) == (0, 3, 6)
    assert b.held_ids(99) == ()


def test_pytorch_backend_is_split_layer_always_false():
    b = PyTorchBackend(device="cpu")
    assert b.is_split_layer(0) is False
    assert b.is_split_layer(15) is False


def test_pytorch_backend_tensor_to_bytes_roundtrips_bfloat16():
    b = PyTorchBackend(device="cpu")
    t = torch.full((2, 4), 1.5, dtype=torch.bfloat16)
    raw = b.tensor_to_bytes(t)
    recovered = b.bytes_to_tensor(raw, shape=[2, 4], dtype=b.dtype_to_wire(t))
    assert torch.equal(recovered, t)


def test_pytorch_backend_argmax_last_returns_int():
    b = PyTorchBackend(device="cpu")
    logits = torch.tensor([[[1.0, 2.0, 3.0]]])
    assert b.argmax_last(logits) == 2


def test_pytorch_backend_accepts_optional_lock():
    lock = threading.Lock()
    b = PyTorchBackend(device="cpu", torch_lock=lock)
    assert b._torch_lock is lock


def test_pytorch_backend_creates_private_lock_when_none():
    b = PyTorchBackend(device="cpu")
    assert isinstance(b._torch_lock, type(threading.Lock()))


def test_pytorch_backend_dtype_to_wire_bfloat16():
    b = PyTorchBackend(device="cpu")
    t = torch.zeros((1,), dtype=torch.bfloat16)
    from model_shard._pb import wire_pb2
    assert b.dtype_to_wire(t) == wire_pb2.DTYPE_BFLOAT16
