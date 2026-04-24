"""Phase 7-C-3b Task 1: cross-backend wire-format roundtrip.

Both ``mlx_engine.tensor_to_bytes`` and ``pytorch_engine.tensor_to_bytes``
serialize bf16 as raw IEEE 754 bytes. This test pins that contract:
  * Same logical tensor → same bytes from both backends.
  * MLX bytes deserialize correctly via PyTorch ``bytes_to_tensor`` and
    vice versa.
"""
from __future__ import annotations

from typing import Any

import pytest

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")

from model_shard import mlx_engine, pytorch_engine  # noqa: E402
from model_shard._pb import wire_pb2  # noqa: E402


def _mlx_tensor_from_values(values: list[float]) -> Any:
    return mx.array(values, dtype=mx.bfloat16).reshape(1, -1)


def _torch_tensor_from_values(values: list[float]) -> Any:
    return torch.tensor(values, dtype=torch.bfloat16).reshape(1, -1)


def test_mlx_and_pytorch_serialize_bf16_to_same_bytes() -> None:
    """Same logical bf16 tensor → byte-identical from both backends."""
    values = [0.0, 1.0, -1.0, 0.5, -0.5, 1e-3, -1e-3, 12.34, -56.78, 100.0]
    mlx_t = _mlx_tensor_from_values(values)
    pt_t = _torch_tensor_from_values(values)
    mlx_bytes = mlx_engine.tensor_to_bytes(mlx_t)
    pt_bytes = pytorch_engine.tensor_to_bytes(pt_t)
    assert mlx_bytes == pt_bytes, (
        f"MLX bytes differ from PyTorch bytes for the same bf16 tensor; "
        f"mlx={mlx_bytes.hex()} pt={pt_bytes.hex()}"
    )


def test_mlx_bytes_deserialize_via_pytorch() -> None:
    """MLX-serialized bf16 bytes deserialize correctly with PyTorch."""
    values = [0.0, 1.0, -1.0, 12.34, -56.78]
    mlx_t = _mlx_tensor_from_values(values)
    mlx_bytes = mlx_engine.tensor_to_bytes(mlx_t)
    shape = list(mlx_t.shape)
    pt_recovered = pytorch_engine.bytes_to_tensor(
        mlx_bytes, shape, wire_pb2.DTYPE_BFLOAT16,
    )
    assert pt_recovered.dtype == torch.bfloat16
    assert list(pt_recovered.shape) == shape
    pt_expected = _torch_tensor_from_values(values)
    assert torch.equal(pt_recovered, pt_expected), (
        f"MLX bytes deserialized via PyTorch don't match expected; "
        f"got={pt_recovered} expected={pt_expected}"
    )


def test_pytorch_bytes_deserialize_via_mlx() -> None:
    """PyTorch-serialized bf16 bytes deserialize correctly with MLX."""
    values = [0.0, 1.0, -1.0, 12.34, -56.78]
    pt_t = _torch_tensor_from_values(values)
    pt_bytes = pytorch_engine.tensor_to_bytes(pt_t)
    shape = list(pt_t.shape)
    mlx_recovered = mlx_engine.bytes_to_tensor(
        pt_bytes, shape, wire_pb2.DTYPE_BFLOAT16,
    )
    assert mlx_recovered.dtype == mx.bfloat16
    assert list(mlx_recovered.shape) == shape
    mlx_expected = _mlx_tensor_from_values(values)
    assert mx.array_equal(mlx_recovered, mlx_expected).item(), (
        f"PyTorch bytes deserialized via MLX don't match expected; "
        f"max abs diff = {mx.max(mx.abs(mlx_recovered - mlx_expected)).item()}"
    )


def test_full_roundtrip_mlx_to_pytorch_to_mlx() -> None:
    """MLX → bytes → PyTorch tensor → bytes → MLX tensor preserves values."""
    values = [0.0, 1.0, -1.0, 12.34, -56.78, 1e-3, -1e-3]
    mlx_orig = _mlx_tensor_from_values(values)
    shape = list(mlx_orig.shape)
    bytes_a = mlx_engine.tensor_to_bytes(mlx_orig)
    pt_intermediate = pytorch_engine.bytes_to_tensor(
        bytes_a, shape, wire_pb2.DTYPE_BFLOAT16,
    )
    bytes_b = pytorch_engine.tensor_to_bytes(pt_intermediate)
    mlx_final = mlx_engine.bytes_to_tensor(
        bytes_b, shape, wire_pb2.DTYPE_BFLOAT16,
    )
    assert mx.array_equal(mlx_orig, mlx_final).item(), (
        f"Full roundtrip MLX→PT→MLX lost values; "
        f"max abs diff = {mx.max(mx.abs(mlx_orig - mlx_final)).item()}"
    )
