"""Backend protocol and implementations for Phase 7+ multi-backend support.

Phase 7-A shipped the protocol and MLXBackend. Phase 7-B adds
PyTorchBackend. Phase 7-C will add heterogeneous-cluster support.
"""

from model_shard.backends.base import (
    Activation,
    Backend,
    Cache,
    Mask,
    TopK,
)
from model_shard.backends.mlx_backend import MLXBackend
from model_shard.backends.pytorch_backend import PyTorchBackend

__all__ = [
    "Activation",
    "Backend",
    "Cache",
    "MLXBackend",
    "Mask",
    "PyTorchBackend",
    "TopK",
]
