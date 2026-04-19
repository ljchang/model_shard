"""Backend protocol and implementations for Phase 7+ multi-backend support.

Phase 7-A ships the protocol and the MLXBackend. Phase 7-B/C add
PyTorchBackend and heterogeneous-cluster support.
"""

from model_shard.backends.base import (
    Activation,
    Backend,
    Cache,
    Mask,
    TopK,
)

__all__ = ["Activation", "Backend", "Cache", "Mask", "TopK"]
