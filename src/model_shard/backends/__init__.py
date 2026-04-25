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

# MLX is Apple Silicon only. On Linux (e.g. DGX Spark) the mlx package isn't
# installed — we still expose MLXBackend as a sentinel class so isinstance()
# checks elsewhere (node.py, expert_orchestrator.py) continue to work; actual
# instantiation raises ImportError.
try:
    from model_shard.backends.mlx_backend import MLXBackend
except ImportError:
    class MLXBackend:  # type: ignore[no-redef]
        """Sentinel class used when mlx is unavailable (non-Apple platforms).

        Preserves ``isinstance(x, MLXBackend)`` call sites. Attempting to
        instantiate raises ImportError with a pointer to the platform gate.
        """
        name: str = "mlx"

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError(
                "MLXBackend requires the mlx package, which is Apple Silicon "
                "only. On non-darwin hosts, use MODEL_SHARD_BACKEND=pytorch "
                "or construct PyTorchBackend directly."
            )

# torch is in the optional `pytorch` extras group, not a default dep. A Mac
# user running `uv sync --extra dev` (no `--extra pytorch`) won't have torch
# installed — same sentinel pattern as MLXBackend keeps `isinstance()` call
# sites working.
try:
    from model_shard.backends.pytorch_backend import PyTorchBackend
except ImportError:
    class PyTorchBackend:  # type: ignore[no-redef]
        """Sentinel class used when torch is unavailable.

        Preserves ``isinstance(x, PyTorchBackend)`` call sites. Attempting
        to instantiate raises ImportError with a pointer to the extras
        group that installs torch.
        """
        name: str = "pytorch"

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError(
                "PyTorchBackend requires the torch package. Install with "
                "`uv sync --extra pytorch`, or use MODEL_SHARD_BACKEND=mlx "
                "on Apple Silicon."
            )

__all__ = [
    "Activation",
    "Backend",
    "Cache",
    "MLXBackend",
    "Mask",
    "PyTorchBackend",
    "TopK",
]
