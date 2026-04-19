"""Phase 7-B Task 6: Node._default_backend() auto-detect + MODEL_SHARD_BACKEND env var."""
from __future__ import annotations

import pytest

from model_shard.backends import MLXBackend


def test_env_var_pytorch_forces_pytorch_backend(monkeypatch):
    monkeypatch.setenv("MODEL_SHARD_BACKEND", "pytorch")
    pytest.importorskip("torch")
    from model_shard.backends import PyTorchBackend
    from model_shard.node import _default_backend
    b = _default_backend()
    assert isinstance(b, PyTorchBackend)


def test_env_var_mlx_forces_mlx_backend(monkeypatch):
    monkeypatch.setenv("MODEL_SHARD_BACKEND", "mlx")
    from model_shard.node import _default_backend
    b = _default_backend()
    assert isinstance(b, MLXBackend)


def test_env_var_unset_prefers_mlx_on_apple_silicon(monkeypatch):
    monkeypatch.delenv("MODEL_SHARD_BACKEND", raising=False)
    import mlx.core as mx
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    from model_shard.node import _default_backend
    b = _default_backend()
    assert isinstance(b, MLXBackend)


def test_orchestrator_backend_now_required():
    """Phase 7-A had ``backend: Backend | None = None``. Phase 7-B removes
    the default; the dataclass field should be required."""
    from dataclasses import MISSING, fields

    from model_shard.expert_orchestrator import ExpertOrchestrator
    backend_field = next(f for f in fields(ExpertOrchestrator) if f.name == "backend")
    assert backend_field.default is MISSING and backend_field.default_factory is MISSING


def test_node_lm_property_removed():
    """Phase 7-A added Node._lm as a back-compat @property. Phase 7-B removes it."""
    from model_shard.node import Node
    assert not isinstance(vars(Node).get("_lm"), property), (
        "Node._lm property should have been removed in Phase 7-B"
    )
