"""Phase 7-A Task 4: Node accepts a `backend` kwarg."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from model_shard.backends import Backend, MLXBackend
from model_shard.node import Node
from model_shard.shard_map import NodeAddress, ShardMap, ShardSpec


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("ENABLE_GOSSIP", "false")
    monkeypatch.setenv("ENABLE_PARTIAL_LOAD", "false")
    monkeypatch.setenv("ENABLE_DYNAMIC_MIGRATION", "false")
    monkeypatch.setenv("ENABLE_EXPERT_RETRY", "false")
    yield


def _mk_spec(sid: str, port: int) -> ShardSpec:
    return ShardSpec(
        shard_id=sid,
        address=NodeAddress(host="127.0.0.1", port=port),
        start_layer=0, end_layer=30, moe_experts={},
    )


def test_node_default_backend_is_mlx_backend_wrapping_loaded_model():
    """Legacy path: Node(loaded_model=<mock>) wraps it in MLXBackend.from_loaded_model."""
    spec_a = _mk_spec("A", 32000)
    spec_b = _mk_spec("B", 32001)
    sm = ShardMap({"A": spec_a, "B": spec_b})
    lm = MagicMock()
    n = Node(shard=spec_a, shard_map=sm, loaded_model=lm, total_layers=30)
    assert isinstance(n._backend, Backend)
    assert n._backend.name == "mlx"
    assert n._backend._lm is lm  # MLXBackend holds the lm internally


def test_node_accepts_explicit_backend():
    """Explicit backend kwarg is honored over loaded_model."""
    spec_a = _mk_spec("A", 32002)
    spec_b = _mk_spec("B", 32003)
    sm = ShardMap({"A": spec_a, "B": spec_b})
    lm = MagicMock()
    b = MLXBackend.from_loaded_model(lm)
    n = Node(shard=spec_a, shard_map=sm, backend=b, total_layers=30)
    assert n._backend is b


def test_node_passes_mlx_lock_into_backend():
    """Node's _MLX_COMPUTE_LOCK is passed into MLXBackend.__init__ so
    slice/attach/detach serialize against concurrent compute."""
    from model_shard.node import _MLX_COMPUTE_LOCK
    spec_a = _mk_spec("A", 32006)
    spec_b = _mk_spec("B", 32007)
    sm = ShardMap({"A": spec_a, "B": spec_b})
    lm = MagicMock()
    n = Node(shard=spec_a, shard_map=sm, loaded_model=lm, total_layers=30)
    # Either the Node set the backend's lock to _MLX_COMPUTE_LOCK at init,
    # or the backend's lock is a private one. Prefer the former.
    assert n._backend._mlx_lock is _MLX_COMPUTE_LOCK
