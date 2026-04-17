"""Server-side handler for ExpertWeightRequest (source side of migration)."""
from __future__ import annotations

import pytest

from model_shard.migration import ExpertWeightPeerRPC

pytestmark = pytest.mark.slow


def test_source_slices_and_transfers_held_expert(partial_load_fixture):
    _node, port = partial_load_fixture
    rpc = ExpertWeightPeerRPC(
        addresses={"src": ("127.0.0.1", port)}, timeout_s=30.0
    )
    tensors = rpc.pull(source_shard_id="src", layer_idx=15, expert_id=3)
    assert len(tensors) == 9


def test_source_returns_error_on_unheld(partial_load_fixture):
    _node, port = partial_load_fixture
    rpc = ExpertWeightPeerRPC(
        addresses={"src": ("127.0.0.1", port)}, timeout_s=30.0
    )
    with pytest.raises(RuntimeError, match="not held"):
        rpc.pull(source_shard_id="src", layer_idx=15, expert_id=1)
