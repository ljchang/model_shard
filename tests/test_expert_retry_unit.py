"""Phase 6-A expert-retry unit tests."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from model_shard.expert_orchestrator import ExpertOrchestrator, ExpertRpcFailure


def test_expert_rpc_failure_has_typed_fields():
    exc = ExpertRpcFailure(
        "peer 'B' died", failed_peer="B", layer_idx=15
    )
    assert exc.failed_peer == "B"
    assert exc.layer_idx == 15
    assert "peer 'B' died" in str(exc)


def test_expert_rpc_failure_rejects_missing_typed_fields():
    # Positional message-only construction should fail — these fields are required.
    with pytest.raises(TypeError):
        ExpertRpcFailure("something broke")  # type: ignore[call-arg]


def test_orchestrator_accepts_retry_fields_defaults():
    orch = ExpertOrchestrator(
        self_shard_id="A",
        owners={"A": {3}},
        peer_rpc=MagicMock(),
        rpc_timeout_s=1.0,
    )
    assert orch.retry_max_attempts == 3
    assert orch.retry_backoff_ms == (100, 500)


def test_orchestrator_accepts_explicit_retry_fields():
    orch = ExpertOrchestrator(
        self_shard_id="A",
        owners={"A": {3}},
        peer_rpc=MagicMock(),
        rpc_timeout_s=1.0,
        retry_max_attempts=5,
        retry_backoff_ms=(10, 50, 200),
    )
    assert orch.retry_max_attempts == 5
    assert orch.retry_backoff_ms == (10, 50, 200)
