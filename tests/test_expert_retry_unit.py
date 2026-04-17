"""Phase 6-A expert-retry unit tests."""
from __future__ import annotations

import pytest

from model_shard.expert_orchestrator import ExpertRpcFailure


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
