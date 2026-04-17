"""Phase 6-A expert-retry unit tests."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import mlx.core as mx
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


@dataclass
class _FlakyPeerRPC:
    """Test double: peer_rpc that fails on a set of peer shard_ids the first
    time they're called, then succeeds thereafter. Returns per-expert fake
    tensors keyed by expert id."""

    fail_once_for: set[str] = field(default_factory=set)
    _already_failed: set[str] = field(default_factory=set)
    calls: list[tuple[str, list[int]]] = field(default_factory=list)

    def call(
        self,
        peer_shard_id: str,
        request_id: str,
        layer_idx: int,
        expert_ids: list[int],
        h: mx.array,
    ) -> dict[int, mx.array]:
        self.calls.append((peer_shard_id, list(expert_ids)))
        if peer_shard_id in self.fail_once_for and peer_shard_id not in self._already_failed:
            self._already_failed.add(peer_shard_id)
            raise RuntimeError(f"injected failure for {peer_shard_id}")
        return {
            eid: mx.full((1, 1, 8), float(eid), dtype=mx.bfloat16)
            for eid in expert_ids
        }


def _run_test_fanout(
    *,
    owners: dict[str, set[int]],
    ids_to_fan: list[int],
    peer_rpc: Any,
    live_owners: dict[int, set[str]],
    max_attempts: int = 3,
    backoff_ms: tuple[int, ...] = (0, 0),
) -> tuple[dict[int, mx.array], Any]:
    """Exercise just the Phase B retry loop by constructing an orchestrator
    and invoking its internal `_phase_b_with_retry` helper. Returns (outputs, orch)."""
    import random

    orch = ExpertOrchestrator(
        self_shard_id="self",
        owners=owners,
        peer_rpc=peer_rpc,
        rpc_timeout_s=1.0,
        rng=random.Random(0),
        live_owners_provider=lambda eid: live_owners.get(eid, set()),
        retry_max_attempts=max_attempts,
        retry_backoff_ms=backoff_ms,
    )
    post_attn = mx.zeros((1, 1, 8), dtype=mx.bfloat16)
    outputs = orch._phase_b_with_retry(
        post_attn=post_attn,
        all_ids=ids_to_fan,
        layer_idx=15,
        request_id="req-1",
        initial_local_ids=[],
        lm=None,  # local retry not exercised in these tests.
    )
    orch.close()
    return outputs, orch


def test_retry_succeeds_on_second_attempt_to_replica():
    owners = {"B": {7}, "C": {7}}
    live = {7: {"B", "C"}}
    rpc = _FlakyPeerRPC(fail_once_for={"B"})
    outputs, _ = _run_test_fanout(
        owners=owners, ids_to_fan=[7], peer_rpc=rpc, live_owners=live,
    )
    assert 7 in outputs
    peers_called = [p for p, _ in rpc.calls]
    assert "B" in peers_called
    assert "C" in peers_called


def test_partial_outputs_preserved_across_retry():
    # B owns {3}, C owns {7}, D owns {11}. B fails; C and D succeed.
    # After retry (B's work re-routed), expect all three outputs.
    owners = {"B": {3}, "C": {7}, "D": {11}, "E": {3}}
    live = {3: {"B", "E"}, 7: {"C"}, 11: {"D"}}
    rpc = _FlakyPeerRPC(fail_once_for={"B"})
    outputs, _ = _run_test_fanout(
        owners=owners, ids_to_fan=[3, 7, 11],
        peer_rpc=rpc, live_owners=live,
    )
    assert set(outputs.keys()) == {3, 7, 11}
    c_calls = [ids for p, ids in rpc.calls if p == "C"]
    d_calls = [ids for p, ids in rpc.calls if p == "D"]
    assert len(c_calls) == 1
    assert len(d_calls) == 1


def test_retry_exhaustion_raises_typed_failure():
    # Single-owner expert, only owner fails — no replica, exhaust retries.
    owners = {"B": {7}}
    live = {7: {"B"}}
    rpc = _FlakyPeerRPC(fail_once_for={"B", "C"})
    with pytest.raises(ExpertRpcFailure) as excinfo:
        _run_test_fanout(
            owners=owners, ids_to_fan=[7], peer_rpc=rpc, live_owners=live,
            max_attempts=3,
        )
    assert excinfo.value.failed_peer == "B"
    assert excinfo.value.layer_idx == 15


def test_excluded_peer_stays_excluded_within_invocation():
    # B owns {3, 11}; E owns {3}; F owns {11}. B fails once on call routing both.
    # Retry should route 3 to E and 11 to F — not back to B.
    owners = {"B": {3, 11}, "E": {3}, "F": {11}}
    live = {3: {"B", "E"}, 11: {"B", "F"}}
    rpc = _FlakyPeerRPC(fail_once_for={"B"})
    outputs, _ = _run_test_fanout(
        owners=owners, ids_to_fan=[3, 11], peer_rpc=rpc, live_owners=live,
    )
    assert set(outputs.keys()) == {3, 11}
    second_calls = rpc.calls[1:]
    assert all(p != "B" for p, _ in second_calls)


def test_retry_disabled_matches_phase5b_behavior():
    # With max_attempts=1, first failure bubbles up immediately.
    owners = {"B": {7}, "C": {7}}
    live = {7: {"B", "C"}}
    rpc = _FlakyPeerRPC(fail_once_for={"B"})
    with pytest.raises(ExpertRpcFailure):
        _run_test_fanout(
            owners=owners, ids_to_fan=[7], peer_rpc=rpc, live_owners=live,
            max_attempts=1,
        )


def test_retry_to_self_cleans_up_in_flight():
    """If retry routes all missing experts to self (e.g., all peer replicas
    excluded), _in_flight must not retain a stale entry for this request."""
    # B owns {7}, self also owns {7}. B fails once; retry should route 7 to self.
    owners = {"B": {7}, "self": {7}}
    live = {7: {"B", "self"}}
    rpc = _FlakyPeerRPC(fail_once_for={"B"})

    import random

    import model_shard.expert_orchestrator as orch_mod

    # Fake lm with run_selected_experts — the helper runs local experts
    # when routed to self after exclusions.
    class _FakeLm:
        pass

    # Monkey-patch moe.run_selected_experts so the local-on-retry path succeeds.

    def _fake_rse(lm, h, layer_idx, ids):
        return {eid: mx.full((1, 1, 8), float(eid), dtype=mx.bfloat16) for eid in ids}

    orig = orch_mod.run_selected_experts
    orch_mod.run_selected_experts = _fake_rse
    try:
        orch = ExpertOrchestrator(
            self_shard_id="self",
            owners=owners,
            peer_rpc=rpc,
            rpc_timeout_s=1.0,
            rng=random.Random(0),
            live_owners_provider=lambda eid: live.get(eid, set()),
            retry_max_attempts=3,
            retry_backoff_ms=(0, 0),
        )
        post_attn = mx.zeros((1, 1, 8), dtype=mx.bfloat16)
        outputs = orch._phase_b_with_retry(
            post_attn=post_attn,
            all_ids=[7],
            layer_idx=15,
            request_id="r-leak",
            initial_local_ids=[],
            lm=_FakeLm(),
        )
        assert 7 in outputs
        # The critical assertion: no leaked _in_flight entry.
        with orch._in_flight_lock:
            assert "r-leak" not in orch._in_flight, (
                f"_in_flight leaked: {orch._in_flight}"
            )
        orch.close()
    finally:
        orch_mod.run_selected_experts = orig
