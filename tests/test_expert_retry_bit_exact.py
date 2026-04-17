"""Phase 6-A bit-exact correctness: retry output == no-failure output."""
from __future__ import annotations

import random

import mlx.core as mx
import pytest

from model_shard.expert_orchestrator import ExpertOrchestrator
from model_shard.mlx_engine import load_model
from model_shard.moe import run_selected_experts

pytestmark = pytest.mark.slow

_HF_ID = "mlx-community/gemma-4-26b-a4b-it-4bit"
_LAYER = 15


@pytest.fixture(scope="module")
def lm():
    return load_model(_HF_ID)


class _SharedLmPeerRPC:
    """Test double: 'peer RPC' that just runs run_selected_experts on a
    shared lm — mimics what a real peer would produce. Supports one-shot
    failure injection on named peers."""

    def __init__(self, lm, fail_once_for: set[str] | None = None) -> None:
        self._lm = lm
        self._fail_once_for = set(fail_once_for or set())
        self._already_failed: set[str] = set()

    def call(
        self, peer_shard_id, request_id, layer_idx, expert_ids, h
    ):
        if (
            peer_shard_id in self._fail_once_for
            and peer_shard_id not in self._already_failed
        ):
            self._already_failed.add(peer_shard_id)
            raise RuntimeError(f"injected fail on {peer_shard_id}")
        return run_selected_experts(self._lm, h, layer_idx, list(expert_ids))


def test_retry_output_matches_no_failure_output(lm):
    # Setup: "peers" B, C, D own disjoint subsets. E replicates expert 3;
    # F replicates expert 6 so that every B-owned expert has a retry target.
    owners = {
        "B": {3, 6},
        "C": {7, 10},
        "D": {11, 14},
        "E": {3},  # replica of 3 — retry target after B fails on expert 3
        "F": {6},  # replica of 6 — retry target after B fails on expert 6
    }

    def live_owners(eid: int) -> set[str]:
        return {sid for sid, ids in owners.items() if eid in ids}

    ids_to_fan = [3, 6, 7, 10, 11, 14]

    # Synthetic input: stays on no-sort path per 5a §7.5.
    mx.random.seed(7)
    hidden = lm.text_model.layers[_LAYER].pre_feedforward_layernorm_2.weight.shape[0]
    post_attn = mx.random.normal((1, 3, hidden)).astype(mx.bfloat16)

    # Baseline: no failure.
    rpc_nofail = _SharedLmPeerRPC(lm, fail_once_for=set())
    orch_nofail = ExpertOrchestrator(
        self_shard_id="self",
        owners=owners,
        peer_rpc=rpc_nofail,
        rpc_timeout_s=5.0,
        rng=random.Random(0),
        live_owners_provider=live_owners,
        retry_max_attempts=3,
        retry_backoff_ms=(0, 0),
    )
    baseline = orch_nofail._phase_b_with_retry(
        post_attn=post_attn,
        all_ids=ids_to_fan,
        layer_idx=_LAYER,
        request_id="r-base",
        initial_local_ids=[],
        lm=lm,
    )
    orch_nofail.close()

    # With-failure: peer B fails once on expert 3; retry lands on E.
    rpc_fail = _SharedLmPeerRPC(lm, fail_once_for={"B"})
    orch_fail = ExpertOrchestrator(
        self_shard_id="self",
        owners=owners,
        peer_rpc=rpc_fail,
        rpc_timeout_s=5.0,
        rng=random.Random(0),
        live_owners_provider=live_owners,
        retry_max_attempts=3,
        retry_backoff_ms=(0, 0),
    )
    with_fail = orch_fail._phase_b_with_retry(
        post_attn=post_attn,
        all_ids=ids_to_fan,
        layer_idx=_LAYER,
        request_id="r-fail",
        initial_local_ids=[],
        lm=lm,
    )
    orch_fail.close()

    # Bit-exact.
    assert set(baseline.keys()) == set(with_fail.keys()) == set(ids_to_fan)
    for eid in ids_to_fan:
        assert mx.array_equal(baseline[eid], with_fail[eid]).item(), (
            f"expert {eid} differs between no-failure and with-failure runs"
        )
