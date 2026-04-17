"""Expert orchestrator with all experts on the local node (no RPC).

This test locks in that ``ExpertOrchestrator`` composes the ``moe`` helpers
correctly: when every expert id is claimed by the local shard, the outcome
must be bit-exact equal to the atomic ``layer(h, mask, c)`` path. Task 9's
split-equivalence proof already established that the helpers themselves
reproduce the atomic result; this test pins the orchestrator's plumbing to
that same baseline.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import pytest

from model_shard.expert_orchestrator import ExpertOrchestrator, PeerRPC
from model_shard.mlx_engine import embed_tokens, make_cache, make_masks


class _NoRpc(PeerRPC):
    """Fail loudly if the orchestrator tries to RPC in the all-local case."""

    def call(
        self,
        peer_shard_id: str,
        request_id: str,
        layer_idx: int,
        expert_ids: list[int],
        h: mx.array,
        provenance_pb_out: list | None = None,
        provenance_pb_in: list | None = None,
    ) -> dict[int, mx.array]:
        raise AssertionError("should not be called when all experts are local")


def _replay_through(
    lm: Any, tokens: mx.array, layer_idx: int
) -> tuple[mx.array, Any, tuple[Any, Any]]:
    """Embed + run layers [0, layer_idx) on a fresh cache. Returns (h, cache, masks)."""
    h = embed_tokens(lm, tokens)
    cache = make_cache(lm)
    gm, sm = make_masks(lm, h, cache)
    tm = lm.text_model
    for i in range(layer_idx):
        layer = tm.layers[i]
        c = cache[tm.layer_idx_to_cache_idx[i]]
        mask = gm if layer.layer_type == "full_attention" else sm
        h = layer(h, mask, c, per_layer_input=None)
    return h, cache, (gm, sm)


@pytest.mark.slow
def test_orchestrator_all_local_matches_atomic(loaded_model: Any) -> None:
    lm = loaded_model
    layer_idx = 15
    tokens = mx.array([[1, 42, 99]])

    # Orchestrator path: local owns ALL 128 experts; peers own none.
    orch = ExpertOrchestrator(
        self_shard_id="head",
        owners={"head": set(range(128)), "mid": set(), "tail": set()},
        peer_rpc=_NoRpc(),
        rpc_timeout_s=1.0,
    )
    h_orch, cache_orch, masks_orch = _replay_through(lm, tokens, layer_idx)
    out_orch = orch.run_split_layer(
        lm,
        h=h_orch,
        layer_idx=layer_idx,
        cache=cache_orch,
        masks=masks_orch,
        request_id="r1",
    )

    # Atomic baseline: replay layers 0..14 on a SECOND fresh cache, then run
    # layer 15 atomically. Bit-exact comparison locks in that Orchestrator
    # composes the helpers the same way the Task 9 proof did.
    h_atom, cache_atom, (gm2, sm2) = _replay_through(lm, tokens, layer_idx)
    layer = lm.text_model.layers[layer_idx]
    c15 = cache_atom[lm.text_model.layer_idx_to_cache_idx[layer_idx]]
    mask15 = gm2 if layer.layer_type == "full_attention" else sm2
    out_atomic = layer(h_atom, mask15, c15, per_layer_input=None)

    mx.eval(out_orch, out_atomic)
    assert out_orch.shape == h_orch.shape
    max_diff = mx.max(mx.abs(out_orch - out_atomic)).item()
    assert mx.array_equal(out_orch, out_atomic), (
        f"orchestrator != atomic; max abs diff = {max_diff}"
    )


@pytest.mark.slow
def test_orchestrator_multi_owner_routes_to_less_loaded(loaded_model: Any) -> None:
    """Expert 0 has 2 owners (self=head + peer). Peer reports extreme load;
    self is low. Expert 0 stays local — no peer RPC."""
    import random as _random

    lm = loaded_model
    layer_idx = 15

    class _RecordingRpc(PeerRPC):
        def __init__(self) -> None:
            self.calls: list[tuple[str, list[int]]] = []

        def call(
            self,
            peer_shard_id: str,
            request_id: str,
            layer_idx: int,
            expert_ids: list[int],
            h: mx.array,
            provenance_pb_out: list | None = None,
            provenance_pb_in: list | None = None,
        ) -> dict[int, mx.array]:
            self.calls.append((peer_shard_id, sorted(expert_ids)))
            raise AssertionError(
                "peer RPC should not be reached when peer is swamped"
            )

    rpc = _RecordingRpc()
    owners = {"head": set(range(128)), "peer": {0, 1, 2}}
    orch = ExpertOrchestrator(
        self_shard_id="head",
        owners=owners,
        peer_rpc=rpc,
        rpc_timeout_s=1.0,
        loads_provider=lambda: {"peer": 1_000_000, "head": 10},
        rng=_random.Random(0),
    )
    try:
        tokens = mx.array([[1, 42, 99]])
        h = embed_tokens(lm, tokens)
        cache = make_cache(lm)
        gm, sm = make_masks(lm, h, cache)
        for i in range(layer_idx):
            layer = lm.text_model.layers[i]
            c = cache[lm.text_model.layer_idx_to_cache_idx[i]]
            mask = gm if layer.layer_type == "full_attention" else sm
            h = layer(h, mask, c, per_layer_input=None)

        _ = orch.run_split_layer(
            lm,
            h=h,
            layer_idx=layer_idx,
            cache=cache,
            masks=(gm, sm),
            request_id="r1",
        )
        # Peer should not have been called for any of experts 0, 1, 2.
        for peer, eids in rpc.calls:
            for e in eids:
                assert e not in (0, 1, 2), (
                    f"expert {e} went to peer {peer!r} despite peer being high-load"
                )
    finally:
        orch.close()
