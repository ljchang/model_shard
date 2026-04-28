"""Deterministic routing correctness: given known peer loads, the orchestrator
consistently picks the less-loaded candidate for multi-owner experts."""

from __future__ import annotations

import random
from typing import Any

import mlx.core as mx
import pytest

from model_shard.backends import MLXBackend
from model_shard.expert_orchestrator import ExpertOrchestrator, PeerRPC
from model_shard.mlx_engine import embed_tokens, make_cache, make_masks


class _CountingRpc(PeerRPC):
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[int]]] = []

    def call(
        self, peer_shard_id: str, request_id: str, layer_idx: int,
        expert_ids: list[int], h: mx.array,
        provenance_pb_out: list | None = None,
        provenance_pb_in: list | None = None,
    ) -> dict[int, mx.array]:
        self.calls.append((peer_shard_id, sorted(expert_ids)))
        # Return a zero tensor for every requested expert, same shape as h.
        return {int(eid): mx.zeros_like(h) for eid in expert_ids}


def _advance_to_layer(
    lm: Any, layer_idx: int
) -> tuple[mx.array, list[Any], tuple[Any, Any]]:
    tokens = mx.array([[1, 42, 99]])
    h = embed_tokens(lm, tokens)
    cache = make_cache(lm)
    gm, sm = make_masks(lm, h, cache)
    for i in range(layer_idx):
        layer = lm.text_model.layers[i]
        c = cache[lm.text_model.layer_idx_to_cache_idx[i]]
        mask = gm if layer.layer_type == "full_attention" else sm
        h = layer(h, mask, c, per_layer_input=None)
    return h, cache, (gm, sm)


@pytest.mark.slow
def test_multi_owner_orchestrator_picks_less_loaded_peer(loaded_model: Any) -> None:
    """Expert 0 lives on 'head' (self) and 'peer'. Peer reports a massive
    load; self is low. No peer RPC should include expert 0."""
    lm = loaded_model
    layer_idx = 15

    owners = {"head": set(range(128)), "peer": {0, 1, 2}}
    rpc = _CountingRpc()
    orch = ExpertOrchestrator(
        self_shard_id="head",
        owners=owners,
        peer_rpc=rpc,
        rpc_timeout_s=1.0,
        loads_provider=lambda: {"peer": 1_000_000, "head": 10},
        rng=random.Random(0),
        backend=MLXBackend.from_loaded_model(lm),
    )
    try:
        h, cache, masks = _advance_to_layer(lm, layer_idx)
        out = orch.run_split_layer(
            h=h, layer_idx=layer_idx, cache=cache,
            masks=masks, request_id="r1",
        )
        mx.eval(out)
        for peer, eids in rpc.calls:
            for e in eids:
                assert e not in (0, 1, 2), (
                    f"expert {e} went to peer {peer!r} despite peer being high-load"
                )
    finally:
        orch.close()


@pytest.mark.slow
def test_multi_owner_orchestrator_picks_peer_when_self_overloaded(
    loaded_model: Any,
) -> None:
    """Inverse: self is massively loaded, peer is idle. If expert 0
    appears in the batch's top-k, it must go to peer."""
    lm = loaded_model
    layer_idx = 15

    owners = {"head": set(range(128)), "peer": {0}}
    rpc_asked: list[int] = []

    class _EchoRpc(PeerRPC):
        def call(
            self, peer_shard_id: str, request_id: str, layer_idx: int,
            expert_ids: list[int], h: mx.array,
            provenance_pb_out: list | None = None,
            provenance_pb_in: list | None = None,
        ) -> dict[int, mx.array]:
            for eid in expert_ids:
                rpc_asked.append(int(eid))
            return {int(eid): mx.zeros_like(h) for eid in expert_ids}

    orch = ExpertOrchestrator(
        self_shard_id="head",
        owners=owners,
        peer_rpc=_EchoRpc(),
        rpc_timeout_s=1.0,
        loads_provider=lambda: {"peer": 10, "head": 1_000_000},
        rng=random.Random(0),
        backend=MLXBackend.from_loaded_model(lm),
    )
    try:
        h, cache, masks = _advance_to_layer(lm, layer_idx)
        orch.run_split_layer(
            h=h, layer_idx=layer_idx, cache=cache,
            masks=masks, request_id="r2",
        )
        # If expert 0 appeared in top-k anywhere, it must have gone to peer.
        # Peer only owns 0, so any rpc_asked entries are necessarily 0.
        assert all(e == 0 for e in rpc_asked), (
            f"peer was asked for non-0 experts: {set(rpc_asked)}"
        )
    finally:
        orch.close()
