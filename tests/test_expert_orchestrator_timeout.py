"""ExpertOrchestrator surfaces peer RPC failures as ExpertRpcFailure."""

from __future__ import annotations

import mlx.core as mx
import pytest

from model_shard.expert_orchestrator import (
    ExpertOrchestrator,
    ExpertRpcFailure,
    PeerRPC,
)


class _FailingRpc(PeerRPC):
    def call(self, *a, **kw):  # type: ignore[no-untyped-def]
        raise TimeoutError("simulated peer timeout")


@pytest.mark.slow
def test_orchestrator_rpc_failure_raises_expert_rpc_failure(loaded_model) -> None:  # type: ignore[no-untyped-def]
    lm = loaded_model
    owners = {"self": {3, 6}, "dead": {0, 1, 2, 4, 5} | set(range(7, 128))}
    orch = ExpertOrchestrator(
        self_shard_id="self",
        owners=owners,
        peer_rpc=_FailingRpc(),
        rpc_timeout_s=0.1,
    )
    from model_shard.mlx_engine import embed_tokens, make_cache, make_masks
    tokens = mx.array([[1, 2, 3]])
    h = embed_tokens(lm, tokens)
    cache = make_cache(lm)
    gm, sm = make_masks(lm, h, cache)
    for i in range(15):
        layer = lm.text_model.layers[i]
        c = cache[lm.text_model.layer_idx_to_cache_idx[i]]
        mask = gm if layer.layer_type == "full_attention" else sm
        h = layer(h, mask, c, per_layer_input=None)

    with pytest.raises(ExpertRpcFailure, match="peer 'dead'"):
        orch.run_split_layer(lm, h=h, layer_idx=15, cache=cache, masks=(gm, sm), request_id="r1")
    orch.close()
