"""Observer integration: peer leaving ALIVE aborts in-flight expert RPC.

When Phase 2's membership observer fires for a peer whose RPC is in
flight, the orchestrator must abort that RPC immediately rather than
waiting for the TCP timeout (which may be seconds).
"""

from __future__ import annotations

import threading
import time

import mlx.core as mx
import pytest

from model_shard.backends import MLXBackend
from model_shard.expert_orchestrator import (
    ExpertOrchestrator,
    ExpertRpcFailure,
    PeerRPC,
)


class _SlowRpc(PeerRPC):
    def __init__(self, delay_s: float = 30.0) -> None:
        self._delay = delay_s

    def call(self, *a, **kw):  # type: ignore[no-untyped-def]
        time.sleep(self._delay)
        return {}


@pytest.mark.slow
def test_observer_aborts_in_flight_rpc(loaded_model) -> None:  # type: ignore[no-untyped-def]
    lm = loaded_model
    owners = {"self": set(), "peer": set(range(128))}
    orch = ExpertOrchestrator(
        self_shard_id="self",
        owners=owners,
        peer_rpc=_SlowRpc(delay_s=30.0),
        rpc_timeout_s=30.0,
        backend=MLXBackend.from_loaded_model(lm),
    )
    from model_shard.mlx_engine import embed_tokens, make_cache, make_masks
    tokens = mx.array([[1, 2]])
    h = embed_tokens(lm, tokens)
    cache = make_cache(lm)
    gm, sm = make_masks(lm, h, cache)
    for i in range(15):
        layer = lm.text_model.layers[i]
        c = cache[lm.text_model.layer_idx_to_cache_idx[i]]
        mask = gm if layer.layer_type == "full_attention" else sm
        h = layer(h, mask, c, per_layer_input=None)

    # Fire the observer from another thread after 0.5s.
    def fire() -> None:
        time.sleep(0.5)
        orch.notify_peer_left_alive("peer")

    threading.Thread(target=fire, daemon=True).start()

    t0 = time.monotonic()
    with pytest.raises(ExpertRpcFailure, match="peer 'peer' left ALIVE"):
        orch.run_split_layer(
            h=h, layer_idx=15, cache=cache, masks=(gm, sm), request_id="r"
        )
    elapsed = time.monotonic() - t0
    assert elapsed < 5.0, (
        f"observer abort did not short-circuit the 30s timeout (took {elapsed:.1f}s)"
    )
    orch.close()
