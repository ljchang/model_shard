"""Fan-out / fan-in coordinator for expert-level sharded layers (Phase 3).

The orchestrator composes the pure helpers in ``model_shard.moe`` to run
one decoder layer with its 128 experts partitioned across shards. Task 9
proved the helpers reproduce the atomic ``layer(h, mask, c)`` path bit-
for-bit when assembled a particular way; this module assembles them the
same way.

In this task only the all-local path is exercised — peers own no experts,
so ``PeerRPC.call`` is never invoked. Task 12 wires up a real ``TcpPeerRPC``
implementation behind the same ``PeerRPC`` Protocol.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import mlx.core as mx

from model_shard.moe import (
    aggregate_experts,
    group_expert_ids_by_owner,
    run_attention_and_route,
    run_selected_experts,
    run_shared_expert,
)


class PeerRPC(Protocol):
    def call(
        self,
        peer_shard_id: str,
        request_id: str,
        layer_idx: int,
        expert_ids: list[int],
        h: mx.array,
    ) -> dict[int, mx.array]:
        """Send an ExpertRequest to ``peer_shard_id``, block for the
        ExpertResponse, and return ``{expert_id: output tensor}``. Must
        raise on timeout or RPC error."""
        ...


@dataclass(frozen=True)
class ExpertOrchestrator:
    self_shard_id: str
    owners: Mapping[str, set[int]]
    peer_rpc: PeerRPC
    rpc_timeout_s: float

    def run_split_layer(
        self,
        lm: Any,
        h: mx.array,
        layer_idx: int,
        cache: list[Any],
        masks: tuple[Any, Any],
        request_id: str,
    ) -> mx.array:
        post_attn, top_k_ids, top_k_weights = run_attention_and_route(
            lm, h, layer_idx, cache, masks
        )
        mx.eval(top_k_ids)
        # Union of all top-k ids across the batch and sequence.
        all_ids = sorted(
            {int(e) for e in top_k_ids.reshape(-1).tolist()}  # type: ignore[arg-type,union-attr]
        )
        by_owner = group_expert_ids_by_owner(all_ids, self.owners)

        local_ids = by_owner.pop(self.self_shard_id, [])
        shared_out = run_shared_expert(lm, post_attn, layer_idx)
        local_outputs = run_selected_experts(lm, post_attn, layer_idx, local_ids)

        # Serial peer RPC for the local-only test; Task 12 parallelizes this.
        outputs: dict[int, mx.array] = dict(local_outputs)
        for peer, ids in by_owner.items():
            peer_outputs = self.peer_rpc.call(
                peer, request_id, layer_idx, ids, post_attn
            )
            outputs.update(peer_outputs)

        # Aggregate per position — same shape pattern as Task 9's proof.
        layer = lm.text_model.layers[layer_idx]
        post_ffn_ln_2 = layer.post_feedforward_layernorm_2
        h1_plus_h2 = mx.zeros_like(post_attn)
        for b in range(top_k_ids.shape[0]):
            for ll in range(top_k_ids.shape[1]):
                ids = [int(x) for x in top_k_ids[b, ll].tolist()]  # type: ignore[arg-type,union-attr]
                per_pos = {
                    eid: outputs[eid][b : b + 1, ll : ll + 1, :] for eid in ids
                }
                weights = top_k_weights[b : b + 1, ll : ll + 1, :]
                per_pos_shared = shared_out[b : b + 1, ll : ll + 1, :]
                agg = aggregate_experts(
                    per_pos, ids, weights, per_pos_shared, post_ffn_ln_2
                )
                # Splice position ll of h1_plus_h2 with the per-position agg.
                h1_plus_h2 = (
                    mx.concatenate(
                        [h1_plus_h2[:, :ll, :], agg, h1_plus_h2[:, ll + 1 :, :]],
                        axis=1,
                    )
                    if h1_plus_h2.shape[1] > 1
                    else agg
                )

        # Outer layer ops from DecoderLayer.__call__ lines 83-88, 107:
        #   h = post_feedforward_layernorm(h1 + h2)
        #   h = residual_2 + h
        #   h = h * layer_scalar
        # The per-layer-input gating branch (lines 92-105) is skipped here
        # because Gemma 4 26B has hidden_size_per_layer_input=0, so the gate
        # modules are None. If that assumption changes, add a guard.
        out: mx.array = layer.post_feedforward_layernorm(h1_plus_h2)
        out = post_attn + out
        if layer.layer_scalar is not None:
            out = out * layer.layer_scalar
        return out


__all__ = ["ExpertOrchestrator", "PeerRPC"]
