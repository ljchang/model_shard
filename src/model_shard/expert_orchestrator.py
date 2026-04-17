"""Fan-out / fan-in coordinator for expert-level sharded layers (Phase 3).

The orchestrator composes the pure helpers in ``model_shard.moe`` to run
one decoder layer with its 128 experts partitioned across shards. Task 9
proved the helpers reproduce the atomic ``layer(h, mask, c)`` path bit-
for-bit when assembled a particular way; this module assembles them the
same way.

Task 12 wires up ``TcpPeerRPC`` — the production ``PeerRPC`` backed by the
Phase 1 envelope transport — and parallelizes the peer fan-out with a
thread pool. The all-local path (no peer RPCs) remains the fast path when
a single node owns every routed expert for a given layer.
"""

from __future__ import annotations

import contextlib
import socket
import threading
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol

import mlx.core as mx

from model_shard._pb import wire_pb2
from model_shard.envelope import recv_envelope, send_envelope
from model_shard.mlx_engine import bytes_to_tensor, tensor_to_bytes
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


def _dtype_to_wire(dt: mx.Dtype) -> int:
    if dt == mx.bfloat16:
        return int(wire_pb2.DTYPE_BFLOAT16)
    if dt == mx.float32:
        return int(wire_pb2.DTYPE_FLOAT32)
    if dt == mx.float16:
        return int(wire_pb2.DTYPE_FLOAT16)
    raise ValueError(f"unsupported activation dtype: {dt}")


class TcpPeerRPC:
    """PeerRPC backed by the Phase 1 TCP envelope transport.

    Opens a short-lived connection per call for simplicity (Phase 3 prototype
    scope); a later phase can persist connections and multiplex requests.
    The ``tensor_to_bytes`` / ``bytes_to_tensor`` pair in ``mlx_engine`` is
    byte-exact for bf16, so the round-trip preserves the activation exactly.
    """

    def __init__(
        self,
        addresses: dict[str, tuple[str, int]],
        timeout_s: float,
    ) -> None:
        self._addresses = addresses
        self._timeout_s = timeout_s

    def call(
        self,
        peer_shard_id: str,
        request_id: str,
        layer_idx: int,
        expert_ids: list[int],
        h: mx.array,
    ) -> dict[int, mx.array]:
        host, port = self._addresses[peer_shard_id]
        s = socket.create_connection((host, port), timeout=self._timeout_s)
        s.settimeout(self._timeout_s)
        try:
            stream = s.makefile("rwb")
            req = wire_pb2.Envelope()
            req.expert_request.protocol_version = 1
            req.expert_request.request_id = request_id
            req.expert_request.layer_idx = layer_idx
            req.expert_request.expert_ids.extend(expert_ids)
            # ``tensor_to_bytes`` takes only the array; the descriptor
            # fields (shape/dtype/byte_count) are populated here so the
            # receiver can rehydrate without guessing.
            raw = tensor_to_bytes(h)
            req.expert_request.h_spec.shape.extend(list(h.shape))
            req.expert_request.h_spec.dtype = _dtype_to_wire(h.dtype)
            req.expert_request.h_spec.quant = wire_pb2.QUANT_NONE
            req.expert_request.h_spec.byte_count = len(raw)
            send_envelope(stream, req, raw)
            stream.flush()

            env, tensor = recv_envelope(stream)
            if env.WhichOneof("payload") == "error":
                raise RuntimeError(
                    f"peer {peer_shard_id} returned error "
                    f"{env.error.code}: {env.error.detail}"
                )
            if env.WhichOneof("payload") != "expert_response":
                raise RuntimeError(
                    f"unexpected payload from peer {peer_shard_id}: "
                    f"{env.WhichOneof('payload')}"
                )
            resp = env.expert_response
            stacked = bytes_to_tensor(
                tensor,
                shape=list(resp.outputs_spec.shape),
                dtype=resp.outputs_spec.dtype,
            )
            # Unstack along axis 2: [B, L, len(expert_ids), hidden] →
            # {eid: [B, L, hidden]}.
            return {
                int(eid): stacked[:, :, j, :]
                for j, eid in enumerate(resp.expert_ids)
            }
        finally:
            s.close()


@dataclass
class ExpertOrchestrator:
    """Runs one decoder layer with experts partitioned across shards.

    Not frozen: holds a ``ThreadPoolExecutor`` for parallel peer fan-out.
    The executor lifetime matches the orchestrator instance. Create one
    orchestrator per node for the lifetime of the decode loop, and call
    ``close()`` on shutdown to release the fan-out executor threads.
    """

    self_shard_id: str
    owners: Mapping[str, set[int]]
    peer_rpc: PeerRPC
    rpc_timeout_s: float
    # Optional process-wide lock serializing MLX graph construction across
    # threads. Required when multiple in-process nodes share a single MLX
    # runtime (the test fixture); harmless in production (each node is a
    # separate process, so the lock never contends with anything).
    mlx_lock: threading.Lock | None = None
    _executor: ThreadPoolExecutor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # One worker per potential peer is plenty for Phase 3's 3-node cluster.
        # ``max_workers=8`` leaves headroom without over-subscribing.
        self._executor = ThreadPoolExecutor(
            max_workers=8, thread_name_prefix="expert-rpc"
        )

    def close(self) -> None:
        """Shut down the fan-out executor. Idempotent."""
        self._executor.shutdown(wait=False, cancel_futures=True)

    @contextlib.contextmanager
    def _mlx_guard(self) -> Iterator[None]:
        if self.mlx_lock is None:
            yield
            return
        self.mlx_lock.acquire()
        try:
            yield
        finally:
            self.mlx_lock.release()

    def run_split_layer(
        self,
        lm: Any,
        h: mx.array,
        layer_idx: int,
        cache: list[Any],
        masks: tuple[Any, Any],
        request_id: str,
    ) -> mx.array:
        # Phase A — local MLX graph construction for attention, routing, the
        # shared expert, and any experts this shard owns. Guarded by
        # ``mlx_lock`` so peer-handler threads (running on the same Python
        # process in the in-process test fixture) cannot race the default MLX
        # stream with concurrent graph construction.
        with self._mlx_guard():
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
            # Force the local compute graph to realize before releasing the
            # lock; otherwise the peer handlers could start evaluating on the
            # default stream while our local graph is still being built.
            mx.eval(post_attn, shared_out, *local_outputs.values())

        # Phase B — peer fan-out. Lock is deliberately not held here: peer
        # threads need to acquire it to run their experts.
        outputs: dict[int, mx.array] = dict(local_outputs)
        futures = {
            peer: self._executor.submit(
                self.peer_rpc.call, peer, request_id, layer_idx, ids, post_attn
            )
            for peer, ids in by_owner.items()
        }
        for peer, fut in futures.items():
            try:
                outputs.update(fut.result(timeout=self.rpc_timeout_s))
            except Exception as e:
                raise RuntimeError(
                    f"expert RPC to {peer} failed for layer {layer_idx}: {e}"
                ) from e

        # Phase C — aggregation + outer ops. Re-acquire the lock for the
        # final graph construction.
        with self._mlx_guard():
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
            mx.eval(out)
        return out


__all__ = ["ExpertOrchestrator", "PeerRPC", "TcpPeerRPC"]
