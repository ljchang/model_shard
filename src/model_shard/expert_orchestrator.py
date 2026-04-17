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
import random
import socket
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, BinaryIO, Protocol, cast

import mlx.core as mx

from model_shard._pb import wire_pb2
from model_shard.envelope import recv_envelope, send_envelope
from model_shard.mlx_engine import bytes_to_tensor, tensor_to_bytes
from model_shard.moe import (
    aggregate_experts,
    group_expert_ids_by_owner_loaded,
    run_attention_and_route,
    run_selected_experts,
    run_shared_expert,
)


class ExpertRpcFailure(RuntimeError):  # noqa: N818 — explicit name per plan
    """Raised by ExpertOrchestrator when a peer RPC fails (timeout, broken
    pipe, observer-triggered close). The node's request handler translates
    this into Error{SHARD_UNAVAILABLE, is_final=true} for the client."""


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
    if dt == mx.uint32:
        return int(wire_pb2.DTYPE_UINT32)
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
            send_envelope(cast(BinaryIO, stream), req, raw)
            stream.flush()

            env, tensor = recv_envelope(cast(BinaryIO, stream))
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
    # Phase 4: P2C routing inputs. ``loads_provider`` returns the most recent
    # peer loads (shard_id -> EMA x 100) seen via gossip; default is a no-op
    # returning ``{}``, which makes the orchestrator behave like Phase 3
    # (every candidate ties at the sentinel, single-owner experts unchanged).
    # ``rng`` is used by ``group_expert_ids_by_owner_loaded`` to sample two
    # candidates when an expert has >=3 owners.
    loads_provider: Callable[[], Mapping[str, int]] = field(
        default_factory=lambda: (lambda: {})
    )
    rng: random.Random = field(default_factory=random.Random)
    live_owners_provider: Callable[[int], set[str]] | None = None
    heat_observer: Callable[[int, list[int]], None] | None = None
    _executor: ThreadPoolExecutor = field(init=False, repr=False)
    # Observer-abort bookkeeping. Each active `run_split_layer` registers a
    # per-peer threading.Event; ``notify_peer_left_alive`` sets every event
    # whose peer matches, so the polling loop in ``run_split_layer`` can
    # short-circuit the wait and raise ``ExpertRpcFailure`` instead of
    # blocking until the TCP timeout fires. Structure:
    #   {request_id: {peer_shard_id: threading.Event}}
    # Guarded by ``_in_flight_lock`` because registrations, lookups, and
    # cleanups cross threads (the fan-out thread pool + the observer thread
    # + node.py's membership callback thread).
    _in_flight: dict[str, dict[str, threading.Event]] = field(
        init=False, repr=False, default_factory=dict
    )
    _in_flight_lock: threading.Lock = field(
        init=False, repr=False, default_factory=threading.Lock
    )

    def __post_init__(self) -> None:
        # One worker per potential peer is plenty for Phase 3's 3-node cluster.
        # ``max_workers=8`` leaves headroom without over-subscribing.
        self._executor = ThreadPoolExecutor(
            max_workers=8, thread_name_prefix="expert-rpc"
        )

    def close(self) -> None:
        """Shut down the fan-out executor. Idempotent."""
        self._executor.shutdown(wait=False, cancel_futures=True)

    def notify_peer_left_alive(self, peer_shard_id: str) -> None:
        """Signal every in-flight RPC targeting ``peer_shard_id`` to abort.

        Wired from ``node.py``'s membership observer: when a peer transitions
        out of ALIVE (SUSPECT or DEAD), any outstanding expert RPC to that
        peer would otherwise sit on its socket until the TCP timeout. Setting
        the per-peer event here lets ``run_split_layer``'s polling loop raise
        ``ExpertRpcFailure`` immediately instead.

        Safe to call for peers with no outstanding RPCs (no-op). Safe to call
        concurrently from multiple threads.
        """
        with self._in_flight_lock:
            # Snapshot the events to set so we don't hold the lock across
            # Event.set() (cheap, but keeps the critical section minimal).
            to_set: list[threading.Event] = []
            for peers in self._in_flight.values():
                ev = peers.get(peer_shard_id)
                if ev is not None:
                    to_set.append(ev)
        for ev in to_set:
            ev.set()

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

    def _gather_with_abort(
        self,
        futures: dict[str, Future[dict[int, mx.array]]],
        abort_events: dict[str, threading.Event],
        outputs: dict[int, mx.array],
        layer_idx: int,
    ) -> None:
        """Block until every peer future resolves, or any peer's abort event
        fires, or the overall ``rpc_timeout_s`` elapses.

        Polls with 100ms granularity so an observer-triggered abort short-
        circuits the TCP timeout (up to ~5s) without spin-waiting. On any
        abort or error this raises ``ExpertRpcFailure``; futures that are
        still in the queue (not yet running) are cancelled so the pool
        doesn't issue further RPCs for a dead request.
        """
        poll_s = 0.1
        deadline: float | None = None
        if self.rpc_timeout_s is not None:
            deadline = time.monotonic() + self.rpc_timeout_s

        remaining = dict(futures)
        while remaining:
            # Abort wins immediately — before we even check completions —
            # so a fast-firing observer interrupts the wait at once.
            for peer, ev in abort_events.items():
                if ev.is_set() and peer in remaining:
                    self._cancel_all(remaining.values())
                    raise ExpertRpcFailure(
                        f"peer {peer!r} left ALIVE mid-request for layer "
                        f"{layer_idx}"
                    )

            done_peers: list[str] = []
            for peer, fut in remaining.items():
                if fut.done():
                    try:
                        outputs.update(fut.result())
                    except Exception as e:
                        self._cancel_all(remaining.values())
                        raise ExpertRpcFailure(
                            f"expert RPC to peer {peer!r} failed for layer "
                            f"{layer_idx}: {e}"
                        ) from e
                    done_peers.append(peer)
            for peer in done_peers:
                del remaining[peer]
            if not remaining:
                break

            if deadline is not None and time.monotonic() >= deadline:
                # Any peer that didn't finish is responsible for the
                # timeout; surface the first as the failure peer.
                stuck_peer = next(iter(remaining))
                self._cancel_all(remaining.values())
                raise ExpertRpcFailure(
                    f"expert RPC to peer {stuck_peer!r} failed for layer "
                    f"{layer_idx}: timeout after {self.rpc_timeout_s}s"
                )

            # Wait up to ``poll_s`` on ANY abort event rather than sleeping
            # blindly: this keeps the loop responsive without spinning.
            # ``Event.wait`` cannot wait on multiple events at once, so we
            # poll each; the common case (no abort) returns immediately
            # after ``poll_s``. When an abort does fire we catch it on the
            # very next loop iteration.
            for ev in abort_events.values():
                if ev.wait(timeout=poll_s / max(1, len(abort_events))):
                    break

    @staticmethod
    def _cancel_all(futs: Any) -> None:
        """Best-effort cancel of any futures still in the queue. Futures that
        are already running can't be cancelled; the handlers on those peers
        will eventually time out and return (or drop on TCP close)."""
        for fut in futs:
            with contextlib.suppress(Exception):
                fut.cancel()

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
                lm, h, layer_idx, cache, masks,
                heat_observer=self.heat_observer,
            )
            mx.eval(top_k_ids)
            # Union of all top-k ids across the batch and sequence.
            all_ids = sorted(
                {int(e) for e in top_k_ids.reshape(-1).tolist()}  # type: ignore[arg-type,union-attr]
            )
            peer_loads = self.loads_provider()
            self_load = peer_loads.get(self.self_shard_id, 0)
            by_owner = group_expert_ids_by_owner_loaded(
                all_ids,
                owners=self.owners,
                peer_loads=peer_loads,
                self_shard_id=self.self_shard_id,
                self_load=self_load,
                rng=self.rng,
                live_owners_provider=self.live_owners_provider,
            )

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
        futures: dict[str, Future[dict[int, mx.array]]] = {
            peer: self._executor.submit(
                self.peer_rpc.call, peer, request_id, layer_idx, ids, post_attn
            )
            for peer, ids in by_owner.items()
        }
        # Register per-peer abort events under this request_id so the
        # membership observer (via notify_peer_left_alive) can signal us to
        # stop waiting. We keep the registration even with no peers because
        # notify_peer_left_alive iterates defensively.
        abort_events: dict[str, threading.Event] = {
            peer: threading.Event() for peer in futures
        }
        if abort_events:
            with self._in_flight_lock:
                self._in_flight[request_id] = abort_events
        try:
            self._gather_with_abort(
                futures, abort_events, outputs, layer_idx
            )
        finally:
            if abort_events:
                with self._in_flight_lock:
                    self._in_flight.pop(request_id, None)

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


__all__ = ["ExpertOrchestrator", "ExpertRpcFailure", "PeerRPC", "TcpPeerRPC"]
