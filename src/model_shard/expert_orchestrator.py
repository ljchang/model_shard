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

try:
    import mlx.core as mx
except ImportError:
    mx = None  # type: ignore[assignment]

from model_shard._pb import wire_pb2
from model_shard.backends import Backend, PyTorchBackend
from model_shard.envelope import recv_envelope, send_envelope
from model_shard.mlx_engine import _mx_to_wire_dtype, bytes_to_tensor, tensor_to_bytes
from model_shard.moe import group_expert_ids_by_owner_loaded
from model_shard.request import OpDescriptor, OpType, ProvenanceEntry


class ExpertRpcFailure(RuntimeError):  # noqa: N818 — explicit name per plan
    """Raised by ExpertOrchestrator when a peer RPC fails (timeout, broken
    pipe, observer-triggered close). The node's request handler translates
    this into Error{SHARD_UNAVAILABLE, is_final=true} for the client.

    Phase 6-A: gains typed ``failed_peer`` and ``layer_idx`` fields so the
    retry loop in ``run_split_layer`` can exclude the known-failed peer
    from subsequent dispatches."""

    def __init__(self, message: str, *, failed_peer: str, layer_idx: int) -> None:
        super().__init__(message)
        self.failed_peer = failed_peer
        self.layer_idx = layer_idx


class PeerRPC(Protocol):
    def call(
        self,
        peer_shard_id: str,
        request_id: str,
        layer_idx: int,
        expert_ids: list[int],
        h: mx.array,
        provenance_pb_out: list[Any] | None = None,
        provenance_pb_in: list[Any] | None = None,
    ) -> dict[int, mx.array]:
        """Send an ExpertRequest to ``peer_shard_id``, block for the
        ExpertResponse, and return ``{expert_id: output tensor}``. Must
        raise on timeout or RPC error.

        ``provenance_pb_in`` is attached to the outbound ExpertRequest.
        ``provenance_pb_out`` is a mutable list; the implementation appends
        any provenance entries from the ExpertResponse into it."""
        ...


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
        provenance_pb_out: list[Any] | None = None,
        provenance_pb_in: list[Any] | None = None,
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
            req.expert_request.h_spec.dtype = _mx_to_wire_dtype(h.dtype)
            req.expert_request.h_spec.quant = wire_pb2.QUANT_NONE
            req.expert_request.h_spec.byte_count = len(raw)
            if provenance_pb_in:
                req.expert_request.provenance.extend(provenance_pb_in)
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
            if provenance_pb_out is not None:
                provenance_pb_out.extend(resp.provenance)
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


@dataclass(kw_only=True)
class ExpertOrchestrator:
    """Runs one decoder layer with experts partitioned across shards.

    Not frozen: holds a ``ThreadPoolExecutor`` for parallel peer fan-out.
    The executor lifetime matches the orchestrator instance. Create one
    orchestrator per node for the lifetime of the decode loop, and call
    ``close()`` on shutdown to release the fan-out executor threads.

    Phase 7-B: ``backend`` is required. The orchestrator always dispatches
    compute via ``Backend`` primitives; the legacy ``moe.run_*`` fallback
    has been removed.
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
    backend: Backend
    retry_max_attempts: int = 3
    retry_backoff_ms: tuple[int, ...] = (100, 500)
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
                        f"{layer_idx}",
                        failed_peer=peer,
                        layer_idx=layer_idx,
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
                            f"{layer_idx}: {e}",
                            failed_peer=peer,
                            layer_idx=layer_idx,
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
                    f"{layer_idx}: timeout after {self.rpc_timeout_s}s",
                    failed_peer=stuck_peer,
                    layer_idx=layer_idx,
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

    def _phase_b_with_retry(
        self,
        post_attn: mx.array,
        all_ids: list[int],
        layer_idx: int,
        request_id: str,
        initial_local_ids: list[int],
        provenance_chain: list[ProvenanceEntry] | None = None,
        ar_hash: bytes | None = None,
    ) -> dict[int, mx.array]:
        """Run the peer fan-out with retries on ``ExpertRpcFailure``.

        Preserves partial outputs across retries: experts that already
        completed (in ``outputs``) are never re-dispatched. Each retry
        excludes peers that previously failed in THIS invocation.

        ``provenance_chain`` and ``ar_hash`` are threaded through for Phase 6-B
        provenance recording. When ``provenance_chain is None``, all provenance
        code is inert.
        """
        import time as _time

        # ids we still need outputs for (local ids handled by caller).
        remote_ids_needed = [e for e in all_ids if e not in initial_local_ids]

        # Build pb_prefix once from the current chain (snapshot before fan-out).
        pb_prefix: list[Any] = []
        if provenance_chain is not None:
            from model_shard.provenance import entry_to_pb
            pb_prefix = [entry_to_pb(e) for e in provenance_chain]

        # Initial routing.
        peer_loads = self.loads_provider()
        self_load = peer_loads.get(self.self_shard_id, 0)
        by_owner = group_expert_ids_by_owner_loaded(
            remote_ids_needed,
            owners=self.owners,
            peer_loads=peer_loads,
            self_shard_id=self.self_shard_id,
            self_load=self_load,
            rng=self.rng,
            live_owners_provider=self.live_owners_provider,
        )
        local_ids_extra = by_owner.pop(self.self_shard_id, [])
        outputs: dict[int, mx.array] = {}
        if local_ids_extra:
            with self._mlx_guard():
                outputs.update(
                    self.backend.run_selected_experts(
                        layer_idx, post_attn, local_ids_extra,
                    )
                )
            if provenance_chain is not None and ar_hash is not None:
                from model_shard.provenance import build_entry
                for eid in local_ids_extra:
                    provenance_chain.append(
                        build_entry(
                            node_id=self.self_shard_id,
                            op=OpDescriptor(
                                op_type=OpType.OP_EXPERT,
                                layer_idx=layer_idx,
                                expert_id=eid,
                            ),
                            output_tensor=outputs[eid],
                            parent_hashes=(ar_hash,),
                        )
                    )

        # per-peer mutable lists to collect response provenance entries.
        per_peer_response_pb: dict[str, list[Any]] = {peer: [] for peer in by_owner}

        futures: dict[str, Future[dict[int, mx.array]]] = {
            peer: self._executor.submit(
                self.peer_rpc.call,
                peer, request_id, layer_idx, ids, post_attn,
                per_peer_response_pb[peer],  # provenance_pb_out
                pb_prefix if pb_prefix else None,  # provenance_pb_in
            )
            for peer, ids in by_owner.items()
        }
        abort_events: dict[str, threading.Event] = {
            peer: threading.Event() for peer in futures
        }
        # Track whether we EVER registered in _in_flight, not just the current
        # attempt's state — so the finally block's cleanup is unconditional.
        registered = False
        if abort_events:
            with self._in_flight_lock:
                self._in_flight[request_id] = abort_events
                registered = True

        excluded_peers: set[str] = set()
        attempts = 0
        try:
            while True:
                attempts += 1
                try:
                    self._gather_with_abort(
                        futures, abort_events, outputs, layer_idx
                    )
                    # Merge per-peer response provenance into main chain.
                    if provenance_chain is not None:
                        from model_shard.provenance import entry_from_pb
                        for _peer, pbs in per_peer_response_pb.items():
                            for pb in pbs:
                                provenance_chain.append(entry_from_pb(pb))
                    break
                except ExpertRpcFailure as exc:
                    if attempts >= self.retry_max_attempts:
                        raise
                    excluded_peers.add(exc.failed_peer)
                    backoff_idx = min(
                        attempts - 1, max(0, len(self.retry_backoff_ms) - 1)
                    )
                    if self.retry_backoff_ms:
                        _time.sleep(self.retry_backoff_ms[backoff_idx] / 1000.0)

                    # Drain any survivor futures that already completed so
                    # their experts are counted as done before we compute
                    # what's missing (avoids re-dispatching them).
                    for peer, fut in futures.items():
                        if peer != exc.failed_peer and fut.done():
                            with contextlib.suppress(Exception):
                                outputs.update(fut.result())
                    # Also merge response provenance from survivors.
                    if provenance_chain is not None:
                        from model_shard.provenance import entry_from_pb
                        for _peer, pbs in per_peer_response_pb.items():
                            if _peer != exc.failed_peer:
                                for pb in pbs:
                                    provenance_chain.append(entry_from_pb(pb))

                    missing = [e for e in remote_ids_needed if e not in outputs]
                    if not missing:
                        break

                    def _filtered_provider(eid: int) -> set[str]:
                        base = (
                            self.live_owners_provider(eid)
                            if self.live_owners_provider is not None
                            else set()
                        )
                        return base - excluded_peers

                    filtered_owners = {
                        sid: ids for sid, ids in self.owners.items()
                        if sid not in excluded_peers
                    }
                    try:
                        by_owner_retry = group_expert_ids_by_owner_loaded(
                            missing,
                            owners=filtered_owners,
                            peer_loads=self.loads_provider(),
                            self_shard_id=self.self_shard_id,
                            self_load=self.loads_provider().get(
                                self.self_shard_id, 0
                            ),
                            rng=self.rng,
                            live_owners_provider=_filtered_provider,
                        )
                    except KeyError:
                        # All remaining owners for some expert have been
                        # excluded — re-raise the original failure.
                        raise exc from None
                    local_retry = by_owner_retry.pop(self.self_shard_id, [])
                    if local_retry:
                        with self._mlx_guard():
                            outputs.update(
                                self.backend.run_selected_experts(
                                    layer_idx, post_attn, local_retry,
                                )
                            )
                        if provenance_chain is not None and ar_hash is not None:
                            from model_shard.provenance import build_entry
                            for eid in local_retry:
                                provenance_chain.append(
                                    build_entry(
                                        node_id=self.self_shard_id,
                                        op=OpDescriptor(
                                            op_type=OpType.OP_EXPERT,
                                            layer_idx=layer_idx,
                                            expert_id=eid,
                                        ),
                                        output_tensor=outputs[eid],
                                        parent_hashes=(ar_hash,),
                                    )
                                )
                    per_peer_response_pb = {peer: [] for peer in by_owner_retry}
                    futures = {
                        peer: self._executor.submit(
                            self.peer_rpc.call,
                            peer, request_id, layer_idx, ids, post_attn,
                            per_peer_response_pb[peer],  # provenance_pb_out
                            pb_prefix if pb_prefix else None,  # provenance_pb_in
                        )
                        for peer, ids in by_owner_retry.items()
                    }
                    abort_events = {
                        peer: threading.Event() for peer in futures
                    }
                    if abort_events:
                        with self._in_flight_lock:
                            self._in_flight[request_id] = abort_events
                            registered = True
        finally:
            if registered:
                with self._in_flight_lock:
                    self._in_flight.pop(request_id, None)

        return outputs

    def run_split_layer(
        self,
        h: mx.array,
        layer_idx: int,
        cache: list[Any],
        masks: tuple[Any, Any],
        request_id: str,
        provenance_chain: list[ProvenanceEntry] | None = None,
    ) -> mx.array:
        # Phase A — local MLX graph construction for attention, routing, the
        # shared expert, and any experts this shard owns. Guarded by
        # ``mlx_lock`` so peer-handler threads (running on the same Python
        # process in the in-process test fixture) cannot race the default MLX
        # stream with concurrent graph construction.
        ar_hash: bytes | None = None
        with self._mlx_guard():
            post_attn, top_k = self.backend.run_attention_and_route(
                layer_idx, h, cache, masks,
                heat_observer=self.heat_observer,
            )
            top_k_ids, top_k_weights = top_k
            mx.eval(top_k_ids)
            # Union of all top-k ids across the batch and sequence.
            all_ids = sorted(
                {int(e) for e in top_k_ids.reshape(-1).tolist()}
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
            shared_out = self.backend.run_shared_expert(layer_idx, post_attn)
            local_outputs = self.backend.run_selected_experts(
                layer_idx, post_attn, local_ids,
            )
            # Force the local compute graph to realize before releasing the
            # lock; otherwise the peer handlers could start evaluating on the
            # default stream while our local graph is still being built.
            mx.eval(post_attn, shared_out, *local_outputs.values())

            if provenance_chain is not None:
                from model_shard.provenance import build_entry
                prev_hash = provenance_chain[-1].hash if provenance_chain else b""
                parent_hashes_ar: tuple[bytes, ...] = (prev_hash,) if prev_hash else ()
                ar_entry = build_entry(
                    node_id=self.self_shard_id,
                    op=OpDescriptor(op_type=OpType.OP_ATTENTION_ROUTE, layer_idx=layer_idx),
                    output_tensor=post_attn,
                    parent_hashes=parent_hashes_ar,
                )
                provenance_chain.append(ar_entry)
                shared_entry = build_entry(
                    node_id=self.self_shard_id,
                    op=OpDescriptor(op_type=OpType.OP_SHARED_EXPERT, layer_idx=layer_idx),
                    output_tensor=shared_out,
                    parent_hashes=(ar_entry.hash,),
                )
                provenance_chain.append(shared_entry)
                ar_hash = ar_entry.hash

        # Build OP_EXPERT entries for local_outputs (experts initially routed to self).
        if provenance_chain is not None and ar_hash is not None:
            from model_shard.provenance import build_entry
            for eid, output in local_outputs.items():
                provenance_chain.append(
                    build_entry(
                        node_id=self.self_shard_id,
                        op=OpDescriptor(
                            op_type=OpType.OP_EXPERT,
                            layer_idx=layer_idx,
                            expert_id=eid,
                        ),
                        output_tensor=output,
                        parent_hashes=(ar_hash,),
                    )
                )

        # Phase B — peer fan-out with retry on peer failure.
        outputs: dict[int, mx.array] = dict(local_outputs)
        remote_outputs = self._phase_b_with_retry(
            post_attn=post_attn,
            all_ids=all_ids,
            layer_idx=layer_idx,
            request_id=request_id,
            initial_local_ids=local_ids,
            provenance_chain=provenance_chain,
            ar_hash=ar_hash,
        )
        outputs.update(remote_outputs)

        # Phase C — aggregation + outer ops, both via Backend.
        # See Backend.aggregate_experts for the per-position routing logic
        # and Backend.apply_outer_decoder_ops for the outer
        # post_feedforward_layernorm + residual + layer_scalar chain.
        # HF reference: docs/superpowers/reference/2026-04-19-hf-gemma4-forward-signatures.md
        is_pt = isinstance(self.backend, PyTorchBackend)
        with self._mlx_guard():
            h1_plus_h2 = self.backend.aggregate_experts(
                layer_idx, outputs, top_k_ids, top_k_weights, shared_out,
            )
            out: mx.array = self.backend.apply_outer_decoder_ops(
                layer_idx, h1_plus_h2, post_attn,
            )
            if not is_pt:
                mx.eval(out)

        # Phase C provenance: emit OP_AGGREGATE entry after aggregation.
        if provenance_chain is not None:
            split_entries = [
                e for e in provenance_chain
                if e.op is not None
                and e.op.layer_idx == layer_idx
                and e.op.op_type in (OpType.OP_SHARED_EXPERT, OpType.OP_EXPERT)
            ]
            parent_hashes_agg = tuple(e.hash for e in split_entries)
            from model_shard.provenance import build_entry
            agg_entry = build_entry(
                node_id=self.self_shard_id,
                op=OpDescriptor(op_type=OpType.OP_AGGREGATE, layer_idx=layer_idx),
                output_tensor=out,
                parent_hashes=parent_hashes_agg,
            )
            provenance_chain.append(agg_entry)

        return out


__all__ = ["ExpertOrchestrator", "ExpertRpcFailure", "PeerRPC", "TcpPeerRPC"]
