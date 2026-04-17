"""Decentralized node — hosts one shard, forwards activations peer-to-peer.

Topology (Phase 1 static pipeline):

    Client ──BeginRequest──▶ Node(head)
                              │
                              ▼ Activation (persistent outbound peer connection)
                            Node(mid)
                              │
                              ▼ Activation
                            Node(tail) ── samples next token from logits
                              │
                              ▼ SampledToken (dials head)
                            Node(head) ── forwards to client on inbound conn
                              │
                              ▼ Activation (next decode round)
                            ...

Each node dials exactly one downstream peer:
  * non-tail → the shard whose start_layer equals self.end_layer
  * tail     → the shard whose start_layer == 0 (the head)

The head's client-connection handler thread *drives* the decode loop.
Incoming SampledToken messages from the tail are routed to that thread via
a queue. This avoids cross-thread writes on the client socket.
"""

from __future__ import annotations

import contextlib
import logging
import os
import queue
import random as _random_mod
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any, BinaryIO, cast

import mlx.core as mx

from model_shard._pb import wire_pb2
from model_shard.envelope import recv_envelope, send_envelope
from model_shard.expert_orchestrator import (
    ExpertOrchestrator,
    ExpertRpcFailure,
    TcpPeerRPC,
)
from model_shard.load import LoadTracker
from model_shard.membership import MembershipRunner, PeerSpec, SwimConfig
from model_shard.membership.records import LoadReportRecord, StateTransition
from model_shard.mlx_engine import (
    LoadedModel,
    bytes_to_tensor,
    embed_tokens,
    finalize,
    make_cache,
    make_masks,
    run_layers,
    tensor_to_bytes,
)
from model_shard.partial_load import slice_expert
from model_shard.shard_map import ShardMap, ShardSpec

_LOG = logging.getLogger(__name__)
_PROTOCOL_VERSION = 1

# Process-wide MLX serialization lock. In production each node is its own
# process, so this lock never contends. In the in-process test fixture we run
# three nodes in a single Python process — concurrent MLX evaluations from
# different threads on the shared LoadedModel can abort the Metal backend, so
# we serialize the expert-RPC compute path (which is the only place multiple
# node threads run MLX at the same time under Phase 3 expert splitting).
_MLX_COMPUTE_LOCK = threading.Lock()


@dataclass
class _HeadRequestState:
    client_stream: BinaryIO
    max_new_tokens: int
    generated: int = 0
    token_queue: queue.Queue[int] = field(default_factory=queue.Queue)


class Node:
    def __init__(
        self,
        shard: ShardSpec,
        shard_map: ShardMap,
        loaded_model: LoadedModel | None = None,
        total_layers: int = 0,
    ) -> None:
        self._shard = shard
        self._shard_map = shard_map
        # Phase 5b: runtime expert ownership registry (see spec D9).
        # Seeded from the frozen ShardSpec at boot; mutated by migration attach.
        self._live_experts: dict[int, set[int]] = {
            layer: set(ids) for layer, ids in shard.moe_experts.items()
        }
        # Union of bootstrap moe_experts across ALL shards + received
        # OwnershipDelta ADDs (see spec D10).
        self._ownership_seen: set[tuple[str, int, int]] = set()
        for sid in shard_map.all_shards():
            peer_spec = shard_map.lookup(sid)
            for layer, ids in peer_spec.moe_experts.items():
                for eid in ids:
                    self._ownership_seen.add((sid, layer, eid))
        self._ownership_seen_lock = threading.Lock()
        # Phase 5a: when ENABLE_PARTIAL_LOAD is set AND the caller did not
        # pass a pre-loaded model AND this shard actually hosts routed
        # experts, build the model via ``load_model_partial`` so only the
        # held expert slices are materialized. Otherwise preserve the prior
        # contract (caller-supplied model).
        if loaded_model is None and _partial_load_enabled() and shard.moe_experts:
            from model_shard.mlx_engine import load_model_partial
            held = {k: list(v) for k, v in shard.moe_experts.items()}
            self._lm: LoadedModel = load_model_partial(
                "mlx-community/gemma-4-26b-a4b-it-4bit",
                held,
            )
        else:
            # Pre-Phase-5a contract: caller is responsible for supplying a
            # fully-loaded model. Type is narrowed to ``LoadedModel`` — if a
            # caller passes ``None`` without flipping the partial-load gate,
            # subsequent attribute access on ``self._lm`` will fail loudly,
            # which matches prior behavior.
            self._lm = cast(LoadedModel, loaded_model)
        self._total_layers = total_layers
        self._downstream: ShardSpec = _resolve_downstream(shard, shard_map, total_layers)

        self._state_lock = threading.Lock()
        self._kv_caches: dict[str, list[Any]] = {}
        self._head_states: dict[str, _HeadRequestState] = {}

        self._out_lock = threading.Lock()
        self._out_sock: socket.socket | None = None
        self._out_stream: BinaryIO | None = None

        # Per-request debug captures: (next_layer_idx, hidden) recorded when
        # this node forwards an activation. In-process test hook only.
        self._debug_captures: dict[str, list[tuple[int, mx.array]]] = {}

        self._stopping = threading.Event()
        self._server_sock: socket.socket | None = None

        self._membership: MembershipRunner | None = None
        if _gossip_enabled():
            self._membership = self._build_membership_runner()

        # Phase 4 load tracking. The tracker is always constructed (even when
        # gossip is disabled) so ``_handle_expert_request`` has somewhere to
        # post queue-depth samples. The runner-side ``start_load_source``
        # registration is only useful when gossip is active.
        self._load_tracker = LoadTracker(
            alpha=0.3, jitter_pct=0.1, rng=_random_mod.Random()
        )
        self._in_flight_expert_requests: int = 0

        if self._membership is not None:
            def _load_source() -> LoadReportRecord:
                return LoadReportRecord(
                    shard_id=self._shard.shard_id,
                    queue_depth_ema=self._load_tracker.report(),
                    ts_unix_ms=int(time.time() * 1000),
                )
            self._membership.start_load_source(_load_source)

        # Phase 3 expert sharding. Construct an ``ExpertOrchestrator`` only on
        # nodes whose layer range (i.e., whose attention) covers a split layer
        # in ``self._shard.moe_experts``. Other nodes merely serve inbound
        # ``ExpertRequest`` RPCs (see ``_handle_expert_request``); they do not
        # need an orchestrator because they never run the attention for the
        # split layer, only its experts on demand.
        self._split_layers: set[int] = set()
        self._orchestrator: ExpertOrchestrator | None = None
        if _expert_shard_enabled() and self._shard.moe_experts:
            my_split = {
                layer_idx
                for layer_idx in self._shard.moe_experts
                if self._shard.start_layer <= layer_idx < self._shard.end_layer
            }
            if my_split:
                self._split_layers = my_split
                self._orchestrator = self._build_expert_orchestrator()

    # ------------------------------------------------------------------ roles

    @property
    def is_head(self) -> bool:
        return self._shard.start_layer == 0

    @property
    def is_tail(self) -> bool:
        return self._shard.end_layer == self._total_layers

    # ---------------------------------------------------------------- server

    def serve_forever(self) -> None:
        if self._membership is not None:
            self._membership.subscribe(self._on_membership_change)
            self._membership.start()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self._shard.address.host, self._shard.address.port))
        sock.listen(8)
        sock.settimeout(0.25)
        self._server_sock = sock
        try:
            while not self._stopping.is_set():
                try:
                    conn, _ = sock.accept()
                except TimeoutError:
                    continue
                t = threading.Thread(
                    target=self._handle_connection, args=(conn,), daemon=True
                )
                t.start()
        finally:
            sock.close()
            self._server_sock = None
            self._close_outbound()

    def shutdown(self) -> None:
        self._stopping.set()
        if self._membership is not None:
            self._membership.stop()
        if self._orchestrator is not None:
            self._orchestrator.close()

    def _handle_connection(self, conn: socket.socket) -> None:
        try:
            with cast(BinaryIO, conn.makefile("rwb", buffering=0)) as stream:
                while not self._stopping.is_set():
                    try:
                        env, tensor_bytes = recv_envelope(stream)
                    except EOFError:
                        return
                    try:
                        self._dispatch(env, tensor_bytes, stream)
                    except Exception:
                        _LOG.exception("dispatch error")
        finally:
            conn.close()

    # -------------------------------------------------------------- dispatch

    def _dispatch(
        self,
        env: wire_pb2.Envelope,
        tensor_bytes: bytes,
        inbound_stream: BinaryIO,
    ) -> None:
        which = env.WhichOneof("payload")
        if which == "begin":
            self._handle_begin(env.begin, inbound_stream)
        elif which == "activation":
            self._handle_activation(env.activation, tensor_bytes, inbound_stream)
        elif which == "sampled_token":
            self._handle_sampled_token(env.sampled_token)
        elif which == "end":
            self._handle_end(env.end)
        elif which == "expert_request":
            self._handle_expert_request(
                env.expert_request, tensor_bytes, inbound_stream
            )
        elif which == "expert_weight_request":
            self._handle_expert_weight_request(
                env.expert_weight_request, inbound_stream
            )
        else:
            _LOG.warning("unknown envelope payload %r", which)

    def _handle_begin(
        self, req: wire_pb2.BeginRequest, client_stream: BinaryIO
    ) -> None:
        """Only the head should receive BeginRequest (from a Client)."""
        if not self.is_head:
            _send_error(
                client_stream,
                req.request_id,
                wire_pb2.ERR_WRONG_SHARD,
                "BeginRequest to non-head shard",
            )
            return

        unavailable = self._unavailable_peer()
        if unavailable is not None:
            _send_error(
                client_stream,
                req.request_id,
                wire_pb2.ERR_SHARD_UNAVAILABLE,
                f"shard {unavailable!r} not alive",
            )
            return

        cache = make_cache(self._lm)
        state = _HeadRequestState(
            client_stream=client_stream,
            max_new_tokens=int(req.max_new_tokens) or 1,
        )
        with self._state_lock:
            self._kv_caches[req.request_id] = cache
            self._head_states[req.request_id] = state

        # Prefill on this shard's layer range.
        prompt_tokens = list(req.prompt_token_ids)
        token_ids = mx.array([prompt_tokens])
        h = embed_tokens(self._lm, token_ids)
        try:
            h = self._run_my_layers(h, cache, request_id=req.request_id)
        except ExpertRpcFailure as exc:
            _LOG.warning("prefill aborted by expert RPC failure: %s", exc)
            with contextlib.suppress(OSError):
                _send_error(
                    client_stream,
                    req.request_id,
                    wire_pb2.ERR_SHARD_UNAVAILABLE,
                    str(exc),
                )
            with self._state_lock:
                self._kv_caches.pop(req.request_id, None)
                self._head_states.pop(req.request_id, None)
            return
        self._forward_activation(req.request_id, h)

        # Decode loop. Blocks on the queue until the tail returns tokens.
        self._drive_decode_loop(req.request_id, state)

    def _drive_decode_loop(
        self, request_id: str, state: _HeadRequestState
    ) -> None:
        try:
            while state.generated < state.max_new_tokens:
                token_id = state.token_queue.get()
                state.generated += 1
                is_final = state.generated >= state.max_new_tokens

                _send_sampled_token_to(
                    state.client_stream,
                    request_id,
                    token_id,
                    position=state.generated - 1,
                    is_final=is_final,
                )

                if is_final:
                    break

                # Next decode round: embed this token and forward an activation.
                with self._state_lock:
                    cache = self._kv_caches[request_id]
                h = embed_tokens(self._lm, mx.array([[token_id]]))
                h = self._run_my_layers(h, cache, request_id=request_id)
                self._forward_activation(request_id, h)

            # Clean up everywhere.
            self._broadcast_end(request_id)
        except OSError as exc:
            _LOG.warning("decode loop aborted by broken pipe: %s", exc)
            with contextlib.suppress(OSError):
                _send_error(
                    state.client_stream,
                    request_id,
                    wire_pb2.ERR_SHARD_UNAVAILABLE,
                    f"downstream peer unavailable: {exc}",
                )
            with self._state_lock:
                self._kv_caches.pop(request_id, None)
                self._head_states.pop(request_id, None)
        except ExpertRpcFailure as exc:
            _LOG.warning("decode loop aborted by expert RPC failure: %s", exc)
            with contextlib.suppress(OSError):
                _send_error(
                    state.client_stream,
                    request_id,
                    wire_pb2.ERR_SHARD_UNAVAILABLE,
                    str(exc),
                )
            with self._state_lock:
                self._kv_caches.pop(request_id, None)
                self._head_states.pop(request_id, None)

    def _handle_activation(
        self,
        act: wire_pb2.Activation,
        tensor_bytes: bytes,
        inbound_stream: BinaryIO,
    ) -> None:
        if int(act.next_layer_idx) != self._shard.start_layer:
            _LOG.error(
                "activation for layer %d arrived at shard starting at %d",
                int(act.next_layer_idx),
                self._shard.start_layer,
            )
            return

        with self._state_lock:
            cache = self._kv_caches.setdefault(act.request_id, make_cache(self._lm))

        h = bytes_to_tensor(
            tensor_bytes, shape=list(act.tensor.shape), dtype=act.tensor.dtype
        )
        try:
            h = self._run_my_layers(h, cache, request_id=act.request_id)
        except ExpertRpcFailure as exc:
            # Reuse the Phase 2 broken-pipe pattern: send an Error envelope
            # back upstream on the inbound connection and close both the
            # inbound and outbound sockets so the upstream peer's next write
            # fails with a broken pipe (which its own decode loop then
            # converts to Error{SHARD_UNAVAILABLE} for the client).
            _LOG.warning(
                "activation aborted by expert RPC failure on %s: %s",
                self._shard.shard_id,
                exc,
            )
            with contextlib.suppress(OSError):
                _send_error(
                    inbound_stream,
                    act.request_id,
                    wire_pb2.ERR_SHARD_UNAVAILABLE,
                    str(exc),
                )
            with contextlib.suppress(OSError):
                inbound_stream.close()
            self._close_outbound()
            with self._state_lock:
                self._kv_caches.pop(act.request_id, None)
            return

        if self.is_tail:
            logits = finalize(self._lm, h)
            token_id = int(mx.argmax(logits[0, -1, :]).item())
            # Position is managed by the head; we leave it 0 here.
            self._send_sampled_token(act.request_id, token_id, position=0)
        else:
            self._forward_activation(act.request_id, h)

    def _handle_sampled_token(self, tok: wire_pb2.SampledToken) -> None:
        """Only the head should receive inbound SampledTokens (from the tail)."""
        if not self.is_head:
            _LOG.warning("unexpected SampledToken on non-head shard")
            return
        with self._state_lock:
            state = self._head_states.get(tok.request_id)
        if state is None:
            _LOG.warning("SampledToken for unknown request_id %s", tok.request_id)
            return
        state.token_queue.put(int(tok.token_id))

    def _handle_end(self, req: wire_pb2.EndRequest) -> None:
        with self._state_lock:
            self._kv_caches.pop(req.request_id, None)
            self._head_states.pop(req.request_id, None)
            # Intentionally do NOT drop _debug_captures here — tests read them
            # after the request completes. Production impact is a tiny memory
            # retention per request; cleared explicitly via clear_debug_captures().

    def _handle_expert_request(
        self,
        req: wire_pb2.ExpertRequest,
        tensor_bytes: bytes,
        inbound_stream: BinaryIO,
    ) -> None:
        """Run this shard's hosted experts for ``req.layer_idx`` and reply
        with an ``ExpertResponse``.

        Fail with ``Error{ERR_WRONG_SHARD}`` if the client asks for any
        expert id this shard does not own — we must not silently run only
        the subset we host, because the caller would stack in-order outputs
        and get wrong aggregation.
        """
        from model_shard.moe import run_selected_experts

        layer_idx = int(req.layer_idx)
        requested = [int(e) for e in req.expert_ids]

        # S3: empty expert_ids is malformed — the caller must ask for at least
        # one expert so we have something to stack and reply with.
        if not requested:
            _send_error(
                inbound_stream,
                req.request_id,
                wire_pb2.ERR_INTERNAL,
                f"ExpertRequest for layer {layer_idx} had empty expert_ids",
            )
            return

        hosted = set(self._shard.moe_experts.get(layer_idx, ()))
        missing = [eid for eid in requested if eid not in hosted]
        if missing:
            _send_error(
                inbound_stream,
                req.request_id,
                wire_pb2.ERR_WRONG_SHARD,
                (
                    f"shard {self._shard.shard_id!r} does not host experts "
                    f"{missing} for layer {layer_idx}"
                ),
            )
            return

        # I1: cross-check the out-of-band tensor length against the declared
        # byte_count before we try to reshape it. A mismatch means the frame
        # is malformed (sender/receiver disagree on the payload size).
        if int(req.h_spec.byte_count) != len(tensor_bytes):
            _send_error(
                inbound_stream,
                req.request_id,
                wire_pb2.ERR_INTERNAL,
                (
                    f"ExpertRequest h_spec.byte_count={int(req.h_spec.byte_count)} "
                    f"does not match tensor payload length {len(tensor_bytes)}"
                ),
            )
            return

        h = bytes_to_tensor(
            tensor_bytes,
            shape=list(req.h_spec.shape),
            dtype=req.h_spec.dtype,
        )

        # I2: any failure inside the expert compute (OOM, unknown layer in
        # run_selected_experts, mlx errors) must be reported as
        # Error{ERR_SHARD_UNAVAILABLE} so the caller fails fast instead of
        # waiting for TCP timeout. Matches the Phase 2 broken-pipe pattern.
        #
        # Acquire the process-wide MLX lock around evaluation so concurrent
        # ExpertRequest handlers (possible when 3 in-process nodes share the
        # default MLX stream in the test fixture) don't race on Metal.
        #
        # Phase 4: bracket the entire handler body with an in-flight counter
        # so ``LoadTracker`` sees the arrival and departure of every expert
        # request on this shard. The outer try/finally guarantees the
        # decrement runs even on the error-path early return below.
        self._in_flight_expert_requests += 1
        self._load_tracker.observe(self._in_flight_expert_requests)
        try:
            try:
                with _MLX_COMPUTE_LOCK:
                    outputs = run_selected_experts(
                        self._lm, h, layer_idx, requested
                    )
                    # Stack in request order so the caller can unstack by index.
                    stacked = mx.stack(
                        [outputs[eid] for eid in requested], axis=2
                    )
                    # Force realization before releasing the lock so the
                    # serialized bytes are based on fully-computed data (and no
                    # dangling graph refs cross threads).
                    mx.eval(stacked)
                    raw = tensor_to_bytes(stacked)
            except Exception as exc:
                _LOG.exception("expert fan-out raised")
                _send_error(
                    inbound_stream,
                    req.request_id,
                    wire_pb2.ERR_SHARD_UNAVAILABLE,
                    f"expert execution failed: {exc}",
                )
                return

            resp = wire_pb2.Envelope()
            resp.expert_response.protocol_version = _PROTOCOL_VERSION
            resp.expert_response.request_id = req.request_id
            resp.expert_response.layer_idx = layer_idx
            resp.expert_response.expert_ids.extend(requested)
            resp.expert_response.outputs_spec.shape.extend(list(stacked.shape))
            resp.expert_response.outputs_spec.dtype = _dtype_to_wire(stacked.dtype)
            resp.expert_response.outputs_spec.quant = wire_pb2.QUANT_NONE
            resp.expert_response.outputs_spec.byte_count = len(raw)
            send_envelope(inbound_stream, resp, raw)
        finally:
            self._in_flight_expert_requests -= 1
            self._load_tracker.observe(self._in_flight_expert_requests)

    def _handle_expert_weight_request(
        self,
        req: wire_pb2.ExpertWeightRequest,
        inbound_stream: BinaryIO,
    ) -> None:
        """Source-side of Phase 5b migration: slice the requested expert
        out of our compact stack and reply with ExpertWeightTransfer.

        Error{ERR_SHARD_UNAVAILABLE} on miss (expert no longer held)."""
        layer_idx = int(req.layer_idx)
        expert_id = int(req.expert_id)
        try:
            tensors = slice_expert(
                self._lm, layer_idx, expert_id, _MLX_COMPUTE_LOCK
            )
        except KeyError as e:
            _send_error(
                inbound_stream,
                req.request_id,
                wire_pb2.ERR_SHARD_UNAVAILABLE,
                str(e),
            )
            return
        resp = wire_pb2.Envelope()
        resp.expert_weight_transfer.protocol_version = _PROTOCOL_VERSION
        resp.expert_weight_transfer.request_id = req.request_id
        resp.expert_weight_transfer.layer_idx = layer_idx
        resp.expert_weight_transfer.expert_id = expert_id
        resp.expert_weight_transfer.tensor_count = 9
        blobs: list[bytes] = []
        for t in tensors:
            d = resp.expert_weight_transfer.tensors.add()
            d.shape.extend(list(t.shape))
            d.dtype = _dtype_to_wire(t.dtype)
            d.quant = wire_pb2.QUANT_NONE
            raw = tensor_to_bytes(t)
            d.byte_count = len(raw)
            blobs.append(raw)
        send_envelope(inbound_stream, resp, b"".join(blobs))

    # ------------------------------------------------------------ forwarding

    def _run_my_layers(
        self, h: mx.array, cache: list[Any], request_id: str = ""
    ) -> mx.array:
        global_mask, sliding_mask = make_masks(self._lm, h, cache)
        return run_layers(
            self._lm,
            h,
            self._shard.start_layer,
            self._shard.end_layer,
            cache,
            global_mask,
            sliding_mask,
            split_layers=self._split_layers,
            orchestrator=self._orchestrator,
            request_id=request_id,
        )

    def _forward_activation(self, request_id: str, h: mx.array) -> None:
        """Send an Activation to the downstream peer (mid→next, or tail should not reach here)."""
        assert not self.is_tail, "tail should call _send_sampled_token instead"
        # Capture for in-process Tier 2 testing.
        self._debug_captures.setdefault(request_id, []).append(
            (self._shard.end_layer, h)
        )
        env, raw = _activation_envelope(request_id, self._shard.end_layer, h)
        self._write_out(env, raw)

    def _send_sampled_token(
        self, request_id: str, token_id: int, position: int
    ) -> None:
        """Tail → head SampledToken."""
        env = wire_pb2.Envelope()
        env.sampled_token.protocol_version = _PROTOCOL_VERSION
        env.sampled_token.request_id = request_id
        env.sampled_token.token_id = token_id
        env.sampled_token.position = position
        env.sampled_token.is_final = False  # head decides finality
        self._write_out(env, b"")

    def _broadcast_end(self, request_id: str) -> None:
        """Send EndRequest downstream so peers drop their caches."""
        env = wire_pb2.Envelope()
        env.end.protocol_version = _PROTOCOL_VERSION
        env.end.request_id = request_id
        # Fire-and-forget down the outbound peer; the peer forwards it in turn.
        # (For Phase 1 we emit once to the immediate peer; each peer receives
        #  EndRequest and, per _handle_end, clears its state. If we wanted
        #  full propagation, each node would need to relay EndRequest further.)
        with contextlib.suppress(OSError):
            self._write_out(env, b"")
        # Also clear local state (head already did this for head_states, but
        # clear kv_cache too). Debug captures are kept until the test clears
        # them explicitly via clear_debug_captures().
        with self._state_lock:
            self._kv_caches.pop(request_id, None)

    # ---------------------------------------------------------- outbound conn

    def _write_out(self, env: wire_pb2.Envelope, tensor_bytes: bytes) -> None:
        with self._out_lock:
            stream = self._ensure_out_stream()
            send_envelope(stream, env, tensor_bytes)

    def _ensure_out_stream(self) -> BinaryIO:
        if self._out_stream is None:
            sock = socket.create_connection(
                (self._downstream.address.host, self._downstream.address.port),
                timeout=30.0,
            )
            self._out_sock = sock
            self._out_stream = cast(BinaryIO, sock.makefile("rwb", buffering=0))
        return self._out_stream

    def _close_outbound(self) -> None:
        with self._out_lock:
            if self._out_stream is not None:
                with contextlib.suppress(OSError):
                    self._out_stream.close()
                self._out_stream = None
            if self._out_sock is not None:
                with contextlib.suppress(OSError):
                    self._out_sock.close()
                self._out_sock = None

    # ---------------------------------------------------------------- testing

    def debug_captures_for(self, request_id: str) -> list[tuple[int, mx.array]]:
        """In-process test hook: returns (next_layer_idx, hidden) pairs recorded
        while this node forwarded activations for `request_id`. Not part of the
        production API."""
        return list(self._debug_captures.get(request_id, []))

    def clear_debug_captures(self) -> None:
        """Test hook: reset all debug captures across requests."""
        with self._state_lock:
            self._debug_captures.clear()

    # ------------------------------------------------------------ membership

    @property
    def membership(self) -> MembershipRunner | None:
        return self._membership

    def self_load_report(self) -> LoadReportRecord:
        """Return a fresh load report for this node. Used by the /loads debug
        endpoint so callers see a complete view including the local shard —
        ``MembershipRunner.latest_loads()`` is peer-only by design (it caches
        reports received over the wire)."""
        return LoadReportRecord(
            shard_id=self._shard.shard_id,
            queue_depth_ema=self._load_tracker.report(),
            ts_unix_ms=int(time.time() * 1000),
        )

    def owners_of(self, layer_idx: int, expert_id: int) -> set[str]:
        """Return the current live owner set for (layer_idx, expert_id).

        Union of bootstrap ShardSpec.moe_experts and gossip-observed ADDs.
        Used by ExpertOrchestrator.live_owners_provider in Phase 5b."""
        with self._ownership_seen_lock:
            return {
                sid for (sid, lyr, eid) in self._ownership_seen
                if lyr == layer_idx and eid == expert_id
            }

    def _build_expert_orchestrator(self) -> ExpertOrchestrator:
        """Construct an ExpertOrchestrator that fans out to peers via TCP.

        ``owners`` is the union of per-layer expert ownership across ALL split
        layers this node attends. In Phase 3 only layer 15 is split so this is
        effectively single-layer, but we build the union generically so future
        multi-split-layer configs do not require another code change. The
        orchestrator dispatches by layer_idx per call, so as long as every
        split layer's owners map is represented the local/remote routing is
        correct.
        """
        owners: dict[str, set[int]] = {}
        for sid in self._shard_map.all_shards():
            spec = self._shard_map.lookup(sid)
            ids: set[int] = set()
            for layer_idx in self._split_layers:
                ids.update(spec.moe_experts.get(layer_idx, ()))
            owners[sid] = ids

        addresses = {
            sid: (
                self._shard_map.lookup(sid).address.host,
                self._shard_map.lookup(sid).address.port,
            )
            for sid in self._shard_map.all_shards()
            if sid != self._shard.shard_id
        }

        def _loads_provider() -> dict[str, int]:
            if self._membership is None:
                return {}
            return {
                sid: lr.queue_depth_ema
                for sid, lr in self._membership.latest_loads().items()
            }

        return ExpertOrchestrator(
            self_shard_id=self._shard.shard_id,
            owners=owners,
            peer_rpc=TcpPeerRPC(addresses=addresses, timeout_s=30.0),
            rpc_timeout_s=30.0,
            mlx_lock=_MLX_COMPUTE_LOCK,
            loads_provider=_loads_provider,
            rng=_random_mod.Random(),
        )

    def _build_membership_runner(self) -> MembershipRunner:
        self_spec = PeerSpec(
            shard_id=self._shard.shard_id,
            host=self._shard.address.host,
            udp_port=self._shard.udp_port,
        )
        peer_specs = [
            PeerSpec(
                shard_id=sid,
                host=self._shard_map.lookup(sid).address.host,
                udp_port=self._shard_map.lookup(sid).udp_port,
            )
            for sid in self._shard_map.all_shards()
            if sid != self._shard.shard_id
        ]
        return MembershipRunner(
            self_spec=self_spec,
            peers=peer_specs,
            config=SwimConfig(),
        )

    def _on_membership_change(self, transition: StateTransition) -> None:
        # Two concerns:
        #   (1) Pipeline continuity — close our outbound TCP iff the
        #       downstream peer (our one forwarding target) left ALIVE.
        #   (2) Expert-RPC abort — tell the orchestrator so any in-flight
        #       ExpertRequest to the transitioned peer short-circuits
        #       instead of blocking on TCP timeout. Applies to ANY peer
        #       because the orchestrator may RPC to any shard that hosts
        #       a split-layer expert.
        new_state = transition.new_record.state
        left_alive = (
            transition.old_state is not None
            and transition.old_state.name == "ALIVE"
            and new_state.name in ("SUSPECT", "DEAD")
        )

        if transition.shard_id == self._downstream.shard_id:
            if new_state.name in ("SUSPECT", "DEAD"):
                _LOG.info(
                    "downstream peer %s -> %s; closing outbound TCP",
                    transition.shard_id,
                    new_state.name,
                )
                self._close_outbound()
            elif new_state.name == "ALIVE" and transition.old_state is not None:
                _LOG.info(
                    "downstream peer %s -> ALIVE; outbound TCP will redial on next send",
                    transition.shard_id,
                )
                # The lazy `_ensure_out_stream` already redials on next write.

        if left_alive and self._orchestrator is not None:
            _LOG.info(
                "peer %s left ALIVE; aborting in-flight expert RPCs",
                transition.shard_id,
            )
            self._orchestrator.notify_peer_left_alive(transition.shard_id)

    def _unavailable_peer(self) -> str | None:
        if self._membership is None:
            return None
        view = self._membership.state.view()
        for sid in self._shard_map.all_shards():
            rec = view.get(sid)
            if rec is None or rec.state.name != "ALIVE":
                return sid
        return None


# -------------------------------------------------------------------- helpers


def _resolve_downstream(
    shard: ShardSpec, shard_map: ShardMap, total_layers: int
) -> ShardSpec:
    """Non-tail → shard starting at self.end_layer. Tail → the head (start=0)."""
    is_tail = shard.end_layer == total_layers
    target_start = 0 if is_tail else shard.end_layer
    for sid in shard_map.all_shards():
        peer = shard_map.lookup(sid)
        if peer.start_layer == target_start and peer.shard_id != shard.shard_id:
            return peer
    raise ValueError(
        f"no downstream peer for shard {shard.shard_id} "
        f"(looking for start_layer={target_start})"
    )


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


def _activation_envelope(
    request_id: str, next_layer: int, h: mx.array
) -> tuple[wire_pb2.Envelope, bytes]:
    raw = tensor_to_bytes(h)
    env = wire_pb2.Envelope()
    env.activation.protocol_version = _PROTOCOL_VERSION
    env.activation.request_id = request_id
    env.activation.next_layer_idx = next_layer
    env.activation.tensor.shape.extend(list(h.shape))
    env.activation.tensor.dtype = _dtype_to_wire(h.dtype)
    env.activation.tensor.quant = wire_pb2.QUANT_NONE
    env.activation.tensor.byte_count = len(raw)
    return env, raw


def _send_sampled_token_to(
    stream: BinaryIO,
    request_id: str,
    token_id: int,
    position: int,
    is_final: bool,
) -> None:
    env = wire_pb2.Envelope()
    env.sampled_token.protocol_version = _PROTOCOL_VERSION
    env.sampled_token.request_id = request_id
    env.sampled_token.token_id = token_id
    env.sampled_token.position = position
    env.sampled_token.is_final = is_final
    send_envelope(stream, env)


def _send_error(
    stream: BinaryIO, request_id: str, code: int, detail: str
) -> None:
    env = wire_pb2.Envelope()
    env.error.protocol_version = _PROTOCOL_VERSION
    env.error.request_id = request_id
    env.error.code = code
    env.error.detail = detail
    send_envelope(stream, env)


def _gossip_enabled() -> bool:
    return os.environ.get("ENABLE_GOSSIP", "true").lower() not in ("0", "false", "no")


def _expert_shard_enabled() -> bool:
    """Phase 3 expert sharding gate. Default OFF so Phase 1/2 tests are
    unaffected; the Phase 3 fixture flips it on before constructing nodes."""
    return os.environ.get("ENABLE_EXPERT_SHARD", "false").lower() in ("1", "true", "yes")


def _partial_load_enabled() -> bool:
    """Phase 5a partial-expert-load gate. Default OFF so Phase 1-4 callers
    that pass a pre-loaded model (or do not populate ``moe_experts``) keep
    behaving exactly as before. When ON, ``Node.__init__`` skips the
    pre-loaded-model path and instead calls ``load_model_partial`` using
    ``shard.moe_experts`` as the held-expert map."""
    return os.environ.get("ENABLE_PARTIAL_LOAD", "false").lower() in ("1", "true", "yes")


__all__ = ["Node"]
