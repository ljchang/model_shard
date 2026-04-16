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
import socket
import threading
from dataclasses import dataclass, field
from typing import Any, BinaryIO, cast

import mlx.core as mx

from model_shard._pb import wire_pb2
from model_shard.envelope import recv_envelope, send_envelope
from model_shard.membership import MembershipRunner, PeerSpec, SwimConfig
from model_shard.membership.records import StateTransition
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
from model_shard.shard_map import ShardMap, ShardSpec

_LOG = logging.getLogger(__name__)
_PROTOCOL_VERSION = 1


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
        loaded_model: LoadedModel,
        total_layers: int,
    ) -> None:
        self._shard = shard
        self._shard_map = shard_map
        self._lm = loaded_model
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
            self._handle_activation(env.activation, tensor_bytes)
        elif which == "sampled_token":
            self._handle_sampled_token(env.sampled_token)
        elif which == "end":
            self._handle_end(env.end)
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
        h = self._run_my_layers(h, cache)
        self._forward_activation(req.request_id, h)

        # Decode loop. Blocks on the queue until the tail returns tokens.
        self._drive_decode_loop(req.request_id, state)

    def _drive_decode_loop(
        self, request_id: str, state: _HeadRequestState
    ) -> None:
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
            h = self._run_my_layers(h, cache)
            self._forward_activation(request_id, h)

        # Clean up everywhere.
        self._broadcast_end(request_id)

    def _handle_activation(
        self, act: wire_pb2.Activation, tensor_bytes: bytes
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
        h = self._run_my_layers(h, cache)

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

    # ------------------------------------------------------------ forwarding

    def _run_my_layers(self, h: mx.array, cache: list[Any]) -> mx.array:
        global_mask, sliding_mask = make_masks(self._lm, h, cache)
        return run_layers(
            self._lm,
            h,
            self._shard.start_layer,
            self._shard.end_layer,
            cache,
            global_mask,
            sliding_mask,
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
        # Wired in Task 25 to drop/redial TCP peer connections.
        pass


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


__all__ = ["Node"]
