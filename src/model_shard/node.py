"""Node server — hosts a single shard and responds to orchestrator messages.

Wire protocol (see proto/wire.proto):
  BeginRequest    — first shard only: tokenize + embed + run shard's layers.
                    Response: Activation (mid-pipeline) or Logits (if last shard).
  ContinueRequest — first shard only, during decode: embed single token,
                    run layers. Response: Activation or Logits.
  Activation      — mid/tail shard: continue from the incoming hidden state,
                    run shard's layers. Response: Activation or Logits.
  EndRequest      — drop the KV cache for the given request_id.

Phase 1 uses a simple blocking, threaded socket server. One connection per
orchestrator link; orchestrator sends messages sequentially per request.
"""

from __future__ import annotations

import logging
import socket
import threading
from typing import Any, BinaryIO, cast

import mlx.core as mx

from model_shard._pb import wire_pb2
from model_shard.envelope import recv_envelope, send_envelope
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
from model_shard.shard_map import ShardSpec

_LOG = logging.getLogger(__name__)
_PROTOCOL_VERSION = 1


class Node:
    def __init__(
        self, shard: ShardSpec, loaded_model: LoadedModel, total_layers: int
    ) -> None:
        self._shard = shard
        self._lm = loaded_model
        self._total_layers = total_layers
        self._caches: dict[str, list[Any]] = {}
        self._stopping = threading.Event()
        self._server_sock: socket.socket | None = None

    @property
    def is_head(self) -> bool:
        return self._shard.start_layer == 0

    @property
    def is_tail(self) -> bool:
        return self._shard.end_layer == self._total_layers

    def serve_forever(self) -> None:
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
                try:
                    self._handle_connection(conn)
                except Exception:
                    _LOG.exception("connection handler crashed")
                finally:
                    conn.close()
        finally:
            sock.close()
            self._server_sock = None

    def shutdown(self) -> None:
        self._stopping.set()

    def _handle_connection(self, conn: socket.socket) -> None:
        with cast(BinaryIO, conn.makefile("rwb", buffering=0)) as stream:
            while not self._stopping.is_set():
                try:
                    env, tensor_bytes = recv_envelope(stream)
                except EOFError:
                    return
                response, response_tensor = self._dispatch(env, tensor_bytes)
                send_envelope(stream, response, response_tensor)

    # ------------------------------------------------------------------ dispatch

    def _dispatch(
        self, env: wire_pb2.Envelope, tensor_bytes: bytes
    ) -> tuple[wire_pb2.Envelope, bytes]:
        which = env.WhichOneof("payload")
        try:
            if which == "begin":
                return self._handle_begin(env.begin)
            if which == "cont":
                return self._handle_continue(env.cont)
            if which == "activation":
                return self._handle_activation(env.activation, tensor_bytes)
            if which == "end":
                return self._handle_end(env.end)
            return _error(env, wire_pb2.ERR_UNSPECIFIED, f"unknown payload {which!r}")
        except Exception as e:
            _LOG.exception("dispatch error")
            request_id = _extract_request_id(env) or ""
            return _error_from_id(request_id, wire_pb2.ERR_INTERNAL, repr(e))

    def _handle_begin(self, req: wire_pb2.BeginRequest) -> tuple[wire_pb2.Envelope, bytes]:
        if not self.is_head:
            return _error_from_id(
                req.request_id,
                wire_pb2.ERR_WRONG_SHARD,
                f"shard {self._shard.shard_id} is not the head shard",
            )
        cache = make_cache(self._lm)
        self._caches[req.request_id] = cache

        token_ids = mx.array([list(req.prompt_token_ids)])
        h = embed_tokens(self._lm, token_ids)
        h = self._run_my_layers(h, cache)
        return self._emit_hidden_or_logits(req.request_id, h)

    def _handle_continue(
        self, req: wire_pb2.ContinueRequest
    ) -> tuple[wire_pb2.Envelope, bytes]:
        if not self.is_head:
            return _error_from_id(
                req.request_id,
                wire_pb2.ERR_WRONG_SHARD,
                f"shard {self._shard.shard_id} is not the head shard",
            )
        cache = self._caches.get(req.request_id)
        if cache is None:
            return _error_from_id(
                req.request_id, wire_pb2.ERR_UNKNOWN_REQUEST, "no cache for request_id"
            )
        token_ids = mx.array([[req.token_id]])
        h = embed_tokens(self._lm, token_ids)
        h = self._run_my_layers(h, cache)
        return self._emit_hidden_or_logits(req.request_id, h)

    def _handle_activation(
        self, act: wire_pb2.Activation, tensor_bytes: bytes
    ) -> tuple[wire_pb2.Envelope, bytes]:
        if act.next_layer_idx != self._shard.start_layer:
            return _error_from_id(
                act.request_id,
                wire_pb2.ERR_WRONG_SHARD,
                f"shard expects start_layer={self._shard.start_layer}, "
                f"got next_layer_idx={act.next_layer_idx}",
            )
        cache = self._caches.get(act.request_id)
        if cache is None:
            cache = make_cache(self._lm)
            self._caches[act.request_id] = cache

        h = bytes_to_tensor(
            tensor_bytes, shape=list(act.tensor.shape), dtype=act.tensor.dtype
        )
        h = self._run_my_layers(h, cache)
        return self._emit_hidden_or_logits(act.request_id, h)

    def _handle_end(self, req: wire_pb2.EndRequest) -> tuple[wire_pb2.Envelope, bytes]:
        self._caches.pop(req.request_id, None)
        env = wire_pb2.Envelope()
        env.end.protocol_version = _PROTOCOL_VERSION
        env.end.request_id = req.request_id
        return env, b""

    # ----------------------------------------------------------------- internals

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

    def _emit_hidden_or_logits(
        self, request_id: str, h: mx.array
    ) -> tuple[wire_pb2.Envelope, bytes]:
        if self.is_tail:
            logits = finalize(self._lm, h)
            return _logits_envelope(request_id, logits)
        return _activation_envelope(request_id, self._shard.end_layer, h)


# --------------------------------------------------------------------- helpers


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


def _logits_envelope(
    request_id: str, logits: mx.array
) -> tuple[wire_pb2.Envelope, bytes]:
    raw = tensor_to_bytes(logits)
    env = wire_pb2.Envelope()
    env.logits.protocol_version = _PROTOCOL_VERSION
    env.logits.request_id = request_id
    env.logits.tensor.shape.extend(list(logits.shape))
    env.logits.tensor.dtype = _dtype_to_wire(logits.dtype)
    env.logits.tensor.quant = wire_pb2.QUANT_NONE
    env.logits.tensor.byte_count = len(raw)
    return env, raw


def _extract_request_id(env: wire_pb2.Envelope) -> str | None:
    which = env.WhichOneof("payload")
    if which == "begin":
        return str(env.begin.request_id)
    if which == "cont":
        return str(env.cont.request_id)
    if which == "activation":
        return str(env.activation.request_id)
    if which == "end":
        return str(env.end.request_id)
    if which == "logits":
        return str(env.logits.request_id)
    return None


def _error_from_id(
    request_id: str, code: int, detail: str
) -> tuple[wire_pb2.Envelope, bytes]:
    env = wire_pb2.Envelope()
    env.error.protocol_version = _PROTOCOL_VERSION
    env.error.request_id = request_id
    env.error.code = code
    env.error.detail = detail
    return env, b""


def _error(
    env_in: wire_pb2.Envelope, code: int, detail: str
) -> tuple[wire_pb2.Envelope, bytes]:
    return _error_from_id(_extract_request_id(env_in) or "", code, detail)


__all__ = ["Node"]
