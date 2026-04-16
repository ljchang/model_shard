"""Orchestrator — drives a linear pipeline of shards to produce generated tokens.

Phase 1 wiring:
  * Star topology. Orchestrator holds a TCP connection to each shard and
    forwards activations through them in order.
  * For each new token (prefill or decode step), the first shard receives
    BeginRequest / ContinueRequest and the last shard returns Logits.
  * Greedy sampling: next_token = argmax(logits[:, -1, :]).

This is the single client that exercises the wire protocol against real
Node servers.
"""

from __future__ import annotations

import socket
import uuid
from dataclasses import dataclass
from itertools import pairwise
from typing import BinaryIO, cast

import mlx.core as mx

from model_shard._pb import wire_pb2
from model_shard.envelope import recv_envelope, send_envelope
from model_shard.mlx_engine import bytes_to_tensor
from model_shard.shard_map import ShardMap, ShardSpec


@dataclass
class PrefillCapture:
    """Hidden state observed at the output of a non-final shard during prefill.

    ``next_layer_idx`` is the layer index that the *next* shard would run — so
    this hidden state is comparable to ``ReferenceModel.prefill_trace().layer_inputs[next_layer_idx]``.
    """

    next_layer_idx: int
    hidden: mx.array


@dataclass
class PrefillResult:
    boundary_captures: list[PrefillCapture]
    final_logits: mx.array

_PROTOCOL_VERSION = 1


class Orchestrator:
    def __init__(
        self, shard_map: ShardMap, total_layers: int, hidden_size: int
    ) -> None:
        self._shard_map = shard_map
        self._total_layers = total_layers
        self._hidden_size = hidden_size
        self._pipeline: list[ShardSpec] = _linear_pipeline(shard_map, total_layers)

    def generate_greedy(
        self, prompt_tokens: list[int], max_new_tokens: int
    ) -> list[int]:
        request_id = str(uuid.uuid4())
        streams = [_open_stream(s) for s in self._pipeline]
        try:
            logits = self._run_pipeline(
                streams, _make_begin(request_id, prompt_tokens), b""
            ).final_logits
            next_token = int(mx.argmax(logits[0, -1, :]).item())
            generated: list[int] = [next_token]

            for _ in range(max_new_tokens - 1):
                logits = self._run_pipeline(
                    streams, _make_continue(request_id, next_token), b""
                ).final_logits
                next_token = int(mx.argmax(logits[0, -1, :]).item())
                generated.append(next_token)

            self._send_end(streams, request_id)
            return generated
        finally:
            for conn, _stream in streams:
                conn.close()

    def prefill_with_capture(self, prompt_tokens: list[int]) -> PrefillResult:
        """Run one prefill pass and capture the hidden state at each shard boundary.

        Used by the Tier 2 acceptance test to compare intermediate hidden states
        against the reference oracle. Single-shot: issues BeginRequest followed
        by EndRequest (no decode loop).
        """
        request_id = str(uuid.uuid4())
        streams = [_open_stream(s) for s in self._pipeline]
        try:
            result = self._run_pipeline(
                streams, _make_begin(request_id, prompt_tokens), b"", capture=True
            )
            self._send_end(streams, request_id)
            return result
        finally:
            for conn, _stream in streams:
                conn.close()

    def _run_pipeline(
        self,
        streams: list[tuple[socket.socket, BinaryIO]],
        first_message: wire_pb2.Envelope,
        first_tensor: bytes,
        *,
        capture: bool = False,
    ) -> PrefillResult:
        msg = first_message
        tensor_bytes = first_tensor
        captures: list[PrefillCapture] = []
        for i, (_conn, stream) in enumerate(streams):
            send_envelope(stream, msg, tensor_bytes)
            resp_env, resp_tensor = recv_envelope(stream)
            which = resp_env.WhichOneof("payload")
            is_last = i == len(streams) - 1
            if which == "error":
                raise RuntimeError(
                    f"shard {self._pipeline[i].shard_id} returned error: "
                    f"{resp_env.error.detail}"
                )
            if is_last:
                if which != "logits":
                    raise RuntimeError(
                        f"tail shard {self._pipeline[i].shard_id} returned {which}, "
                        f"expected logits"
                    )
                logits = bytes_to_tensor(
                    resp_tensor,
                    shape=list(resp_env.logits.tensor.shape),
                    dtype=resp_env.logits.tensor.dtype,
                )
                return PrefillResult(boundary_captures=captures, final_logits=logits)
            if which != "activation":
                raise RuntimeError(
                    f"mid shard {self._pipeline[i].shard_id} returned {which}, "
                    f"expected activation"
                )
            if capture:
                hidden = bytes_to_tensor(
                    resp_tensor,
                    shape=list(resp_env.activation.tensor.shape),
                    dtype=resp_env.activation.tensor.dtype,
                )
                captures.append(
                    PrefillCapture(
                        next_layer_idx=int(resp_env.activation.next_layer_idx),
                        hidden=hidden,
                    )
                )
            msg = resp_env
            tensor_bytes = resp_tensor
        raise RuntimeError("empty pipeline")

    def _send_end(
        self,
        streams: list[tuple[socket.socket, BinaryIO]],
        request_id: str,
    ) -> None:
        for _conn, stream in streams:
            end_env = wire_pb2.Envelope()
            end_env.end.protocol_version = _PROTOCOL_VERSION
            end_env.end.request_id = request_id
            send_envelope(stream, end_env)
            recv_envelope(stream)  # drain ack


def _make_begin(request_id: str, prompt_tokens: list[int]) -> wire_pb2.Envelope:
    env = wire_pb2.Envelope()
    env.begin.protocol_version = _PROTOCOL_VERSION
    env.begin.request_id = request_id
    env.begin.sequence_id = request_id
    env.begin.prompt_token_ids.extend(prompt_tokens)
    env.begin.sampling.greedy = True
    env.begin.start_layer = 0
    return env


def _make_continue(request_id: str, token_id: int) -> wire_pb2.Envelope:
    env = wire_pb2.Envelope()
    env.cont.protocol_version = _PROTOCOL_VERSION
    env.cont.request_id = request_id
    env.cont.token_id = token_id
    env.cont.position = 0  # Node tracks via cache offset
    return env


def _linear_pipeline(shard_map: ShardMap, total_layers: int) -> list[ShardSpec]:
    """Sort shards by start_layer and validate they cover [0, total_layers)."""
    shards = sorted(
        (shard_map.lookup(sid) for sid in shard_map.all_shards()),
        key=lambda s: s.start_layer,
    )
    if not shards:
        raise ValueError("shard map is empty")
    if shards[0].start_layer != 0:
        raise ValueError(f"pipeline must start at layer 0, got {shards[0].start_layer}")
    if shards[-1].end_layer != total_layers:
        raise ValueError(
            f"pipeline must end at layer {total_layers}, got {shards[-1].end_layer}"
        )
    for prev, nxt in pairwise(shards):
        if prev.end_layer != nxt.start_layer:
            raise ValueError(
                f"gap/overlap: {prev.shard_id} ends at {prev.end_layer} but "
                f"{nxt.shard_id} starts at {nxt.start_layer}"
            )
    return shards


def _open_stream(shard: ShardSpec) -> tuple[socket.socket, BinaryIO]:
    conn = socket.create_connection((shard.address.host, shard.address.port), timeout=30.0)
    stream = cast(BinaryIO, conn.makefile("rwb", buffering=0))
    return conn, stream


__all__ = ["Orchestrator"]
