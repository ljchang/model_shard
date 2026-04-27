"""Thin client — submits a prompt to the head node, streams sampled tokens back.

Has no pipeline logic. Does not know which nodes exist beyond the head. All
forwarding between shards happens inside the node network.
"""

from __future__ import annotations

import socket
import uuid
from collections.abc import Iterator
from contextlib import closing
from typing import BinaryIO, cast

from model_shard._pb import wire_pb2
from model_shard.envelope import recv_envelope, send_envelope
from model_shard.shard_map import NodeAddress

_PROTOCOL_VERSION = 1


class Client:
    def __init__(self, head_address: NodeAddress) -> None:
        self._head = head_address

    def generate(
        self, prompt_tokens: list[int], max_new_tokens: int
    ) -> list[int]:
        """Blocking: returns all generated tokens once the stream completes."""
        return list(self.generate_streaming(prompt_tokens, max_new_tokens))

    def generate_streaming(
        self, prompt_tokens: list[int], max_new_tokens: int
    ) -> Iterator[int]:
        """Yield tokens as they arrive. Closes connection on is_final or EOF."""
        request_id = str(uuid.uuid4())
        conn = socket.create_connection(
            (self._head.host, self._head.port), timeout=30.0
        )
        # Read timeout has to absorb cold-start prefill JIT at any shard in
        # the pipeline (HF transformers' _grouped_mm on Grace Blackwell +
        # CUDA 13 compiles a fresh kernel per (batch, seq_len) shape, which
        # can be ~3 min on first hit). Steady-state requests return in
        # seconds; SWIM gossip is the dead-peer detector, not the socket
        # read timeout. 300s is a generous backstop, not a perf target.
        conn.settimeout(300.0)
        with closing(conn), cast(BinaryIO, conn.makefile("rwb", buffering=0)) as stream:
            begin = wire_pb2.Envelope()
            begin.begin.protocol_version = _PROTOCOL_VERSION
            begin.begin.request_id = request_id
            begin.begin.sequence_id = request_id
            begin.begin.prompt_token_ids.extend(prompt_tokens)
            begin.begin.sampling.greedy = True
            begin.begin.start_layer = 0
            begin.begin.max_new_tokens = max_new_tokens
            send_envelope(stream, begin)

            while True:
                env, _ = recv_envelope(stream)
                which = env.WhichOneof("payload")
                if which == "sampled_token":
                    yield int(env.sampled_token.token_id)
                    if env.sampled_token.is_final:
                        return
                elif which == "error":
                    raise RuntimeError(
                        f"server error {env.error.code}: {env.error.detail}"
                    )
                else:
                    raise RuntimeError(f"unexpected envelope {which} from head")


__all__ = ["Client"]
