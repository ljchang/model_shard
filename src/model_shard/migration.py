"""Phase 5b target-pull migration: policy + peer RPC + scanner.

Layering:
  * ExpertWeightPeerRPC — TCP client for ExpertWeightRequest/Transfer.
  * MigrationPolicy     — knobs (thresholds, intervals). [added in Task 15]
  * MigrationScanner    — periodic daemon thread that decides + pulls. [Task 15]
"""

from __future__ import annotations

import logging
import socket
from typing import BinaryIO, cast

import mlx.core as mx

from model_shard._pb import wire_pb2
from model_shard.envelope import recv_envelope, send_envelope
from model_shard.mlx_engine import bytes_to_tensor

_LOG = logging.getLogger(__name__)


class ExpertWeightPeerRPC:
    """TCP client for pulling expert weights from a source shard.

    Opens a short-lived connection per call; target sends
    ``ExpertWeightRequest`` and blocks on ``ExpertWeightTransfer``. Splits
    the 9-tensor out-of-band payload using each descriptor's ``byte_count``.
    """

    def __init__(
        self,
        addresses: dict[str, tuple[str, int]],
        timeout_s: float,
    ) -> None:
        self._addresses = addresses
        self._timeout_s = timeout_s

    def pull(
        self,
        source_shard_id: str,
        layer_idx: int,
        expert_id: int,
    ) -> list[mx.array]:
        host, port = self._addresses[source_shard_id]
        s = socket.create_connection((host, port), timeout=self._timeout_s)
        s.settimeout(self._timeout_s)
        try:
            stream = cast(BinaryIO, s.makefile("rwb"))
            req = wire_pb2.Envelope()
            req.expert_weight_request.protocol_version = 1
            req.expert_weight_request.request_id = (
                f"pull-{layer_idx}-{expert_id}-{id(self)}"
            )
            req.expert_weight_request.layer_idx = layer_idx
            req.expert_weight_request.expert_id = expert_id
            send_envelope(stream, req)
            stream.flush()

            env, tensor_bytes = recv_envelope(stream)
            which = env.WhichOneof("payload")
            if which == "error":
                raise RuntimeError(
                    f"source {source_shard_id} returned error "
                    f"{env.error.code}: {env.error.detail}"
                )
            if which != "expert_weight_transfer":
                raise RuntimeError(
                    f"unexpected payload from source {source_shard_id}: {which}"
                )
            resp = env.expert_weight_transfer
            if int(resp.tensor_count) != 9 or len(resp.tensors) != 9:
                raise RuntimeError(
                    f"ExpertWeightTransfer must have 9 tensors, "
                    f"got tensor_count={resp.tensor_count} len={len(resp.tensors)}"
                )
            offset = 0
            out: list[mx.array] = []
            for d in resp.tensors:
                nbytes = int(d.byte_count)
                blob = tensor_bytes[offset : offset + nbytes]
                if len(blob) != nbytes:
                    raise RuntimeError(
                        f"ExpertWeightTransfer payload short: "
                        f"descriptor byte_count={nbytes}, got {len(blob)}"
                    )
                offset += nbytes
                arr = bytes_to_tensor(blob, shape=list(d.shape), dtype=d.dtype)
                out.append(arr)
            if offset != len(tensor_bytes):
                raise RuntimeError(
                    f"ExpertWeightTransfer payload had {len(tensor_bytes) - offset} "
                    f"trailing bytes after 9 tensors"
                )
            return out
        finally:
            s.close()


__all__ = ["ExpertWeightPeerRPC"]
