"""Phase 5b target-pull migration: policy + peer RPC + scanner.

Layering:
  * ExpertWeightPeerRPC — TCP client for ExpertWeightRequest/Transfer.
  * MigrationPolicy     — knobs (thresholds, intervals). [added in Task 15]
  * MigrationScanner    — periodic daemon thread that decides + pulls. [Task 15]
"""

from __future__ import annotations

import logging
import random as _random
import socket
import threading as _threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, BinaryIO, cast

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


@dataclass(frozen=True)
class MigrationPolicy:
    scan_interval_s: float
    heat_threshold: int
    max_experts_per_layer: int


class MigrationScanner:
    """Periodic target-pull scanner (Phase 5b decider, simple threshold).

    Responsibilities:
      * Rank locally-routed experts by heat (EMA x 100).
      * Skip experts this node already hosts or layers at capacity.
      * Below threshold -> no migration this tick.
      * Single in-flight cap prevents stampedes on a single node.
      * Picks the least-loaded peer owner via `load_provider` as source.

    Task 16 adds the background-thread lifecycle on top of this.
    """

    def __init__(
        self,
        self_shard_id: str,
        policy: MigrationPolicy,
        heat_tracker: Any,
        live_experts: dict[int, set[int]],
        owner_lookup: Callable[[int, int], set[str]],
        load_provider: Callable[[], dict[str, int]],
        peer_rpc: Any,
        attacher: Callable[[int, int, list[mx.array]], None],
        ownership_announcer: Callable[[int, int], None],
        rng: _random.Random | None = None,
    ) -> None:
        self._self_shard_id = self_shard_id
        self._policy = policy
        self._heat_tracker = heat_tracker
        self._live_experts = live_experts
        self._owner_lookup = owner_lookup
        self._load_provider = load_provider
        self._peer_rpc = peer_rpc
        self._attacher = attacher
        self._ownership_announcer = ownership_announcer
        self._rng = rng or _random.Random()
        self._stopping = _threading.Event()
        self._thread: _threading.Thread | None = None
        self._in_flight = _threading.Lock()

    def _select_candidate(self) -> tuple[int, int, str] | None:
        """Return (layer_idx, expert_id, source_shard_id) or None."""
        report = sorted(
            self._heat_tracker.report(), key=lambda t: t[2], reverse=True
        )
        for layer_idx, expert_id, _ema in report:
            held = self._live_experts.get(layer_idx, set())
            if expert_id in held:
                continue
            if len(held) >= self._policy.max_experts_per_layer:
                continue
            if self._heat_tracker.local_heat(
                layer_idx, expert_id
            ) < self._policy.heat_threshold:
                continue
            owners = self._owner_lookup(layer_idx, expert_id) - {self._self_shard_id}
            if not owners:
                continue
            loads = self._load_provider()
            source = min(owners, key=lambda s: loads.get(s, 2**31 - 1))
            return layer_idx, expert_id, source
        return None

    def _scan_once(self) -> None:
        if not self._in_flight.acquire(blocking=False):
            return
        try:
            pick = self._select_candidate()
            if pick is None:
                return
            layer_idx, expert_id, source = pick
            try:
                tensors = self._peer_rpc.pull(
                    source_shard_id=source,
                    layer_idx=layer_idx,
                    expert_id=expert_id,
                )
            except Exception:
                _LOG.exception(
                    "migration pull failed: %s layer=%d expert=%d",
                    source, layer_idx, expert_id,
                )
                return
            try:
                self._attacher(layer_idx, expert_id, tensors)
            except Exception:
                _LOG.exception(
                    "attach failed after pull: layer=%d expert=%d",
                    layer_idx, expert_id,
                )
                return
            self._ownership_announcer(layer_idx, expert_id)
        finally:
            self._in_flight.release()


__all__ = ["ExpertWeightPeerRPC", "MigrationPolicy", "MigrationScanner"]
