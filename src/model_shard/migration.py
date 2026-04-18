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
    evict_cooldown_s: float = 30.0
    eviction_enabled: bool = True


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
        bootstrap_held: dict[int, set[int]],
        attach_ts_provider: Callable[[int, int], float],
        evict_fn: Callable[[int, int], None],
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
        self._bootstrap_held = bootstrap_held
        self._attach_ts_provider = attach_ts_provider
        self._evict_fn = evict_fn
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
            self._maybe_pull_one()
            self._maybe_evict_one()
        finally:
            self._in_flight.release()

    def _maybe_pull_one(self) -> None:
        """Existing pull logic extracted to its own method so eviction can
        run alongside cleanly (both under the single in-flight lock)."""
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

    def _maybe_evict_one(self) -> None:
        """Eviction pass — runs after _maybe_pull_one under the same in-flight
        lock. Only fires when a layer is at capacity. Picks the coldest
        non-bootstrap, non-cooldown expert. Last-replica guard is enforced by
        the evict_fn (Node.migration_detach) raising LastReplicaError; the
        scanner catches it and moves on to the next layer."""
        if not self._policy.eviction_enabled:
            return
        import time as _time
        now = _time.time()
        for layer_idx in list(self._live_experts.keys()):
            held = set(self._live_experts.get(layer_idx, set()))
            if len(held) < self._policy.max_experts_per_layer:
                continue
            bootstrap = self._bootstrap_held.get(layer_idx, set())
            eligible = {
                e for e in held - bootstrap
                if now - self._attach_ts_provider(layer_idx, e)
                   >= self._policy.evict_cooldown_s
            }
            if not eligible:
                continue
            victim = min(
                eligible, key=lambda e: self._heat_tracker.local_heat(layer_idx, e)
            )
            try:
                self._evict_fn(layer_idx, victim)
            except Exception:
                # LastReplicaError, or any other evict-side refusal: try next layer.
                _LOG.exception(
                    "eviction skipped: layer=%d expert=%d",
                    layer_idx, victim,
                )
                continue
            return  # evict at most one per tick

    def start(self) -> None:
        """Start the background scan thread. Idempotent."""
        if self._thread is not None:
            return
        self._thread = _threading.Thread(
            target=self._run_loop, name="migration-scanner", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the scan thread to stop and join. Idempotent."""
        self._stopping.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run_loop(self) -> None:
        """Scan on a jittered interval until stopped.

        The ±25% jitter per tick avoids synchronized cluster-wide scans that
        could stampede the same hot expert simultaneously (see spec §R5).
        Per-tick exceptions are logged and do not halt the loop — the scanner
        must be robust against downstream failures in pull/attach/announce.
        """
        while not self._stopping.is_set():
            jitter = 1.0 + self._rng.uniform(-0.25, 0.25)
            self._stopping.wait(self._policy.scan_interval_s * jitter)
            if self._stopping.is_set():
                return
            try:
                self._scan_once()
            except Exception:
                _LOG.exception("scan_once raised")


__all__ = ["ExpertWeightPeerRPC", "MigrationPolicy", "MigrationScanner"]
