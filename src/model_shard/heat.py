"""Per-node heat tracker for Phase 5b expert migration.

Counts how often *this node's router* picks each (layer, expert) pair,
maintained as an EMA (same shape as LoadTracker). Reports the sparse
top-N entries so the gossip payload fits in UDP MTU.
"""

from __future__ import annotations

import threading
from collections import defaultdict


class HeatTracker:
    def __init__(self, alpha: float = 0.3, top_n: int = 16) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if top_n <= 0:
            raise ValueError(f"top_n must be positive, got {top_n}")
        self._alpha = alpha
        self._top_n = top_n
        self._ema: dict[tuple[int, int], float] = defaultdict(float)
        self._lock = threading.Lock()

    def observe(self, layer_idx: int, expert_ids: list[int]) -> None:
        """Record one batch of router picks at ``layer_idx``.

        Counts occurrences per expert id in ``expert_ids`` and folds each
        count into its expert's EMA. Experts present in the tracker but not
        in this batch decay toward zero (alpha weighting)."""
        if not expert_ids:
            return
        # Normalize every id to plain int, not NumPy/MLX int — the encode
        # adapter (_heat_to_pb) relies on protobuf's strict type checking.
        counts: dict[int, int] = defaultdict(int)
        for eid in expert_ids:
            counts[int(eid)] += 1
        with self._lock:
            layer_idx = int(layer_idx)
            # Decay every currently-tracked expert for this layer toward 0
            # by (1-alpha), then fold in the observed counts.
            for (l, e), v in list(self._ema.items()):
                if l == layer_idx and e not in counts:
                    self._ema[(l, e)] = (1.0 - self._alpha) * v
            for eid, c in counts.items():
                prev = self._ema[(layer_idx, eid)]
                self._ema[(layer_idx, eid)] = (
                    self._alpha * float(c) + (1.0 - self._alpha) * prev
                )

    def report(self) -> list[tuple[int, int, int]]:
        """Return [(layer_idx, expert_id, ema_x100), ...] sorted by EMA desc,
        capped at ``top_n``. Suitable for UDP piggyback.

        All ints are plain ``int`` (not NumPy/MLX int subtypes) so protobuf
        encoding via ``_heat_to_pb`` in membership/messages.py never trips
        on type mismatch."""
        with self._lock:
            snapshot = [
                (int(l), int(e), int(round(v * 100.0)))
                for (l, e), v in self._ema.items()
                if v > 0.0
            ]
        snapshot.sort(key=lambda t: t[2], reverse=True)
        return snapshot[: self._top_n]

    def local_heat(self, layer_idx: int, expert_id: int) -> int:
        """Return current EMA×100 for one (layer, expert), or 0 if untracked."""
        with self._lock:
            return int(round(self._ema.get(
                (int(layer_idx), int(expert_id)), 0.0
            ) * 100.0))


__all__ = ["HeatTracker"]
