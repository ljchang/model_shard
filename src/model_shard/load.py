"""EMA-based queue-depth tracker for Phase 4 load-aware routing.

Observes integer depth samples, maintains an exponential moving average,
and produces a jittered integer report (EMA * 100) suitable for gossip.
Thread-safe for concurrent observe() calls from handler threads while
report() is called from the gossip thread.
"""

from __future__ import annotations

import random
import threading


class LoadTracker:
    def __init__(
        self,
        alpha: float = 0.3,
        jitter_pct: float = 0.1,
        rng: random.Random | None = None,
    ) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if jitter_pct < 0.0:
            raise ValueError(f"jitter_pct must be >= 0, got {jitter_pct}")
        self._alpha = alpha
        self._jitter_pct = jitter_pct
        self._rng = rng if rng is not None else random.Random()
        self._ema: float = 0.0
        self._lock = threading.Lock()

    def observe(self, depth: int) -> None:
        """Record one queue-depth sample."""
        with self._lock:
            self._ema = self._alpha * depth + (1.0 - self._alpha) * self._ema

    def report(self) -> int:
        """Return jittered EMA scaled by 100 (integer wire form)."""
        with self._lock:
            ema = self._ema
        jitter = 1.0
        if self._jitter_pct > 0.0:
            jitter = 1.0 + self._rng.uniform(-self._jitter_pct, self._jitter_pct)
        return max(0, round(ema * 100.0 * jitter))


__all__ = ["LoadTracker"]
