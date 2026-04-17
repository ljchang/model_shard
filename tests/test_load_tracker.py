"""Fast unit tests for LoadTracker — EMA + jittered report."""

from __future__ import annotations

import random

from model_shard.load import LoadTracker


def test_tracker_initial_report_is_zero() -> None:
    tk = LoadTracker(alpha=0.3, jitter_pct=0.0, rng=random.Random(0))
    assert tk.report() == 0


def test_tracker_ema_converges_on_steady_depth() -> None:
    tk = LoadTracker(alpha=0.5, jitter_pct=0.0, rng=random.Random(0))
    for _ in range(20):
        tk.observe(10)
    assert abs(tk.report() - 1000) < 5


def test_tracker_ema_tracks_step_change() -> None:
    tk = LoadTracker(alpha=0.5, jitter_pct=0.0, rng=random.Random(0))
    for _ in range(20):
        tk.observe(10)
    before = tk.report()
    for _ in range(20):
        tk.observe(0)
    after = tk.report()
    assert after < before // 2


def test_tracker_jitter_bounded() -> None:
    """With jitter_pct=0.1, report is within +/-10% of underlying EMA * 100."""
    tk = LoadTracker(alpha=0.5, jitter_pct=0.1, rng=random.Random(42))
    for _ in range(20):
        tk.observe(10)
    samples = [tk.report() for _ in range(200)]
    assert all(900 <= s <= 1100 for s in samples), f"out of range: {samples[:5]}"
    assert len(set(samples)) > 1


def test_tracker_rng_determinism() -> None:
    tk1 = LoadTracker(alpha=0.5, jitter_pct=0.1, rng=random.Random(7))
    tk2 = LoadTracker(alpha=0.5, jitter_pct=0.1, rng=random.Random(7))
    for _ in range(10):
        tk1.observe(5)
        tk2.observe(5)
    assert [tk1.report() for _ in range(20)] == [tk2.report() for _ in range(20)]


def test_tracker_thread_safe_observe() -> None:
    import threading
    tk = LoadTracker(alpha=0.5, jitter_pct=0.0, rng=random.Random(0))
    def work() -> None:
        for _ in range(100):
            tk.observe(5)
    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert abs(tk.report() - 500) < 10
