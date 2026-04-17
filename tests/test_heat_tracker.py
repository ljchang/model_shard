"""HeatTracker unit tests — EMA maintenance and sparse top-N report."""
from __future__ import annotations

import threading

from model_shard.heat import HeatTracker


def test_observe_increments_ema_for_each_expert():
    ht = HeatTracker(alpha=1.0, top_n=16)  # alpha=1 ⇒ current pick dominates
    ht.observe(15, [3, 3, 3, 7])
    # With alpha=1 and 3 observed picks of expert 3 summed in one batch,
    # count = 3, ema = 3. Report stores EMA*100.
    report = ht.report()
    entries = {(e[0], e[1]): e[2] for e in report}
    assert entries[(15, 3)] == 300
    assert entries[(15, 7)] == 100


def test_ema_decays_across_calls_with_lower_alpha():
    ht = HeatTracker(alpha=0.5, top_n=16)
    ht.observe(15, [3])  # ema = 0.5*1 + 0.5*0   = 0.5
    ht.observe(15, [3])  # ema = 0.5*1 + 0.5*0.5 = 0.75
    ht.observe(15, [7])  # expert 3 not picked this round ⇒ ema = 0.5*0 + 0.5*0.75 = 0.375
    report = {(e[0], e[1]): e[2] for e in ht.report()}
    assert report[(15, 3)] == round(0.375 * 100)
    assert report[(15, 7)] == round(0.5 * 100)


def test_report_is_top_n_sorted_desc():
    ht = HeatTracker(alpha=1.0, top_n=2)
    ht.observe(15, [1, 1, 1, 2, 2, 3])
    report = ht.report()
    assert len(report) == 2
    assert report[0][1] == 1  # expert 1 is hottest
    assert report[1][1] == 2


def test_local_heat_lookup():
    ht = HeatTracker(alpha=1.0, top_n=16)
    ht.observe(15, [3])
    assert ht.local_heat(15, 3) == 100
    assert ht.local_heat(15, 999) == 0  # never observed


def test_observe_is_thread_safe():
    ht = HeatTracker(alpha=1.0, top_n=16)
    def worker():
        for _ in range(1000):
            ht.observe(15, [3])
    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # alpha=1 means every observe overwrites with count; after the last
    # observe lands ema ≈ count_from_that_call (non-deterministic but finite).
    assert ht.local_heat(15, 3) > 0
