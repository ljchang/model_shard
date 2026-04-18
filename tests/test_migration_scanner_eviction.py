"""Phase 6-C: MigrationScanner._maybe_evict_one policy tests."""
from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

from model_shard.migration import MigrationPolicy, MigrationScanner


def _base_policy() -> MigrationPolicy:
    return MigrationPolicy(
        scan_interval_s=0.0,
        heat_threshold=50,
        max_experts_per_layer=3,
        evict_cooldown_s=0.0,
        eviction_enabled=True,
    )


def _make_scanner(
    *,
    live_experts: dict[int, set[int]],
    bootstrap_held: dict[int, set[int]],
    attach_ts: dict[tuple[int, int], float],
    heat: dict[tuple[int, int], int] | None = None,
    policy: MigrationPolicy | None = None,
    evicted: list[tuple[int, int]] | None = None,
) -> MigrationScanner:
    ht = MagicMock()
    ht.report.return_value = []
    ht.local_heat.side_effect = lambda lyr, eid: (heat or {}).get((lyr, eid), 0)
    peer_rpc = MagicMock()
    evicted_list = evicted if evicted is not None else []
    def evict_fn(lyr: int, eid: int) -> None:
        evicted_list.append((lyr, eid))
        live_experts.get(lyr, set()).discard(eid)
    return MigrationScanner(
        self_shard_id="self",
        policy=policy or _base_policy(),
        heat_tracker=ht,
        live_experts=live_experts,
        owner_lookup=lambda lyr, eid: set(),
        load_provider=lambda: {},
        peer_rpc=peer_rpc,
        attacher=lambda lyr, eid, t: None,
        ownership_announcer=lambda lyr, eid: None,
        bootstrap_held=bootstrap_held,
        attach_ts_provider=lambda lyr, eid: attach_ts.get((lyr, eid), 0.0),
        evict_fn=evict_fn,
    )


def test_maybe_evict_one_below_capacity_is_noop():
    live = {15: {0, 3}}  # below capacity (3)
    bootstrap = {15: {0}}
    attach_ts = {(15, 3): 0.0}
    evicted: list = []
    s = _make_scanner(
        live_experts=live, bootstrap_held=bootstrap,
        attach_ts=attach_ts, evicted=evicted,
    )
    s._maybe_evict_one()
    assert evicted == []


def test_maybe_evict_one_at_capacity_evicts_coldest_non_bootstrap():
    live = {15: {0, 3, 7}}  # at capacity (3)
    bootstrap = {15: {0}}    # only 0 is bootstrap
    attach_ts = {(15, 3): 0.0, (15, 7): 0.0}
    heat = {(15, 3): 1000, (15, 7): 50}  # 7 is colder
    evicted: list = []
    s = _make_scanner(
        live_experts=live, bootstrap_held=bootstrap,
        attach_ts=attach_ts, heat=heat, evicted=evicted,
    )
    s._maybe_evict_one()
    assert evicted == [(15, 7)]


def test_maybe_evict_one_skips_bootstrap_held_even_if_coldest():
    live = {15: {0, 3, 7}}
    bootstrap = {15: {0, 7}}  # 0 and 7 are both bootstrap
    attach_ts = {(15, 3): 0.0}
    heat = {(15, 3): 1000, (15, 7): 0, (15, 0): 500}
    evicted: list = []
    s = _make_scanner(
        live_experts=live, bootstrap_held=bootstrap,
        attach_ts=attach_ts, heat=heat, evicted=evicted,
    )
    s._maybe_evict_one()
    # Only non-bootstrap expert is 3 → must be evicted.
    assert evicted == [(15, 3)]


def test_maybe_evict_one_skips_within_cooldown():
    import time
    live = {15: {0, 3, 7}}
    bootstrap = {15: {0}}
    attach_ts = {(15, 3): time.time(), (15, 7): time.time()}  # just attached
    heat = {(15, 3): 1000, (15, 7): 0}
    evicted: list = []
    s = _make_scanner(
        live_experts=live, bootstrap_held=bootstrap,
        attach_ts=attach_ts, heat=heat, evicted=evicted,
        policy=replace(_base_policy(), evict_cooldown_s=30.0),
    )
    s._maybe_evict_one()
    assert evicted == []


def test_maybe_evict_one_disabled_is_noop():
    live = {15: {0, 3, 7}}
    bootstrap = {15: {0}}
    attach_ts = {(15, 3): 0.0, (15, 7): 0.0}
    heat = {(15, 3): 1000, (15, 7): 0}
    evicted: list = []
    s = _make_scanner(
        live_experts=live, bootstrap_held=bootstrap,
        attach_ts=attach_ts, heat=heat, evicted=evicted,
        policy=replace(_base_policy(), eviction_enabled=False),
    )
    s._maybe_evict_one()
    assert evicted == []
