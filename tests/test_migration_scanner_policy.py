"""Scanner policy tests — _scan_once picks the hottest not-held expert."""
from __future__ import annotations

import time
from unittest.mock import MagicMock

from model_shard.migration import MigrationPolicy, MigrationScanner


def _make_scanner(
    *,
    heat: dict[tuple[int, int], int],
    live: dict[int, set[int]],
    owners: dict[tuple[int, int], set[str]],
    pulled: list[tuple[int, int, str]],
) -> MigrationScanner:
    ht = MagicMock()
    ht.report.return_value = [(layer, e, v) for (layer, e), v in heat.items()]
    ht.local_heat.side_effect = lambda layer, e: heat.get((layer, e), 0)

    def owner_lookup(layer: int, e: int) -> set[str]:
        return owners.get((layer, e), set())

    def load_provider() -> dict[str, int]:
        return {}

    peer_rpc = MagicMock()
    peer_rpc.pull.return_value = [MagicMock() for _ in range(9)]

    def attacher(layer: int, e: int, tensors: list) -> None:
        live.setdefault(layer, set()).add(e)

    def announce(layer: int, e: int) -> None:
        pulled.append((layer, e, "announced"))

    return MigrationScanner(
        self_shard_id="self",
        policy=MigrationPolicy(
            scan_interval_s=0.0,
            heat_threshold=50,
            max_experts_per_layer=128,
            evict_cooldown_s=0.0,
            eviction_enabled=False,  # disable eviction for pull-only tests
        ),
        heat_tracker=ht,
        live_experts=live,
        owner_lookup=owner_lookup,
        load_provider=load_provider,
        peer_rpc=peer_rpc,
        attacher=attacher,
        ownership_announcer=announce,
        bootstrap_held={},
        attach_ts_provider=lambda lyr, eid: 0.0,
        evict_fn=lambda lyr, eid: None,
    )


def test_scan_once_pulls_hottest_not_held_over_threshold():
    pulled: list = []
    live = {15: {0, 3, 6, 9}}
    owners = {(15, 1): {"peer-a"}, (15, 7): {"peer-b"}}
    heat = {(15, 1): 600, (15, 7): 400, (15, 3): 999}
    s = _make_scanner(heat=heat, live=live, owners=owners, pulled=pulled)
    s._scan_once()
    # Hottest not-held = 1 (heat 600). Expert 3 at heat 999 is already held.
    s._peer_rpc.pull.assert_called_once_with(
        source_shard_id="peer-a", layer_idx=15, expert_id=1
    )
    assert 1 in live[15]
    assert (15, 1, "announced") in pulled


def test_scan_once_respects_threshold():
    pulled: list = []
    live = {15: {0, 3}}
    owners = {(15, 1): {"peer-a"}}
    heat = {(15, 1): 20}  # below 50 threshold
    s = _make_scanner(heat=heat, live=live, owners=owners, pulled=pulled)
    s._scan_once()
    s._peer_rpc.pull.assert_not_called()
    assert 1 not in live[15]


def test_scan_once_respects_max_experts_per_layer():
    live = {15: set(range(128))}  # full stack
    owners = {(15, 128): {"peer-a"}}  # mythical new expert
    s = MigrationScanner(
        self_shard_id="self",
        policy=MigrationPolicy(
            scan_interval_s=0.0, heat_threshold=50, max_experts_per_layer=128,
            evict_cooldown_s=0.0, eviction_enabled=False,
        ),
        heat_tracker=MagicMock(
            report=MagicMock(return_value=[(15, 128, 5000)]),
            local_heat=MagicMock(return_value=5000),
        ),
        live_experts=live,
        owner_lookup=lambda layer, e: owners.get((layer, e), set()),
        load_provider=lambda: {},
        peer_rpc=MagicMock(),
        attacher=lambda layer, e, t: None,
        ownership_announcer=lambda layer, e: None,
        bootstrap_held={},
        attach_ts_provider=lambda lyr, eid: 0.0,
        evict_fn=lambda lyr, eid: None,
    )
    s._scan_once()
    s._peer_rpc.pull.assert_not_called()


def test_scan_once_respects_in_flight_cap():
    pulled: list = []
    live = {15: {0, 3}}
    owners = {(15, 1): {"peer-a"}, (15, 7): {"peer-b"}}
    heat = {(15, 1): 600, (15, 7): 500}
    s = _make_scanner(heat=heat, live=live, owners=owners, pulled=pulled)
    s._scan_once()
    # Single-in-flight cap means one call per scan. The hottest = 1 wins.
    assert s._peer_rpc.pull.call_count == 1


def test_scan_once_skips_own_shard_as_source():
    pulled: list = []
    live = {15: {0, 3}}
    owners = {(15, 1): {"self"}}  # only self owns this expert
    heat = {(15, 1): 600}
    s = _make_scanner(heat=heat, live=live, owners=owners, pulled=pulled)
    s._scan_once()
    # Would be pointless to pull from self.
    s._peer_rpc.pull.assert_not_called()


def test_scanner_start_and_stop_clean():
    live: dict[int, set[int]] = {15: {0, 3}}
    pulled: list = []
    s = _make_scanner(heat={}, live=live, owners={}, pulled=pulled)
    s.start()
    time.sleep(0.1)
    s.stop()
    assert s._thread is not None
    assert not s._thread.is_alive()


def test_scanner_start_is_idempotent():
    live: dict[int, set[int]] = {15: {0, 3}}
    s = _make_scanner(heat={}, live=live, owners={}, pulled=[])
    s.start()
    first_thread = s._thread
    s.start()  # second call should no-op, not create a new thread
    assert s._thread is first_thread
    s.stop()


def test_scan_once_exception_does_not_kill_loop():
    """If scan_once throws, the loop logs and continues instead of exiting."""
    live: dict[int, set[int]] = {15: {0, 3}}
    s = _make_scanner(heat={}, live=live, owners={}, pulled=[])
    call_count = [0]

    # Monkey-patch _scan_once to raise on first call, succeed on second.
    original = s._scan_once
    def flaky():
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("simulated failure")
        original()
    s._scan_once = flaky  # type: ignore[method-assign]
    s._policy = MigrationPolicy(
        scan_interval_s=0.01, heat_threshold=50, max_experts_per_layer=128,
        evict_cooldown_s=0.0, eviction_enabled=False,
    )

    s.start()
    # Give the loop time to tick a few times.
    time.sleep(0.2)
    s.stop()
    # Loop should have continued past the first exception.
    assert call_count[0] >= 2
