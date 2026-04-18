"""Phase 6-C: Node.migration_detach safety invariants."""
from __future__ import annotations

import time
import types
from unittest.mock import MagicMock

import mlx.core as mx
import pytest

from model_shard.node import LastReplicaError, Node
from model_shard.shard_map import NodeAddress, ShardMap, ShardSpec


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("ENABLE_GOSSIP", "false")
    monkeypatch.setenv("ENABLE_PARTIAL_LOAD", "false")
    monkeypatch.setenv("ENABLE_DYNAMIC_MIGRATION", "false")
    monkeypatch.setenv("ENABLE_EXPERT_RETRY", "false")
    monkeypatch.setenv("MIGRATION_EVICT_COOLDOWN_SECONDS", "0")  # disable for most tests
    yield


def _mk_spec(sid: str, port: int, moe: dict[int, tuple[int, ...]]) -> ShardSpec:
    return ShardSpec(
        shard_id=sid,
        address=NodeAddress(host="127.0.0.1", port=port),
        start_layer=0, end_layer=30,
        moe_experts=moe,
    )


def _fake_lm_for_layer(layer_idx: int, held: tuple[int, ...]) -> object:
    """Build a mock LoadedModel with a mutable switch_glu layer sized for `held`."""
    n = len(held)
    def _stack(cols: int) -> mx.array:
        return mx.zeros((n, 4, cols))
    projs = {
        name: types.SimpleNamespace(
            weight=_stack(4), scales=_stack(8), biases=_stack(8),
        )
        for name in ("gate_proj", "up_proj", "down_proj")
    }
    layer = types.SimpleNamespace(
        experts=types.SimpleNamespace(switch_glu=types.SimpleNamespace(**projs))
    )
    text_model = types.SimpleNamespace(layers=[None] * layer_idx + [layer])
    lm = MagicMock()
    lm.text_model = text_model
    lm.held_ids_per_layer = {layer_idx: held}
    return lm


def test_migration_detach_removes_from_live_experts():
    spec = _mk_spec("self", 31100, {15: (0, 3)})
    spec_peer = _mk_spec("peer", 31101, {15: (1, 7)})
    sm = ShardMap({"self": spec, "peer": spec_peer})
    lm = _fake_lm_for_layer(15, (0, 3, 42))
    n = Node(shard=spec, shard_map=sm, loaded_model=lm, total_layers=30)
    # Seed live_experts as if expert 42 was migrated-in.
    n._live_experts[15].add(42)
    n._live_experts_attach_ts[(15, 42)] = 0.0  # cooldown already elapsed
    # Announce peer also owns 42 so last-replica check succeeds.
    n._ownership_view_put("peer", 15, 42, action=0, ts_unix_ms=1_000_000)
    n.migration_detach(15, 42)
    assert 42 not in n._live_experts[15]
    assert (15, 42) not in n._live_experts_attach_ts


def test_migration_detach_rejects_bootstrap_held():
    spec = _mk_spec("self", 31102, {15: (0, 3)})
    spec_peer = _mk_spec("peer", 31103, {15: (0, 1)})
    sm = ShardMap({"self": spec, "peer": spec_peer})
    lm = _fake_lm_for_layer(15, (0, 3))
    n = Node(shard=spec, shard_map=sm, loaded_model=lm, total_layers=30)
    # Expert 0 is in self's bootstrap moe_experts.
    with pytest.raises(ValueError, match="bootstrap"):
        n.migration_detach(15, 0)


def test_migration_detach_rejects_within_cooldown(monkeypatch):
    monkeypatch.setenv("MIGRATION_EVICT_COOLDOWN_SECONDS", "30")
    spec = _mk_spec("self", 31104, {15: (0, 3)})
    spec_peer = _mk_spec("peer", 31105, {15: (42, 7)})
    sm = ShardMap({"self": spec, "peer": spec_peer})
    lm = _fake_lm_for_layer(15, (0, 3, 42))
    n = Node(shard=spec, shard_map=sm, loaded_model=lm, total_layers=30)
    n._live_experts[15].add(42)
    n._live_experts_attach_ts[(15, 42)] = time.time()  # just attached
    n._ownership_view_put("peer", 15, 42, action=0, ts_unix_ms=1_000_000)
    with pytest.raises(ValueError, match="cooldown"):
        n.migration_detach(15, 42)


def test_migration_detach_last_replica_raises():
    spec = _mk_spec("self", 31106, {15: (0, 3)})
    spec_peer = _mk_spec("peer", 31107, {15: (1, 7)})
    sm = ShardMap({"self": spec, "peer": spec_peer})
    lm = _fake_lm_for_layer(15, (0, 3, 42))
    n = Node(shard=spec, shard_map=sm, loaded_model=lm, total_layers=30)
    n._live_experts[15].add(42)
    n._live_experts_attach_ts[(15, 42)] = 0.0
    # NO other owner announced for expert 42 — self is sole owner.
    with pytest.raises(LastReplicaError):
        n.migration_detach(15, 42)


def test_migration_detach_announces_remove_on_gossip():
    spec = _mk_spec("self", 31108, {15: (0, 3)})
    spec_peer = _mk_spec("peer", 31109, {15: (42, 7)})
    sm = ShardMap({"self": spec, "peer": spec_peer})
    lm = _fake_lm_for_layer(15, (0, 3, 42))
    n = Node(shard=spec, shard_map=sm, loaded_model=lm, total_layers=30)
    n._live_experts[15].add(42)
    n._live_experts_attach_ts[(15, 42)] = 0.0
    n._ownership_view_put("peer", 15, 42, action=0, ts_unix_ms=1_000_000)
    # Inject a membership mock that records calls.
    n._membership = MagicMock()
    n.migration_detach(15, 42)
    n._membership.announce_ownership_remove.assert_called_once_with(15, 42)


def test_migration_attach_records_ts():
    spec = _mk_spec("self", 31110, {15: (0, 3)})
    spec_peer = _mk_spec("peer", 31111, {15: (1, 7)})
    sm = ShardMap({"self": spec, "peer": spec_peer})
    lm = _fake_lm_for_layer(15, (0, 3))
    n = Node(shard=spec, shard_map=sm, loaded_model=lm, total_layers=30)
    # Attach expert 42. Build 9 synthetic tensors in canonical _PROJ_ATTR_ORDER:
    # (gate_proj/weight, gate_proj/scales, gate_proj/biases) * 3 projections
    tensors = [mx.zeros((4, 4)), mx.zeros((4, 8)), mx.zeros((4, 8))] * 3
    t_before = time.time()
    n.migration_attach(15, 42, tensors)
    t_after = time.time()
    assert (15, 42) in n._live_experts_attach_ts
    assert t_before <= n._live_experts_attach_ts[(15, 42)] <= t_after


def test_migration_detach_resets_heat():
    """After eviction, the evicted expert's heat EMA must be reset so the
    scanner doesn't immediately re-pull it."""
    spec = _mk_spec("self", 31200, {15: (0, 3)})
    spec_peer = _mk_spec("peer", 31201, {15: (42, 7)})
    sm = ShardMap({"self": spec, "peer": spec_peer})
    lm = _fake_lm_for_layer(15, (0, 3, 42))
    n = Node(shard=spec, shard_map=sm, loaded_model=lm, total_layers=30)
    n._live_experts[15].add(42)
    n._live_experts_attach_ts[(15, 42)] = 0.0
    n._ownership_view_put("peer", 15, 42, action=0, ts_unix_ms=1_000_000)
    # Seed heat for expert 42.
    n._heat_tracker.observe(15, [42, 42, 42])
    assert n._heat_tracker.local_heat(15, 42) > 0
    # Evict.
    n.migration_detach(15, 42)
    # Heat for the evicted expert must be reset.
    assert n._heat_tracker.local_heat(15, 42) == 0
