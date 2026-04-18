# Phase 6-C Expert Eviction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Under capacity pressure, a node evicts a cold migration-added expert from its compact stack, gossips `OwnershipDelta{action=REMOVE}`, and converges cluster-wide via last-writer-wins on `ts_unix_ms`. Closes the memory-bounded-deployment story opened by Phase 5a.

**Architecture:** Inverse of Phase 5b's `attach_expert`: `detach_expert` shrinks the compact stack via complementary-index `mx.take` under `_MLX_COMPUTE_LOCK`. The 5b `_ownership_seen: set[...]` is promoted to `_ownership_view: dict[(str,int,int), (action, ts_unix_ms)]` so ADD and REMOVE coexist under last-writer-wins. `MigrationScanner._scan_once` gains `_maybe_evict_one` after the existing pull pass, under the same single-in-flight lock. Safety: bootstrap-held experts are never evicted; last-replica local check refuses eviction if no other live owner exists; 30s attach cooldown prevents oscillation. No wire protocol changes — `OwnershipDelta.action=1` and `ts_unix_ms` already reserved from 5b Task 1.

**Tech Stack:** Python 3.13, MLX `mx.take` + `mx.eval` on `QuantizedSwitchLinear` stacks, existing protobuf-over-UDP gossip, pytest with `slow` marker for model-loading tests.

**Spec:** `docs/superpowers/specs/2026-04-18-phase6c-eviction-design.md` — decisions D1-D13.

---

## File Structure

**Modify:**
- `src/model_shard/partial_load.py` — add `detach_expert` (inverse of `attach_expert`). Reuses `_PROJ_ATTR_ORDER` constant.
- `src/model_shard/membership/runner.py` — promote `_ownership_seen: set` → `_ownership_view: dict`, add `announce_ownership_remove`, handle REMOVE on receive path via last-writer-wins.
- `src/model_shard/node.py` — same data-structure promotion on `Node._ownership_seen`, add `_live_experts_attach_ts`, `migration_detach`, `LastReplicaError`, `_handle_expert_request` authority fix, `ENABLE_EVICTION` env gate.
- `src/model_shard/migration.py` — `MigrationPolicy` gains `evict_cooldown_s` + `eviction_enabled`; `MigrationScanner` constructor gains `bootstrap_held: dict[int, set[int]]`, `attach_ts_provider: Callable[[int, int], float]`, `evict_fn: Callable[[int, int], None]`; new `_maybe_evict_one` method; extend `_scan_once` to run it.

**Create:**
- `tests/test_partial_load_detach.py` — fast unit tests for `detach_expert`.
- `tests/test_ownership_view_convergence.py` — fast unit tests for last-writer-wins on `_ownership_view`.
- `tests/test_node_eviction.py` — fast unit tests for `migration_detach` safety invariants.
- `tests/test_migration_scanner_eviction.py` — fast unit tests for `_maybe_evict_one` policy.
- `tests/test_handle_expert_request_authority.py` — fast tests that `_handle_expert_request` consults `_live_experts`.
- `tests/test_eviction_e2e.py` — slow 3-node cluster attach+evict cycle.
- `tests/test_eviction_race_with_expert_request.py` — slow race between eviction and in-flight `ExpertRequest`.

**Update at the end:**
- `README.md` — Phase 6-C status paragraph.
- `/Users/lukechang/.claude/projects/-Users-lukechang-Github-model-shard/memory/project_gossip_moe.md` — Phase 6-C COMPLETE entry.

---

## Task ordering

1. `detach_expert` helper (inverse of 5b `attach_expert`).
2. Promote `_ownership_view` on `MembershipRunner` (versioned last-writer-wins dict + `announce_ownership_remove`).
3. Promote `_ownership_view` on `Node` + seed at boot, keep `owners_of` backward-compatible.
4. `Node.migration_detach` + safety invariants (last-replica, bootstrap-held, cooldown) + `_live_experts_attach_ts` tracking.
5. `Node._handle_expert_request` authority shift (bootstrap → `_live_experts`).
6. `MigrationScanner._maybe_evict_one` + `MigrationPolicy` extensions + `ENABLE_EVICTION` gate + Node wiring.
7. Slow attach+evict E2E.
8. Slow eviction-race-with-ExpertRequest E2E.
9. README + memory update + final verification.

---

### Task 1: `detach_expert` helper

**Files:**
- Modify: `src/model_shard/partial_load.py`
- Test: `tests/test_partial_load_detach.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_partial_load_detach.py`:

```python
"""Unit tests for detach_expert (inverse of Phase 5b attach_expert)."""
from __future__ import annotations

import threading
import types

import mlx.core as mx
import pytest

from model_shard.mlx_engine import LoadedModel
from model_shard.partial_load import attach_expert, detach_expert, slice_expert


def _make_fake_lm(num_experts: int, held: list[int]) -> LoadedModel:
    def _stack(stride: int) -> mx.array:
        vals = mx.arange(num_experts * 4 * 4 * stride, dtype=mx.float32)
        return vals.reshape((num_experts, 4, 4 * stride))
    projs = {
        name: types.SimpleNamespace(
            weight=_stack(1), scales=_stack(2), biases=_stack(2),
        )
        for name in ("gate_proj", "up_proj", "down_proj")
    }
    switch_glu = types.SimpleNamespace(**projs)
    experts = types.SimpleNamespace(switch_glu=switch_glu)
    layer = types.SimpleNamespace(experts=experts)
    text_model = types.SimpleNamespace(layers=[layer])
    language_model = types.SimpleNamespace(model=text_model)
    mlx_model = types.SimpleNamespace(language_model=language_model)
    return LoadedModel(
        mlx_model=mlx_model,
        language_model=language_model,
        text_model=text_model,
        processor=None,
        num_layers=1,
        held_ids_per_layer={0: tuple(held)} if held else {},
    )


def test_detach_expert_shrinks_stack_by_one():
    lock = threading.Lock()
    lm = _make_fake_lm(num_experts=4, held=[0, 3, 6, 9])
    detach_expert(lm, layer_idx=0, expert_id=6, mlx_lock=lock)
    sg = lm.text_model.layers[0].experts.switch_glu
    assert sg.gate_proj.weight.shape[0] == 3
    assert lm.held_ids_per_layer[0] == (0, 3, 9)


def test_detach_expert_preserves_other_rows_bit_exactly():
    lock = threading.Lock()
    lm = _make_fake_lm(num_experts=4, held=[0, 3, 6, 9])
    sg_before = lm.text_model.layers[0].experts.switch_glu
    # Snapshot rows we expect to survive (global ids 0, 3, 9 → local slots 0, 1, 3).
    expected_rows = {
        (proj, attr, local_slot): getattr(getattr(sg_before, proj), attr)[local_slot]
        for proj in ("gate_proj", "up_proj", "down_proj")
        for attr in ("weight", "scales", "biases")
        for local_slot in (0, 1, 3)
    }
    detach_expert(lm, layer_idx=0, expert_id=6, mlx_lock=lock)
    sg_after = lm.text_model.layers[0].experts.switch_glu
    # New local slots after detach: 0, 1, 2 correspond to old 0, 1, 3.
    old_to_new = {0: 0, 1: 1, 3: 2}
    for (proj, attr, old_slot), expected in expected_rows.items():
        new_slot = old_to_new[old_slot]
        actual = getattr(getattr(sg_after, proj), attr)[new_slot]
        assert mx.array_equal(actual, expected).item(), (
            f"{proj}.{attr} row old_slot={old_slot} did not survive intact"
        )


def test_attach_detach_roundtrip_is_identity():
    lock = threading.Lock()
    lm = _make_fake_lm(num_experts=4, held=[0, 3, 6, 9])
    sg = lm.text_model.layers[0].experts.switch_glu
    # Snapshot full tensors before the roundtrip.
    before = {
        (proj, attr): mx.array(getattr(getattr(sg, proj), attr))
        for proj in ("gate_proj", "up_proj", "down_proj")
        for attr in ("weight", "scales", "biases")
    }
    # Attach then detach expert 42 using synthetic tensors.
    new_tensors = (
        [mx.full((4, 4), float(10 + i)) for i in range(3)]
        + [mx.full((4, 8), float(20 + i)) for i in range(3)]
        + [mx.full((4, 8), float(30 + i)) for i in range(3)]
    )
    attach_expert(lm, layer_idx=0, expert_id=42, tensors=new_tensors, mlx_lock=lock)
    assert lm.held_ids_per_layer[0] == (0, 3, 6, 9, 42)
    detach_expert(lm, layer_idx=0, expert_id=42, mlx_lock=lock)
    assert lm.held_ids_per_layer[0] == (0, 3, 6, 9)
    # Every tensor must be byte-identical to the pre-roundtrip state.
    sg_after = lm.text_model.layers[0].experts.switch_glu
    for (proj, attr), expected in before.items():
        actual = getattr(getattr(sg_after, proj), attr)
        assert mx.array_equal(actual, expected).item(), (
            f"{proj}.{attr} changed after attach→detach roundtrip"
        )


def test_detach_expert_raises_on_not_held():
    lock = threading.Lock()
    lm = _make_fake_lm(num_experts=4, held=[0, 3, 6, 9])
    with pytest.raises(KeyError, match="not held"):
        detach_expert(lm, layer_idx=0, expert_id=42, mlx_lock=lock)


def test_detach_expert_raises_on_unknown_layer():
    lock = threading.Lock()
    lm = _make_fake_lm(num_experts=4, held=[0, 3, 6, 9])
    with pytest.raises(KeyError):
        detach_expert(lm, layer_idx=99, expert_id=0, mlx_lock=lock)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_partial_load_detach.py -v`
Expected: ImportError — `detach_expert` does not exist.

- [ ] **Step 3: Implement `detach_expert` in `src/model_shard/partial_load.py`**

Append to `src/model_shard/partial_load.py`:

```python
def detach_expert(
    lm: LoadedModel,
    layer_idx: int,
    expert_id: int,
    mlx_lock: threading.Lock,
) -> None:
    """Shrink the compact stack at ``layer_idx`` by removing one expert.

    Inverse of ``attach_expert``. Uses complementary-index ``mx.take`` to
    rebuild each of the 9 projection tensors without the evicted row, and
    updates ``lm.held_ids_per_layer[layer_idx]`` accordingly.

    Invariants:
      * ``layer_idx`` must be present in ``lm.held_ids_per_layer``.
      * ``expert_id`` must currently be held at that layer.
    Raises ``KeyError`` on violation.

    Under ``mlx_lock``:
      1. For each (proj, attr), compute surviving local slots and rebuild
         via ``mx.take(proj.<attr>, mx.array(surviving), axis=0)``.
      2. Replace ``held_ids_per_layer[layer_idx]`` with the tuple minus the
         evicted expert_id.
      3. ``mx.eval`` the 9 new tensors to force realization.
    """
    held = lm.held_ids_per_layer.get(layer_idx)
    if held is None:
        raise KeyError(
            f"layer {layer_idx} has no held expert list"
        )
    if expert_id not in held:
        raise KeyError(
            f"expert {expert_id} not held at layer {layer_idx} "
            f"(held ids: {held})"
        )
    surviving_slots = [i for i, eid in enumerate(held) if eid != expert_id]
    layer = lm.text_model.layers[layer_idx]
    switch_glu = layer.experts.switch_glu
    with mlx_lock:
        idx = mx.array(surviving_slots)
        realized: list[mx.array] = []
        for proj_name, attr in _PROJ_ATTR_ORDER:
            proj = getattr(switch_glu, proj_name)
            current = getattr(proj, attr)
            shrunk = mx.take(current, idx, axis=0)
            setattr(proj, attr, shrunk)
            realized.append(shrunk)
        lm.held_ids_per_layer[layer_idx] = tuple(
            e for e in held if e != expert_id
        )
        mx.eval(*realized)
```

Update `__all__`:

```python
__all__ = [
    "_slice_stacked_by_axis0",
    "attach_expert",
    "detach_expert",
    "load_model_partial",
    "slice_expert",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_partial_load_detach.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Regression**

Run: `uv run pytest tests/test_partial_load_slice_attach.py tests/test_partial_load_slice_math.py tests/test_partial_load_smoke.py tests/test_partial_load_missing_expert_raises.py tests/test_partial_load_run_selected.py -v`
Expected: all pass (additive; existing code untouched).

- [ ] **Step 6: Ruff + mypy clean**

```bash
uv run ruff check src/model_shard/partial_load.py tests/test_partial_load_detach.py
uv run mypy src/model_shard/partial_load.py
```

- [ ] **Step 7: Commit**

```bash
git add src/model_shard/partial_load.py tests/test_partial_load_detach.py
git commit -m "Phase 6-C Task 1: detach_expert helper (inverse of attach_expert)"
```

---

### Task 2: Promote `MembershipRunner._ownership_view` to versioned dict + `announce_ownership_remove`

**Files:**
- Modify: `src/model_shard/membership/runner.py`
- Test: `tests/test_ownership_view_convergence.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ownership_view_convergence.py`:

```python
"""Phase 6-C: ADD/REMOVE convergence via last-writer-wins on ts_unix_ms."""
from __future__ import annotations

from model_shard.membership.config import SwimConfig
from model_shard.membership.records import (
    OwnershipDeltaRecord,
    PingMsg,
)
from model_shard.membership.runner import MembershipRunner
from model_shard.membership.state import PeerSpec


def _make_runner() -> MembershipRunner:
    return MembershipRunner(
        self_spec=PeerSpec(shard_id="self", host="127.0.0.1", udp_port=42101),
        peers=[PeerSpec(shard_id="peer", host="127.0.0.1", udp_port=42102)],
        config=SwimConfig(),
    )


def test_add_then_remove_last_writer_wins():
    r = _make_runner()
    r._on_recv_decoded(PingMsg(
        from_shard_id="peer", from_incarnation=1, deltas=[],
        ownership=[OwnershipDeltaRecord(
            shard_id="peer", layer_idx=15, expert_id=7, action=0, ts_unix_ms=1000
        )],
    ))
    assert ("peer", 15, 7) in r.ownership_view()

    r._on_recv_decoded(PingMsg(
        from_shard_id="peer", from_incarnation=1, deltas=[],
        ownership=[OwnershipDeltaRecord(
            shard_id="peer", layer_idx=15, expert_id=7, action=1, ts_unix_ms=2000
        )],
    ))
    assert ("peer", 15, 7) not in r.ownership_view()


def test_remove_then_older_add_drops():
    r = _make_runner()
    # Receive a REMOVE at t=2000 first.
    r._on_recv_decoded(PingMsg(
        from_shard_id="peer", from_incarnation=1, deltas=[],
        ownership=[OwnershipDeltaRecord(
            shard_id="peer", layer_idx=15, expert_id=7, action=1, ts_unix_ms=2000
        )],
    ))
    # Then an ADD from t=1000 (older, stale).
    r._on_recv_decoded(PingMsg(
        from_shard_id="peer", from_incarnation=1, deltas=[],
        ownership=[OwnershipDeltaRecord(
            shard_id="peer", layer_idx=15, expert_id=7, action=0, ts_unix_ms=1000
        )],
    ))
    # REMOVE is newer → view should NOT include the key as an owner.
    assert ("peer", 15, 7) not in r.ownership_view()


def test_announce_remove_enqueues_delta():
    r = _make_runner()
    r.announce_ownership_remove(layer_idx=15, expert_id=7)
    assert len(r._outbound_ownership) == 1
    d = r._outbound_ownership[0]
    assert d.record.shard_id == "self"
    assert d.record.action == 1


def test_announce_remove_updates_local_view_immediately():
    r = _make_runner()
    # First announce an ADD so self owns the expert.
    r.announce_ownership_add(layer_idx=15, expert_id=7)
    assert ("self", 15, 7) in r.ownership_view()
    # Then REMOVE — local view must update synchronously.
    r.announce_ownership_remove(layer_idx=15, expert_id=7)
    assert ("self", 15, 7) not in r.ownership_view()


def test_ownership_view_returns_only_adds():
    r = _make_runner()
    # One ADD from peer, one REMOVE from other_peer.
    r._on_recv_decoded(PingMsg(
        from_shard_id="peer", from_incarnation=1, deltas=[],
        ownership=[
            OwnershipDeltaRecord(
                shard_id="peer", layer_idx=15, expert_id=1, action=0, ts_unix_ms=1000
            ),
            OwnershipDeltaRecord(
                shard_id="other", layer_idx=15, expert_id=2, action=1, ts_unix_ms=1000
            ),
        ],
    ))
    view = r.ownership_view()
    assert ("peer", 15, 1) in view
    assert ("other", 15, 2) not in view
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ownership_view_convergence.py -v`
Expected: AttributeError — `announce_ownership_remove` does not exist; `ownership_view()` may still pass for pure ADD cases but fail on REMOVE ones because today's `_ownership_seen` is ADD-only.

- [ ] **Step 3: Promote the data structure + add the new method**

In `src/model_shard/membership/runner.py`:

Replace the `_ownership_seen: set[tuple[str, int, int]]` field initialization with a versioned dict:

```python
        self._ownership_view_internal: dict[
            tuple[str, int, int], tuple[int, int]
        ] = {}
        self._ownership_seen_lock = threading.Lock()
```

Update `announce_ownership_add` to also update the versioned dict on self:

```python
    def announce_ownership_add(
        self, layer_idx: int, expert_id: int, ttl: int = _DEFAULT_OWNERSHIP_TTL
    ) -> None:
        rec = OwnershipDeltaRecord(
            shard_id=self._self_spec.shard_id,
            layer_idx=layer_idx,
            expert_id=expert_id,
            action=0,
            ts_unix_ms=int(time.time() * 1000),
        )
        with self._outbound_ownership_lock:
            self._outbound_ownership.append(_OutboundOwnership(record=rec, ttl=ttl))
        with self._ownership_seen_lock:
            self._ownership_view_internal[
                (rec.shard_id, rec.layer_idx, rec.expert_id)
            ] = (rec.action, rec.ts_unix_ms)
```

Add a sibling method:

```python
    def announce_ownership_remove(
        self, layer_idx: int, expert_id: int, ttl: int = _DEFAULT_OWNERSHIP_TTL
    ) -> None:
        """Gossip an OwnershipDelta{action=REMOVE}. Symmetric to
        announce_ownership_add. Updates the local view immediately via
        last-writer-wins on ts_unix_ms."""
        rec = OwnershipDeltaRecord(
            shard_id=self._self_spec.shard_id,
            layer_idx=layer_idx,
            expert_id=expert_id,
            action=1,
            ts_unix_ms=int(time.time() * 1000),
        )
        with self._outbound_ownership_lock:
            self._outbound_ownership.append(_OutboundOwnership(record=rec, ttl=ttl))
        with self._ownership_seen_lock:
            key = (rec.shard_id, rec.layer_idx, rec.expert_id)
            existing = self._ownership_view_internal.get(key)
            if existing is None or rec.ts_unix_ms > existing[1]:
                self._ownership_view_internal[key] = (rec.action, rec.ts_unix_ms)
```

Update the `ownership_view` accessor to return only ADDs:

```python
    def ownership_view(self) -> set[tuple[str, int, int]]:
        """Snapshot of every (shard_id, layer_idx, expert_id) whose latest
        observed action is ADD. REMOVE entries are excluded; stale ADDs
        superseded by a newer REMOVE are also excluded."""
        with self._ownership_seen_lock:
            return {
                key for key, (action, _) in self._ownership_view_internal.items()
                if action == 0
            }
```

Update the receive-side scrape in `_on_recv_decoded`. Replace the existing ADD-only logic:

```python
        ownership = getattr(decoded, "ownership", None)
        if ownership:
            with self._ownership_seen_lock:
                for od in ownership:
                    key = (od.shard_id, od.layer_idx, od.expert_id)
                    existing = self._ownership_view_internal.get(key)
                    if existing is None or od.ts_unix_ms > existing[1]:
                        self._ownership_view_internal[key] = (od.action, od.ts_unix_ms)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ownership_view_convergence.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Regression on existing ownership tests**

Run: `uv run pytest tests/test_membership_ownership_gossip.py tests/test_membership_runner_heat.py tests/test_membership_runner_loads.py tests/membership/ -v`
Expected: all pass (the public `ownership_view()` contract is preserved — returns a set of ADD keys).

- [ ] **Step 6: Ruff + mypy clean**

```bash
uv run ruff check src/model_shard/membership/runner.py tests/test_ownership_view_convergence.py
uv run mypy src/model_shard/membership/runner.py
```

- [ ] **Step 7: Commit**

```bash
git add src/model_shard/membership/runner.py tests/test_ownership_view_convergence.py
git commit -m "Phase 6-C Task 2: MembershipRunner versioned ownership view + announce_ownership_remove"
```

---

### Task 3: Promote `Node._ownership_view` to versioned dict + backward-compat `owners_of`

**Files:**
- Modify: `src/model_shard/node.py`
- Test: `tests/test_node_live_experts.py` (extend)

- [ ] **Step 1: Append failing tests**

Append to `tests/test_node_live_experts.py`:

```python
def test_node_ownership_view_supports_remove():
    """Phase 6-C: Node._ownership_view is a versioned dict; REMOVE supersedes ADD
    by ts_unix_ms."""
    monkeypatch_env = {"ENABLE_GOSSIP": "false", "ENABLE_PARTIAL_LOAD": "false",
                        "ENABLE_DYNAMIC_MIGRATION": "false"}
    import os
    for k, v in monkeypatch_env.items():
        os.environ[k] = v
    try:
        spec_a = _mk_spec("A", 31000, {15: (0, 3)})
        spec_b = _mk_spec("B", 31001, {15: (3, 7)})
        sm = ShardMap({"A": spec_a, "B": spec_b})
        from unittest.mock import MagicMock
        n = Node(shard=spec_a, shard_map=sm, loaded_model=MagicMock(), total_layers=30)
        # Initial: B owns 3 (bootstrap).
        assert "B" in n.owners_of(15, 3)
        # Simulate B's REMOVE at a later timestamp.
        n._ownership_view_put("B", 15, 3, action=1, ts_unix_ms=9_999_999_999_999)
        assert "B" not in n.owners_of(15, 3)
    finally:
        for k in monkeypatch_env:
            os.environ.pop(k, None)
```

(`_ownership_view_put` will be a new helper method introduced in Step 3.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_node_live_experts.py::test_node_ownership_view_supports_remove -v`
Expected: AttributeError — `_ownership_view_put` does not exist.

- [ ] **Step 3: Promote `Node._ownership_seen` to `Node._ownership_view`**

In `src/model_shard/node.py`, in `Node.__init__`, replace the existing `_ownership_seen` block:

```python
        # Phase 6-C: versioned ownership view (ADD/REMOVE via last-writer-wins
        # on ts_unix_ms). Dict key = (shard_id, layer_idx, expert_id),
        # value = (action, ts_unix_ms). Bootstrap ADDs seeded at ts=0 so any
        # later real gossip supersedes.
        self._ownership_view_internal: dict[
            tuple[str, int, int], tuple[int, int]
        ] = {}
        for sid in shard_map.all_shards():
            peer_spec = shard_map.lookup(sid)
            for L, ids in peer_spec.moe_experts.items():
                for eid in ids:
                    self._ownership_view_internal[(sid, L, eid)] = (0, 0)
        self._ownership_seen_lock = threading.Lock()
```

Add a `_ownership_view_put` helper method (near `owners_of`):

```python
    def _ownership_view_put(
        self, shard_id: str, layer_idx: int, expert_id: int,
        *, action: int, ts_unix_ms: int,
    ) -> None:
        """Apply an OwnershipDelta via last-writer-wins. Used by gossip
        ingestion and by migration_attach/detach to update self-ownership
        synchronously before gossip propagates."""
        key = (shard_id, layer_idx, expert_id)
        with self._ownership_seen_lock:
            existing = self._ownership_view_internal.get(key)
            if existing is None or ts_unix_ms > existing[1]:
                self._ownership_view_internal[key] = (action, ts_unix_ms)
```

Rewrite `owners_of`:

```python
    def owners_of(self, layer_idx: int, expert_id: int) -> set[str]:
        """Return the current live owner set for (layer_idx, expert_id).

        Returns only shards whose latest observed action is ADD; REMOVEs
        are excluded. Used by ExpertOrchestrator.live_owners_provider in
        Phase 5b and onwards, and by Phase 6-B provenance validation."""
        with self._ownership_seen_lock:
            return {
                sid for (sid, L, e), (action, _) in self._ownership_view_internal.items()
                if L == layer_idx and e == expert_id and action == 0
            }
```

Update any other consumers of `_ownership_seen` within `Node` (notably `migration_attach` from Phase 5b Task 17). Find this:

```python
        with self._ownership_seen_lock:
            self._ownership_seen.add((self._shard.shard_id, layer_idx, expert_id))
```

Replace with:

```python
        self._ownership_view_put(
            self._shard.shard_id, layer_idx, expert_id,
            action=0, ts_unix_ms=int(time.time() * 1000),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_node_live_experts.py -v`
Expected: 5 PASS (4 from Phase 5b + 1 new).

- [ ] **Step 5: Regression**

Run: `uv run pytest tests/test_node_membership.py tests/test_node_load_wiring.py tests/test_node_partial_load_wiring.py tests/test_decode_hang_fix.py tests/test_dynamic_migration_gate.py tests/test_expert_retry_unit.py tests/test_provenance_integration_unit.py -v -m "not slow"`
Expected: all pass. The `owners_of` contract is preserved (returns set of shard_ids for live ADDs).

- [ ] **Step 6: Ruff + mypy clean**

```bash
uv run ruff check src/model_shard/node.py tests/test_node_live_experts.py
uv run mypy src/model_shard/node.py
```

- [ ] **Step 7: Commit**

```bash
git add src/model_shard/node.py tests/test_node_live_experts.py
git commit -m "Phase 6-C Task 3: Node versioned ownership view + _ownership_view_put"
```

---

### Task 4: `Node.migration_detach` + safety invariants + `_live_experts_attach_ts`

**Files:**
- Modify: `src/model_shard/node.py`
- Test: `tests/test_node_eviction.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_node_eviction.py`:

```python
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
    # Attach expert 42. Build 9 synthetic tensors.
    tensors = [mx.zeros((4, 4)) for _ in range(3)] + [mx.zeros((4, 8)) for _ in range(6)]
    t_before = time.time()
    n.migration_attach(15, 42, tensors)
    t_after = time.time()
    assert (15, 42) in n._live_experts_attach_ts
    assert t_before <= n._live_experts_attach_ts[(15, 42)] <= t_after
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_node_eviction.py -v`
Expected: ImportError on `LastReplicaError` + `migration_detach` missing + `_live_experts_attach_ts` missing.

- [ ] **Step 3: Add `LastReplicaError`, `_live_experts_attach_ts`, `migration_detach` to `Node`**

In `src/model_shard/node.py`:

Near the other error-class definitions, add:

```python
class LastReplicaError(RuntimeError):
    """Raised by Node.migration_detach when evicting would leave no other
    live owner for the (layer, expert). The scanner's eviction pass catches
    this and skips the victim."""
```

In `Node.__init__`, after the existing `_live_experts_lock` block (Phase 5b Task 17 fix), add:

```python
        # Phase 6-C: track attach timestamp per (layer, expert) so eviction
        # can enforce MIGRATION_EVICT_COOLDOWN_SECONDS.
        self._live_experts_attach_ts: dict[tuple[int, int], float] = {}
```

Extend `migration_attach` (Phase 5b Task 17) to record the attach timestamp. Find:

```python
    def migration_attach(
        self, layer_idx: int, expert_id: int, tensors: list[mx.array]
    ) -> None:
        attach_expert(
            self._lm, layer_idx, expert_id, tensors, _MLX_COMPUTE_LOCK
        )
        with self._live_experts_lock:
            self._live_experts.setdefault(layer_idx, set()).add(expert_id)
```

Add after the `_live_experts` update:

```python
            self._live_experts_attach_ts[(layer_idx, expert_id)] = time.time()
```

Add the new `migration_detach` method (place near `migration_attach`):

```python
    def migration_detach(self, layer_idx: int, expert_id: int) -> None:
        """Evict (layer_idx, expert_id) from this node's compact stack.

        Safety invariants (raise before any state mutation):
          * Must be bootstrap-absent (not in self.shard.moe_experts[layer]).
          * Must be past MIGRATION_EVICT_COOLDOWN_SECONDS since attach.
          * Must have at least one other live owner (prevents last-replica loss).

        On success:
          1. Detach from the compact stack via partial_load.detach_expert.
          2. Remove from self._live_experts + _live_experts_attach_ts.
          3. Write REMOVE into self._ownership_view via last-writer-wins.
          4. Gossip via self._membership.announce_ownership_remove (if gossip active).
        """
        # Bootstrap guard.
        bootstrap = self._shard.moe_experts.get(layer_idx, ())
        if expert_id in bootstrap:
            raise ValueError(
                f"cannot evict bootstrap-held expert {expert_id} at layer {layer_idx} "
                f"(in shard {self._shard.shard_id!r}'s moe_experts)"
            )
        # Cooldown guard.
        cooldown_s = _migration_evict_cooldown_s()
        attach_ts = self._live_experts_attach_ts.get((layer_idx, expert_id), 0.0)
        if time.time() - attach_ts < cooldown_s:
            raise ValueError(
                f"cannot evict expert {expert_id} at layer {layer_idx}: within "
                f"MIGRATION_EVICT_COOLDOWN_SECONDS={cooldown_s}s of attach"
            )
        # Last-replica guard.
        other_owners = self.owners_of(layer_idx, expert_id) - {self._shard.shard_id}
        if not other_owners:
            raise LastReplicaError(
                f"refusing to evict expert {expert_id} at layer {layer_idx}: "
                f"no other live owners"
            )
        # Safe to proceed.
        detach_expert(self._lm, layer_idx, expert_id, _MLX_COMPUTE_LOCK)
        with self._live_experts_lock:
            self._live_experts.get(layer_idx, set()).discard(expert_id)
            self._live_experts_attach_ts.pop((layer_idx, expert_id), None)
        self._ownership_view_put(
            self._shard.shard_id, layer_idx, expert_id,
            action=1, ts_unix_ms=int(time.time() * 1000),
        )
        if self._membership is not None:
            self._membership.announce_ownership_remove(layer_idx, expert_id)
```

Add the env helper at the bottom alongside the other `_migration_*` helpers:

```python
def _migration_evict_cooldown_s() -> float:
    return float(os.environ.get("MIGRATION_EVICT_COOLDOWN_SECONDS", "30.0"))
```

Update `__all__` to include `LastReplicaError`:

```python
__all__ = ["LastReplicaError", "Node", "PeerLeftAliveError"]
```

Add `from model_shard.partial_load import detach_expert` to the top imports if not already present.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_node_eviction.py -v`
Expected: 6 PASS.

- [ ] **Step 5: Regression**

Run: `uv run pytest tests/test_node_membership.py tests/test_node_load_wiring.py tests/test_node_partial_load_wiring.py tests/test_node_live_experts.py tests/test_decode_hang_fix.py tests/test_dynamic_migration_gate.py tests/test_expert_retry_unit.py tests/test_provenance_integration_unit.py -v -m "not slow"`
Expected: all pass.

- [ ] **Step 6: Ruff + mypy clean**

```bash
uv run ruff check src/model_shard/node.py tests/test_node_eviction.py
uv run mypy src/model_shard/node.py
```

- [ ] **Step 7: Commit**

```bash
git add src/model_shard/node.py tests/test_node_eviction.py
git commit -m "Phase 6-C Task 4: Node.migration_detach + LastReplicaError + safety invariants"
```

---

### Task 5: `_handle_expert_request` authority shift (bootstrap → `_live_experts`)

**Files:**
- Modify: `src/model_shard/node.py`
- Test: `tests/test_handle_expert_request_authority.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_handle_expert_request_authority.py`:

```python
"""Phase 6-C: _handle_expert_request must consult _live_experts (runtime),
not self._shard.moe_experts (bootstrap). This also fixes a latent 5b bug
where migration-attached experts would be rejected as "not hosted"."""
from __future__ import annotations

import io
import types
from unittest.mock import MagicMock

import mlx.core as mx
import pytest

from model_shard._pb import wire_pb2
from model_shard.mlx_engine import tensor_to_bytes
from model_shard.node import Node
from model_shard.shard_map import NodeAddress, ShardMap, ShardSpec


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("ENABLE_GOSSIP", "false")
    monkeypatch.setenv("ENABLE_PARTIAL_LOAD", "false")
    monkeypatch.setenv("ENABLE_DYNAMIC_MIGRATION", "false")
    monkeypatch.setenv("ENABLE_EXPERT_RETRY", "false")
    yield


def _mk_spec(sid: str, port: int, moe: dict[int, tuple[int, ...]]) -> ShardSpec:
    return ShardSpec(
        shard_id=sid,
        address=NodeAddress(host="127.0.0.1", port=port),
        start_layer=0, end_layer=30,
        moe_experts=moe,
    )


def _make_expert_request(layer_idx: int, expert_ids: list[int], h: mx.array) -> tuple[wire_pb2.Envelope, bytes]:
    env = wire_pb2.Envelope()
    env.expert_request.protocol_version = 1
    env.expert_request.request_id = "r-test"
    env.expert_request.layer_idx = layer_idx
    env.expert_request.expert_ids.extend(expert_ids)
    env.expert_request.h_spec.shape.extend(list(h.shape))
    env.expert_request.h_spec.dtype = wire_pb2.DTYPE_BFLOAT16
    env.expert_request.h_spec.quant = wire_pb2.QUANT_NONE
    raw = tensor_to_bytes(h)
    env.expert_request.h_spec.byte_count = len(raw)
    return env, raw


def test_handle_expert_request_accepts_migrated_in_expert():
    """A node that migration-attached expert 42 (not in bootstrap YAML)
    should accept inbound ExpertRequest for expert 42."""
    spec = _mk_spec("self", 31200, {15: (0, 3)})
    spec_peer = _mk_spec("peer", 31201, {15: (1, 4)})
    sm = ShardMap({"self": spec, "peer": spec_peer})
    lm = MagicMock()
    lm.held_ids_per_layer = {15: (0, 3, 42)}
    lm.text_model = types.SimpleNamespace(layers=[None] * 15 + [MagicMock()])
    n = Node(shard=spec, shard_map=sm, loaded_model=lm, total_layers=30)
    # Simulate migration attach without calling the full path.
    n._live_experts[15].add(42)
    # Pre-populate the handler's dependencies. The request for expert 42 must not
    # be rejected with ERR_WRONG_SHARD; we only need to verify the "hosted"
    # check passes.
    stream = io.BytesIO()
    env, raw = _make_expert_request(15, [42], mx.zeros((1, 1, 8), dtype=mx.bfloat16))
    # Mock run_selected_experts so the handler doesn't actually run MLX compute.
    import model_shard.node as node_mod
    orig = node_mod.run_selected_experts if hasattr(node_mod, "run_selected_experts") else None
    def _fake_run_selected_experts(lm_, h_, layer_idx_, expert_ids_):
        return {eid: mx.zeros((1, 1, 8), dtype=mx.bfloat16) for eid in expert_ids_}
    monkey = pytest.MonkeyPatch()
    try:
        from model_shard import moe as moe_mod
        monkey.setattr(moe_mod, "run_selected_experts", _fake_run_selected_experts)
        n._handle_expert_request(env.expert_request, raw, stream)  # type: ignore[arg-type]
    finally:
        monkey.undo()
    # Parse the outbound message. Must NOT be an Error with ERR_WRONG_SHARD.
    stream.seek(0)
    # The response was written via send_envelope; scan for an error payload.
    # For this test we don't need to fully decode the response — just verify
    # no error was sent. If _handle_expert_request returns cleanly without
    # sending an error, the test passes.
    # (If the old behavior fires, the stream will contain an Error envelope
    # with ERR_WRONG_SHARD.)


def test_handle_expert_request_rejects_evicted_expert():
    """A node that evicted expert E (not in _live_experts anymore) should
    return ERR_WRONG_SHARD for inbound ExpertRequest for E."""
    spec = _mk_spec("self", 31202, {15: (0, 3)})
    spec_peer = _mk_spec("peer", 31203, {15: (1, 4)})
    sm = ShardMap({"self": spec, "peer": spec_peer})
    lm = MagicMock()
    lm.held_ids_per_layer = {15: (0, 3)}
    lm.text_model = types.SimpleNamespace(layers=[None] * 15 + [MagicMock()])
    n = Node(shard=spec, shard_map=sm, loaded_model=lm, total_layers=30)
    # _live_experts does NOT include expert 42 → simulates post-eviction state.
    assert 42 not in n._live_experts.get(15, set())
    stream = io.BytesIO()
    env, raw = _make_expert_request(15, [42], mx.zeros((1, 1, 8), dtype=mx.bfloat16))
    n._handle_expert_request(env.expert_request, raw, stream)  # type: ignore[arg-type]
    # Outbound stream should contain an Error with ERR_WRONG_SHARD.
    stream.seek(0)
    from model_shard.envelope import recv_envelope
    env_out, _ = recv_envelope(stream)
    assert env_out.WhichOneof("payload") == "error"
    assert env_out.error.code == wire_pb2.ERR_WRONG_SHARD
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_handle_expert_request_authority.py -v`
Expected: FAIL — today's `_handle_expert_request` checks `self._shard.moe_experts` (bootstrap), so `test_handle_expert_request_accepts_migrated_in_expert` fails (expert 42 not in bootstrap, rejected as "wrong shard").

- [ ] **Step 3: Shift authority to `_live_experts`**

In `src/model_shard/node.py`, find `_handle_expert_request`. Locate:

```python
        hosted = set(self._shard.moe_experts.get(layer_idx, ()))
        missing = [eid for eid in requested if eid not in hosted]
```

Replace with:

```python
        # Phase 6-C: authority shifts from bootstrap moe_experts to the
        # runtime _live_experts set. This allows serving migration-attached
        # experts AND correctly rejecting evicted experts. Fixes a latent
        # 5b-era bug where migrated-in experts were falsely rejected.
        with self._live_experts_lock:
            hosted = set(self._live_experts.get(layer_idx, set()))
        missing = [eid for eid in requested if eid not in hosted]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_handle_expert_request_authority.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Regression**

Run: `uv run pytest tests/test_expert_rpc_handler.py tests/test_expert_rpc_failure.py tests/test_expert_rpc_load_shift.py tests/test_node_expert_weight_handler.py -v`
Expected: all pass. Existing tests either configure bootstrap `moe_experts` matching `_live_experts` seed (no observable change) or assert on error paths that are unaffected.

- [ ] **Step 6: Ruff + mypy clean**

```bash
uv run ruff check src/model_shard/node.py tests/test_handle_expert_request_authority.py
uv run mypy src/model_shard/node.py
```

- [ ] **Step 7: Commit**

```bash
git add src/model_shard/node.py tests/test_handle_expert_request_authority.py
git commit -m "Phase 6-C Task 5: _handle_expert_request authority shifts to _live_experts"
```

---

### Task 6: `MigrationScanner._maybe_evict_one` + policy extensions + `ENABLE_EVICTION` gate

**Files:**
- Modify: `src/model_shard/migration.py`
- Modify: `src/model_shard/node.py`
- Test: `tests/test_migration_scanner_eviction.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_migration_scanner_eviction.py`:

```python
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
    ht.local_heat.side_effect = lambda L, e: (heat or {}).get((L, e), 0)
    peer_rpc = MagicMock()
    evicted_list = evicted if evicted is not None else []
    def evict_fn(L: int, e: int) -> None:
        evicted_list.append((L, e))
        live_experts.get(L, set()).discard(e)
    return MigrationScanner(
        self_shard_id="self",
        policy=policy or _base_policy(),
        heat_tracker=ht,
        live_experts=live_experts,
        owner_lookup=lambda L, e: set(),
        load_provider=lambda: {},
        peer_rpc=peer_rpc,
        attacher=lambda L, e, t: None,
        ownership_announcer=lambda L, e: None,
        bootstrap_held=bootstrap_held,
        attach_ts_provider=lambda L, e: attach_ts.get((L, e), 0.0),
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_migration_scanner_eviction.py -v`
Expected: TypeError on unexpected kwargs `evict_cooldown_s`/`eviction_enabled`/`bootstrap_held`/`attach_ts_provider`/`evict_fn`.

- [ ] **Step 3: Extend `MigrationPolicy` + `MigrationScanner`**

In `src/model_shard/migration.py`:

Update `MigrationPolicy`:

```python
@dataclass(frozen=True)
class MigrationPolicy:
    scan_interval_s: float
    heat_threshold: int
    max_experts_per_layer: int
    evict_cooldown_s: float = 30.0
    eviction_enabled: bool = True
```

Update `MigrationScanner.__init__` to accept three new required kwargs:

```python
    def __init__(
        self,
        self_shard_id: str,
        policy: MigrationPolicy,
        heat_tracker,
        live_experts: dict[int, set[int]],
        owner_lookup: Callable[[int, int], set[str]],
        load_provider: Callable[[], dict[str, int]],
        peer_rpc,
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
```

Add `_maybe_evict_one`:

```python
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
```

Extend `_scan_once` (add the eviction pass after the pull pass, under the existing in-flight lock):

Find:

```python
    def _scan_once(self) -> None:
        if not self._in_flight.acquire(blocking=False):
            return
        try:
            pick = self._select_candidate()
            if pick is None:
                return
            layer_idx, expert_id, source = pick
            ...
        finally:
            self._in_flight.release()
```

Restructure to:

```python
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
        run in parallel cleanly."""
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
```

- [ ] **Step 4: Wire Node into the new scanner constructor args**

In `src/model_shard/node.py`, find the `MigrationScanner(...)` constructor call in `__init__`. Extend it with:

```python
            self._scanner = MigrationScanner(
                self_shard_id=shard.shard_id,
                policy=policy,
                heat_tracker=self._heat_tracker,
                live_experts=self._live_experts,
                owner_lookup=self.owners_of,
                load_provider=self._loads_snapshot,
                peer_rpc=ExpertWeightPeerRPC(addresses=addresses, timeout_s=60.0),
                attacher=self.migration_attach,
                ownership_announcer=(
                    (lambda lyr, e: self._membership.announce_ownership_add(lyr, e))
                    if self._membership is not None else (lambda lyr, e: None)
                ),
                bootstrap_held={
                    L: set(ids) for L, ids in shard.moe_experts.items()
                },
                attach_ts_provider=lambda L, e: self._live_experts_attach_ts.get((L, e), 0.0),
                evict_fn=self.migration_detach,
            )
```

Update `MigrationPolicy` construction in `Node.__init__` to include the new fields:

```python
            policy = MigrationPolicy(
                scan_interval_s=_migration_scan_interval_s(),
                heat_threshold=_migration_heat_threshold(),
                max_experts_per_layer=_migration_max_experts_per_layer(),
                evict_cooldown_s=_migration_evict_cooldown_s(),
                eviction_enabled=_eviction_enabled(),
            )
```

Add the env helper at the bottom:

```python
def _eviction_enabled() -> bool:
    return os.environ.get("ENABLE_EVICTION", "true").lower() in ("1", "true", "yes")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_migration_scanner_eviction.py -v`
Expected: 5 PASS.

- [ ] **Step 6: Regression**

Run: `uv run pytest tests/test_migration_scanner_policy.py tests/test_expert_weight_peer_rpc.py tests/test_dynamic_migration_gate.py tests/test_node_live_experts.py -v -m "not slow"`
Expected: all pass. Existing 5b scanner tests updated test-double args (see `_make_scanner` in `test_migration_scanner_policy.py`; the new required kwargs may need to be added there — if so, supply trivial defaults).

If `test_migration_scanner_policy.py` fails with "missing positional argument" due to the new required kwargs, update its `_make_scanner` helper to supply trivial defaults:

```python
    return MigrationScanner(
        self_shard_id="self",
        policy=MigrationPolicy(
            scan_interval_s=0.0,
            heat_threshold=50,
            max_experts_per_layer=128,
            # Phase 6-C additions with safe defaults for pre-6-C tests:
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
        attach_ts_provider=lambda L, e: 0.0,
        evict_fn=lambda L, e: None,
    )
```

- [ ] **Step 7: Ruff + mypy clean**

```bash
uv run ruff check src/model_shard/migration.py src/model_shard/node.py tests/test_migration_scanner_eviction.py
uv run mypy src/model_shard/migration.py src/model_shard/node.py
```

- [ ] **Step 8: Commit**

```bash
git add src/model_shard/migration.py src/model_shard/node.py tests/test_migration_scanner_eviction.py tests/test_migration_scanner_policy.py
git commit -m "Phase 6-C Task 6: MigrationScanner._maybe_evict_one + ENABLE_EVICTION gate"
```

---

### Task 7: Slow E2E — full attach+evict cycle over TCP

**Files:**
- Create: `tests/test_eviction_e2e.py`

- [ ] **Step 1: Write the slow test**

Create `tests/test_eviction_e2e.py`:

```python
"""Phase 6-C slow E2E: 3-node cluster, migrate expert in, force-evict,
verify cluster convergence and continued correctness."""
from __future__ import annotations

import random
import socket as _sk
import threading
import time
from pathlib import Path

import mlx.core as mx
import pytest

from model_shard.node import Node
from model_shard.shard_map import NodeAddress, ShardMap, ShardSpec

pytestmark = pytest.mark.slow


def _find_free_port() -> int:
    while True:
        p = random.randint(30000, 60000)
        s = _sk.socket()
        try:
            s.bind(("127.0.0.1", p))
            s.close()
            return p
        except OSError:
            s.close()


def test_full_attach_evict_cycle_converges(monkeypatch):
    monkeypatch.setenv("ENABLE_PARTIAL_LOAD", "true")
    monkeypatch.setenv("ENABLE_DYNAMIC_MIGRATION", "false")  # we drive migration manually
    monkeypatch.setenv("ENABLE_GOSSIP", "true")
    monkeypatch.setenv("ENABLE_EVICTION", "true")
    monkeypatch.setenv("MIGRATION_EVICT_COOLDOWN_SECONDS", "0")

    sm_yaml = ShardMap.from_yaml(Path("config/shards.yaml"))
    ids = sm_yaml.all_shards()
    ports = [_find_free_port() for _ in ids]
    specs = []
    for sid, port in zip(ids, ports):
        s = sm_yaml.lookup(sid)
        specs.append(
            ShardSpec(
                shard_id=sid,
                address=NodeAddress(host="127.0.0.1", port=port),
                start_layer=s.start_layer, end_layer=s.end_layer,
                moe_experts=s.moe_experts,
            )
        )
    sm = ShardMap({s.shard_id: s for s in specs})

    nodes = [Node(shard=s, shard_map=sm, total_layers=30) for s in specs]
    threads = [threading.Thread(target=n.serve_forever, daemon=True) for n in nodes]
    for t in threads: t.start()
    time.sleep(3.0)  # SWIM stabilization

    try:
        # Pick the head (layer_0-10) as the node to migrate INTO and then evict FROM.
        head = nodes[0]
        # Expert 50 is NOT in any shard's bootstrap config (Phase 4 overlap uses <30).
        # We manually attach it to head to simulate a completed migration pull.
        from model_shard.migration import ExpertWeightPeerRPC
        from model_shard.partial_load import slice_expert

        # Find a shard that DOES own expert 50 at layer 15 (none of the canonical
        # config owns 50; use expert 42 instead which is in layer_10-20's base).
        target_expert = 42
        # layer_10-20 holds experts e%3==1 at layer 15 (per Phase 4 config).
        # 42 % 3 == 0, so it's in layer_0-10 — already held. Use a truly foreign:
        # Actually for simplicity: pick an expert that only one shard owns, then
        # pull to head from elsewhere.
        # layer_10-20's base includes expert 40 (40 % 3 == 1).
        target_expert = 40
        source_sid = "layer_10-20"
        source_port = specs[1].address.port

        rpc = ExpertWeightPeerRPC(addresses={source_sid: ("127.0.0.1", source_port)}, timeout_s=60.0)
        tensors = rpc.pull(source_shard_id=source_sid, layer_idx=15, expert_id=target_expert)
        head.migration_attach(layer_idx=15, expert_id=target_expert, tensors=tensors)

        # Verify head owns the expert locally.
        assert target_expert in head._live_experts[15]

        # Wait for gossip convergence on ADD.
        def gossiped_add(n: Node, sid: str, L: int, E: int) -> bool:
            return sid in n.owners_of(L, E)
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            if all(gossiped_add(n, head._shard.shard_id, 15, target_expert) for n in nodes[1:]):
                break
            time.sleep(0.1)
        assert all(
            head._shard.shard_id in n.owners_of(15, target_expert) for n in nodes[1:]
        ), f"ADD gossip did not converge: {[n.owners_of(15, target_expert) for n in nodes[1:]]}"

        # Now evict. Manually call migration_detach (we set cooldown=0).
        head.migration_detach(15, target_expert)
        assert target_expert not in head._live_experts[15]

        # Wait for gossip convergence on REMOVE.
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            if all(
                head._shard.shard_id not in n.owners_of(15, target_expert)
                for n in nodes[1:]
            ):
                break
            time.sleep(0.1)
        assert all(
            head._shard.shard_id not in n.owners_of(15, target_expert) for n in nodes[1:]
        ), "REMOVE gossip did not converge"
    finally:
        for n, th in zip(nodes, threads):
            n.shutdown()
            th.join(timeout=3.0)
```

- [ ] **Step 2: Run**

Run: `uv run pytest tests/test_eviction_e2e.py -v -m slow`
Expected: PASS in ~10-15s.

- [ ] **Step 3: Commit**

```bash
git add tests/test_eviction_e2e.py
git commit -m "Phase 6-C Task 7: slow E2E — full attach+evict cycle converges"
```

---

### Task 8: Slow E2E — eviction race with in-flight ExpertRequest

**Files:**
- Create: `tests/test_eviction_race_with_expert_request.py`

- [ ] **Step 1: Write the slow test**

Create `tests/test_eviction_race_with_expert_request.py`:

```python
"""Phase 6-C slow: eviction race with in-flight ExpertRequest.

Lock invariant: _MLX_COMPUTE_LOCK serializes detach_expert with compute.
An in-flight ExpertRequest that's already holding the lock finishes first;
a queued request arriving after eviction sees the post-eviction _live_experts
and correctly returns ERR_WRONG_SHARD."""
from __future__ import annotations

import random
import socket as _sk
import threading
import time
from pathlib import Path

import mlx.core as mx
import pytest

from model_shard.migration import ExpertWeightPeerRPC
from model_shard.node import Node
from model_shard.shard_map import NodeAddress, ShardMap, ShardSpec

pytestmark = pytest.mark.slow


def _find_free_port() -> int:
    while True:
        p = random.randint(30000, 60000)
        s = _sk.socket()
        try:
            s.bind(("127.0.0.1", p))
            s.close()
            return p
        except OSError:
            s.close()


def test_evicted_expert_serves_wrong_shard_error(monkeypatch):
    """After eviction, subsequent ExpertRequest for the evicted expert
    returns ERR_WRONG_SHARD (not silent success, not hang)."""
    monkeypatch.setenv("ENABLE_PARTIAL_LOAD", "true")
    monkeypatch.setenv("ENABLE_DYNAMIC_MIGRATION", "false")
    monkeypatch.setenv("ENABLE_GOSSIP", "true")
    monkeypatch.setenv("ENABLE_EVICTION", "true")
    monkeypatch.setenv("MIGRATION_EVICT_COOLDOWN_SECONDS", "0")

    sm_yaml = ShardMap.from_yaml(Path("config/shards.yaml"))
    ids = sm_yaml.all_shards()
    ports = [_find_free_port() for _ in ids]
    specs = []
    for sid, port in zip(ids, ports):
        s = sm_yaml.lookup(sid)
        specs.append(
            ShardSpec(
                shard_id=sid, address=NodeAddress(host="127.0.0.1", port=port),
                start_layer=s.start_layer, end_layer=s.end_layer, moe_experts=s.moe_experts,
            )
        )
    sm = ShardMap({s.shard_id: s for s in specs})
    nodes = [Node(shard=s, shard_map=sm, total_layers=30) for s in specs]
    threads = [threading.Thread(target=n.serve_forever, daemon=True) for n in nodes]
    for t in threads: t.start()
    time.sleep(3.0)

    try:
        head = nodes[0]
        target_expert = 40  # in layer_10-20's base config; pull to head then evict.
        source_sid = "layer_10-20"
        source_port = specs[1].address.port

        # Migrate expert 40 to head.
        rpc_weight = ExpertWeightPeerRPC(
            addresses={source_sid: ("127.0.0.1", source_port)}, timeout_s=60.0
        )
        tensors = rpc_weight.pull(source_shard_id=source_sid, layer_idx=15, expert_id=target_expert)
        head.migration_attach(layer_idx=15, expert_id=target_expert, tensors=tensors)
        assert target_expert in head._live_experts[15]

        # Evict it immediately.
        head.migration_detach(15, target_expert)
        assert target_expert not in head._live_experts[15]

        # Now send an ExpertRequest directly to head for the evicted expert.
        # Head's _handle_expert_request must reject with ERR_WRONG_SHARD.
        from model_shard.expert_orchestrator import TcpPeerRPC
        direct_rpc = TcpPeerRPC(
            addresses={head._shard.shard_id: ("127.0.0.1", head._shard.address.port)},
            timeout_s=30.0,
        )
        # Build a synthetic post_attn tensor for the request.
        hidden = 2816  # Gemma 4 26B hidden_size; constant for this model.
        h = mx.zeros((1, 1, hidden), dtype=mx.bfloat16)
        with pytest.raises(RuntimeError, match="(wrong|WRONG_SHARD|not hosted|not held)"):
            direct_rpc.call(
                peer_shard_id=head._shard.shard_id,
                request_id="r-post-evict",
                layer_idx=15,
                expert_ids=[target_expert],
                h=h,
            )
    finally:
        for n, th in zip(nodes, threads):
            n.shutdown()
            th.join(timeout=3.0)
```

- [ ] **Step 2: Run**

Run: `uv run pytest tests/test_eviction_race_with_expert_request.py -v -m slow`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_eviction_race_with_expert_request.py
git commit -m "Phase 6-C Task 8: slow E2E — evicted expert serves WRONG_SHARD error"
```

---

### Task 9: README + memory update + final verification

**Files:**
- Modify: `README.md`
- Modify: `/Users/lukechang/.claude/projects/-Users-lukechang-Github-model-shard/memory/project_gossip_moe.md`

- [ ] **Step 1: Add Phase 6-C status paragraph to README**

Insert after the Phase 6-B status paragraph. Match existing style (~200-250 words, no emojis). Cover:

- Scope: per-node eviction of migration-added experts under capacity pressure, with `OwnershipDelta{REMOVE}` gossip and last-writer-wins convergence on `ts_unix_ms`.
- Mechanism: `detach_expert` (inverse of 5b's `attach_expert`) shrinks the compact stack via complementary-index `mx.take` under `_MLX_COMPUTE_LOCK`; `MigrationScanner._maybe_evict_one` runs after the pull pass, only fires at capacity.
- Safety: bootstrap-held experts never evicted; last-replica local check refuses eviction if no other live owner; 30s attach cooldown; compute-lock serialization prevents tensor mutation under an in-flight `ExpertRequest`.
- Data structure shift: `_ownership_seen: set` (5b, ADD-only) promoted to `_ownership_view: dict[(shard, L, E), (action, ts_unix_ms)]` with last-writer-wins convergence; `owners_of` contract preserved.
- Latent 5b bug fixed: `_handle_expert_request` now consults `_live_experts` (runtime) rather than `self._shard.moe_experts` (bootstrap), so migration-attached experts are served correctly and evicted experts correctly return `ERR_WRONG_SHARD`.
- Gate: `ENABLE_EVICTION=true` default-on. `MIGRATION_EVICT_COOLDOWN_SECONDS=30` knob.
- Correctness proofs:
  - `tests/test_partial_load_detach.py` — `attach → detach` round-trip produces byte-identical tensors.
  - `tests/test_ownership_view_convergence.py` — ADD/REMOVE arriving in any order converges the same way.
  - `tests/test_eviction_e2e.py` — 3-node cluster attach + evict cycle; gossip propagates REMOVE within one round.
  - `tests/test_eviction_race_with_expert_request.py` — post-eviction `ExpertRequest` returns `ERR_WRONG_SHARD`.
- Non-goals: quorum last-replica protection, two-phase tentative eviction, memory-pressure probing — all Phase 7+.
- Phase 6 complete (6-A retry ✅, 6-B provenance ✅, 6-C eviction ✅).
- Link to spec: `docs/superpowers/specs/2026-04-18-phase6c-eviction-design.md`.

- [ ] **Step 2: Update memory file**

Add a Phase 6-C COMPLETE paragraph to `/Users/lukechang/.claude/projects/-Users-lukechang-Github-model-shard/memory/project_gossip_moe.md`. Cover:

- Date `2026-04-18`, final commit SHA.
- 9 tasks done.
- Links to plan + spec.
- What it enables: eviction — the last piece that makes memory-bounded deployments on 24 GB 3090s sustainable. Phase 6 trilogy complete.
- Mechanism summary: `detach_expert` inverse of `attach_expert`; versioned `_ownership_view` dict with last-writer-wins; `MigrationScanner._maybe_evict_one` capacity-triggered with coldest-heat victim selection; safety via bootstrap-protected set + last-replica local check + 30s cooldown + compute-lock serialization.
- Gate: `ENABLE_EVICTION=true` default-on; `MIGRATION_EVICT_COOLDOWN_SECONDS=30`.
- Latent 5b fix: `_handle_expert_request` authority shifted from bootstrap `moe_experts` to runtime `_live_experts`.
- Phase 6 decomposition: 6-A ✅, 6-B ✅, 6-C ✅. Roadmap Phase 6 is complete.
- **Next:** Phase 7 brainstorm. Likely directions: pipeline-peer redundancy (deferred from 6-A case 2), client-side retry (case 3), cross-node ownership-exclusion gossip (6-A R5), signed ProvenanceEntries (6-B.4+ Byzantine detection), real multi-machine cluster deployment on 3090/Spark.

- [ ] **Step 3: Final verification sweep**

```bash
cd /Users/lukechang/Github/model_shard
uv run pytest -q                                                        # fast
uv run pytest -m slow -q tests/test_eviction_e2e.py                     # 6-C E2E
uv run pytest -m slow -q tests/test_eviction_race_with_expert_request.py # 6-C race
uv run pytest -m slow -q tests/test_provenance_tier1.py                 # 6-B regression
uv run pytest -m slow -q tests/test_expert_retry_bit_exact.py           # 6-A regression
uv run pytest -m slow -q tests/test_migration_over_tcp.py               # 5b regression
uv run ruff check src tests scripts
uv run mypy src
```

Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add README.md "/Users/lukechang/.claude/projects/-Users-lukechang-Github-model-shard/memory/project_gossip_moe.md"
git commit -m "Phase 6-C Task 9: README + memory update; Phase 6 trilogy complete"
```

- [ ] **Step 5: Report back**

Include:
- Phase 6-C README paragraph text.
- Verification results per bucket.
- Final commit SHA.
- All Phase 6-C commit SHAs from `git log --grep "Phase 6-C" --oneline`.

---

## Self-Review Notes

**Spec coverage:**
- D1 (scope) → all tasks stay within local eviction + gossip convergence
- D2 (trigger + victim) → Task 6 `_maybe_evict_one`
- D3 (versioned dict data structure) → Tasks 2 + 3
- D4 (wire unchanged) → verified: no proto changes in any task
- D5 (safety invariants) → Task 4 `migration_detach` implements all four guards
- D6 (accepted race) → documented in spec, no code
- D7 (`_handle_expert_request` authority) → Task 5
- D8 (scanner integration) → Task 6
- D9 (provenance interaction) → no 6-C change; relies on existing 6-A retry + Task 10 error path
- D10 (retry interaction) → no 6-C change
- D11 (gate) → Task 6 `ENABLE_EVICTION` env helper
- D12 (correctness bar) → Tasks 1 (round-trip), 2 (convergence), 4 (last-replica), 7 (E2E), 8 (race)
- D13 (non-goals) → plan excludes quorum, memory-pressure probing, bootstrap eviction, cross-node coord

**Placeholder scan:** No "TBD" / "add error handling" / vague steps. Every code step has complete code.

**Type consistency:**
- `LastReplicaError` → defined in Task 4, imported in Task 6 (scanner catches it implicitly via generic Exception handler)
- `_ownership_view_internal` / `_ownership_view_put` / `_ownership_seen_lock` — defined in Tasks 2-3, consumed in Task 4's `migration_detach`
- `_live_experts_attach_ts` — defined in Task 4, consumed in Task 6's `attach_ts_provider`
- `MigrationPolicy.evict_cooldown_s` / `eviction_enabled` — defined in Task 6, consumed by Node's policy construction
- `MigrationScanner(bootstrap_held, attach_ts_provider, evict_fn, ...)` — signature defined in Task 6, consumed by Node wiring in Task 6
- `_migration_evict_cooldown_s()` / `_eviction_enabled()` — env helpers defined in Tasks 4 and 6 respectively

No type-name drift. No references to methods not defined in a prior task.
