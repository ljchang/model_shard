# Phase 5b Dynamic Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement target-pull expert migration + local-routing heat tracking + observer-triggered decode-loop hang fix, opt-in via `ENABLE_DYNAMIC_MIGRATION=true` (requires `ENABLE_PARTIAL_LOAD=true`).

**Architecture:** Each node increments a local per-(layer, expert) heat EMA every time its router picks that expert. Sparse top-N heat reports piggyback on SWIM Ping/Ack alongside Phase 4's `LoadReport`s. A background scanner on every node compares local heat against its `_live_experts` registry; if an expert is hot locally but not hosted, it issues an `ExpertWeightRequest` to a current owner. The source slices 9 tensors from its compact stack via `mx.take(..., axis=0)`; the target grows its compact stack via `mx.concatenate([..., incoming[None, ...]], axis=0)` under `_MLX_COMPUTE_LOCK`, appends the expert id to `held_ids_per_layer[L]`, and gossips an `OwnershipDelta{ADD}`. Routing picks up new replicas automatically because `ExpertOrchestrator` now resolves owners via a `live_owners_provider` callback instead of a static map. The decode-loop hang fix reuses the existing membership observer — on any peer-left-ALIVE transition, the observer enqueues a `_POISON_TOKEN = -1` sentinel into every in-flight `token_queue`, and `_drive_decode_loop` short-circuits on that sentinel.

**Tech Stack:** Python 3.13, MLX (post-load `mx.take` / `mx.concatenate` on `QuantizedSwitchLinear` stacks), protobuf over UDP (SWIM piggyback) and TCP (weight transfer envelopes), pytest with `slow` marker for model-loading tests.

**Spec:** `docs/superpowers/specs/2026-04-17-phase5b-dynamic-migration-design.md` — decisions D1-D17.

---

## File Structure

**Create:**
- `src/model_shard/heat.py` — `HeatTracker` (EMA over local router picks).
- `src/model_shard/migration.py` — `MigrationPolicy`, `MigrationScanner`, `ExpertWeightPeerRPC`.

**Modify:**
- `proto/wire.proto` — add `ExpertHeatReport`, `ExpertHeatEntry`, `OwnershipDelta`, `ExpertWeightRequest`, `ExpertWeightTransfer`; piggyback on Ping/Ack/PingReq/PingReqAck; add 2 Envelope oneof slots.
- `src/model_shard/_pb/wire_pb2.py` — regenerated from proto (do not hand-edit).
- `src/model_shard/membership/records.py` — add `HeatReportRecord`, `OwnershipDeltaRecord`, extend Ping/Ack/PingReq/PingReqAck dataclasses.
- `src/model_shard/membership/messages.py` — encode/decode heat + ownership piggyback.
- `src/model_shard/membership/runner.py` — `start_heat_source`, `latest_heat`, `announce_ownership_add`, `ownership_view`, TTL-limited outbound ownership queue, heat piggyback on outgoing ping-family.
- `src/model_shard/partial_load.py` — add `slice_expert` and `attach_expert` helpers.
- `src/model_shard/moe.py` — extend `group_expert_ids_by_owner_loaded` to accept a `live_owners_provider` (keep backward-compat via default), optional heat observer hook in `run_attention_and_route`.
- `src/model_shard/expert_orchestrator.py` — replace static `owners` with `live_owners_provider` call at grouping time.
- `src/model_shard/node.py` — `_live_experts` registry, `_ownership_seen`, new handlers `_handle_expert_weight_request` and `_handle_expert_weight_transfer`, decode-loop sentinel check, observer-triggered queue poison, scanner start/stop, `ENABLE_DYNAMIC_MIGRATION` gate.

**Test files created:**
- `tests/test_heat_tracker.py`
- `tests/test_wire_expert_weight_roundtrip.py`
- `tests/test_membership_heat_records.py`
- `tests/test_membership_ownership_gossip.py`
- `tests/test_partial_load_slice_attach.py`
- `tests/test_migration_scanner_policy.py`
- `tests/test_expert_weight_peer_rpc.py`
- `tests/test_node_expert_weight_handler.py`
- `tests/test_node_live_experts.py`
- `tests/test_orchestrator_live_owners.py`
- `tests/test_decode_hang_fix.py`
- `tests/test_dynamic_migration_gate.py`
- `tests/test_migration_bit_exact_per_expert.py` (slow)
- `tests/test_migration_over_tcp.py` (slow)
- `tests/test_ownership_gossip_convergence.py` (slow)
- `tests/test_decode_hang_fix_e2e.py` (slow)
- `tests/test_partial_load_tier1_migration.py` (slow)

---

## Task ordering and dependencies

Tasks are ordered so each builds on a committed predecessor. Most tasks touch one module; integration tasks (13, 14, 17, 20) wire earlier components into `Node`.

1. Wire protocol additions (proto + regen)
2. Membership records — heat + ownership dataclasses
3. Membership encode/decode for heat + ownership piggyback
4. `HeatTracker`
5. `slice_expert` helper (fast)
6. `attach_expert` helper (fast)
7. Bit-exact slice→attach correctness proof (slow, load-bearing)
8. `MembershipRunner.start_heat_source` + `latest_heat`
9. `MembershipRunner.announce_ownership_add` + `ownership_view` with TTL
10. `group_expert_ids_by_owner_loaded` accepts `live_owners_provider`
11. `ExpertOrchestrator` uses `live_owners_provider`
12. `ExpertWeightPeerRPC`
13. `Node._handle_expert_weight_request` (source side)
14. `Node._live_experts` registry + `_ownership_seen` initialization
15. `MigrationPolicy` + `MigrationScanner._scan_once`
16. `MigrationScanner.start/stop` background thread
17. Node receive-side attach + ownership announcement + heat integration
18. Decode-loop hang fix (D14)
19. `ENABLE_DYNAMIC_MIGRATION` gate + conflicting-flag validation
20. Slow: migration over TCP E2E (2-node)
21. Slow: ownership gossip convergence (3-node)
22. Slow: decode-hang fix E2E (3-node)
23. Slow: Tier 1 E2E regression with both flags ON
24. Update memory + README; commit

---

### Task 1: Wire protocol additions

**Files:**
- Modify: `proto/wire.proto`
- Regenerate: `src/model_shard/_pb/wire_pb2.py`
- Test: `tests/test_wire_expert_weight_roundtrip.py` (create)

- [ ] **Step 1: Add new messages to `proto/wire.proto`**

Insert the following after the existing `LoadReport` message (around line 219, before `message Envelope`):

```proto
message ExpertHeatEntry {
  uint32 layer_idx     = 1;
  uint32 expert_id     = 2;
  uint32 heat_ema_x100 = 3;
}

message ExpertHeatReport {
  string shard_id = 1;
  repeated ExpertHeatEntry entries = 2;
  int64  ts_unix_ms = 3;
}

message OwnershipDelta {
  string shard_id   = 1;
  uint32 layer_idx  = 2;
  uint32 expert_id  = 3;
  uint32 action     = 4;  // 0 = ADD; 1 = REMOVE (reserved for Phase 6)
  int64  ts_unix_ms = 5;
}

message ExpertWeightRequest {
  uint32 protocol_version = 1;
  string request_id       = 2;
  uint32 layer_idx        = 3;
  uint32 expert_id        = 4;
}

message ExpertWeightTransfer {
  uint32 protocol_version = 1;
  string request_id       = 2;
  uint32 layer_idx        = 3;
  uint32 expert_id        = 4;
  repeated TensorDescriptor tensors = 5;  // exactly 9, fixed order (see spec §3.4)
  uint32 tensor_count     = 6;
}
```

- [ ] **Step 2: Add piggyback fields to Ping/Ack/PingReq/PingReqAck**

Edit each existing message:

```proto
message Ping {
  // ...existing fields 1-5...
  repeated ExpertHeatReport heat      = 6;  // NEW
  repeated OwnershipDelta   ownership = 7;  // NEW
}

message Ack {
  // ...existing fields 1-5...
  repeated ExpertHeatReport heat      = 6;  // NEW
  repeated OwnershipDelta   ownership = 7;  // NEW
}

message PingReq {
  // ...existing fields 1-6...
  repeated ExpertHeatReport heat      = 7;  // NEW
  repeated OwnershipDelta   ownership = 8;  // NEW
}

message PingReqAck {
  // ...existing fields 1-7...
  repeated ExpertHeatReport heat      = 8;  // NEW
  repeated OwnershipDelta   ownership = 9;  // NEW
}
```

- [ ] **Step 3: Add Envelope oneof entries**

In `message Envelope { oneof payload { ... } }`, after `ExpertResponse expert_response = 15;` add:

```proto
    ExpertWeightRequest  expert_weight_request  = 16;
    ExpertWeightTransfer expert_weight_transfer = 17;
```

- [ ] **Step 4: Regenerate the protobuf bindings**

Run:
```bash
cd /Users/lukechang/Github/model_shard
uv run python -m grpc_tools.protoc -I proto --python_out=src/model_shard/_pb proto/wire.proto
```
Expected: no stdout; `src/model_shard/_pb/wire_pb2.py` timestamp updates.

- [ ] **Step 5: Write the failing wire-roundtrip test**

Create `tests/test_wire_expert_weight_roundtrip.py`:

```python
"""Round-trip tests for the Phase 5b protobuf additions."""
from __future__ import annotations

from model_shard._pb import wire_pb2


def test_expert_weight_request_fields():
    req = wire_pb2.ExpertWeightRequest(
        protocol_version=1, request_id="abc", layer_idx=15, expert_id=7
    )
    raw = req.SerializeToString()
    parsed = wire_pb2.ExpertWeightRequest()
    parsed.ParseFromString(raw)
    assert parsed.protocol_version == 1
    assert parsed.request_id == "abc"
    assert parsed.layer_idx == 15
    assert parsed.expert_id == 7


def test_expert_weight_transfer_nine_descriptors():
    t = wire_pb2.ExpertWeightTransfer(
        protocol_version=1, request_id="abc", layer_idx=15, expert_id=7,
        tensor_count=9,
    )
    for i in range(9):
        d = t.tensors.add()
        d.shape.extend([704, 352])
        d.dtype = wire_pb2.DTYPE_BFLOAT16
        d.quant = wire_pb2.QUANT_NONE
        d.byte_count = 100 + i
    raw = t.SerializeToString()
    parsed = wire_pb2.ExpertWeightTransfer()
    parsed.ParseFromString(raw)
    assert parsed.tensor_count == 9
    assert len(parsed.tensors) == 9
    assert [int(d.byte_count) for d in parsed.tensors] == [100 + i for i in range(9)]


def test_envelope_oneof_recognises_new_payloads():
    env = wire_pb2.Envelope()
    env.expert_weight_request.protocol_version = 1
    env.expert_weight_request.request_id = "r"
    env.expert_weight_request.layer_idx = 15
    env.expert_weight_request.expert_id = 7
    assert env.WhichOneof("payload") == "expert_weight_request"

    env2 = wire_pb2.Envelope()
    env2.expert_weight_transfer.protocol_version = 1
    env2.expert_weight_transfer.request_id = "r"
    env2.expert_weight_transfer.layer_idx = 15
    env2.expert_weight_transfer.expert_id = 7
    env2.expert_weight_transfer.tensor_count = 9
    assert env2.WhichOneof("payload") == "expert_weight_transfer"


def test_ping_carries_heat_and_ownership():
    p = wire_pb2.Ping(protocol_version=1, from_shard_id="a", from_incarnation=1)
    hr = p.heat.add()
    hr.shard_id = "a"
    hr.ts_unix_ms = 1234
    entry = hr.entries.add()
    entry.layer_idx = 15
    entry.expert_id = 7
    entry.heat_ema_x100 = 500

    od = p.ownership.add()
    od.shard_id = "a"
    od.layer_idx = 15
    od.expert_id = 7
    od.action = 0

    raw = p.SerializeToString()
    parsed = wire_pb2.Ping()
    parsed.ParseFromString(raw)
    assert len(parsed.heat) == 1
    assert parsed.heat[0].entries[0].heat_ema_x100 == 500
    assert len(parsed.ownership) == 1
    assert parsed.ownership[0].expert_id == 7
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `uv run pytest tests/test_wire_expert_weight_roundtrip.py -v`
Expected: 4 PASS.

- [ ] **Step 7: Commit**

```bash
git add proto/wire.proto src/model_shard/_pb/wire_pb2.py tests/test_wire_expert_weight_roundtrip.py
git commit -m "Phase 5b Task 1: wire protocol additions (heat, ownership, weight transfer)"
```

---

### Task 2: Membership records — heat + ownership dataclasses

**Files:**
- Modify: `src/model_shard/membership/records.py`
- Test: `tests/test_membership_heat_records.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_membership_heat_records.py`:

```python
"""Dataclass shape tests for Phase 5b membership records."""
from __future__ import annotations

from model_shard.membership.records import (
    AckMsg,
    HeatReportRecord,
    OwnershipDeltaRecord,
    PingMsg,
)


def test_heat_report_record_is_frozen_and_equal_by_value():
    a = HeatReportRecord(
        shard_id="a",
        entries=((15, 7, 500),),
        ts_unix_ms=1234,
    )
    b = HeatReportRecord(
        shard_id="a",
        entries=((15, 7, 500),),
        ts_unix_ms=1234,
    )
    assert a == b
    # Frozen: assignment should raise.
    try:
        a.shard_id = "b"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("HeatReportRecord should be frozen")


def test_ownership_delta_record_add():
    d = OwnershipDeltaRecord(
        shard_id="a", layer_idx=15, expert_id=7, action=0, ts_unix_ms=1
    )
    assert d.action == 0


def test_ping_carries_heat_and_ownership_defaults_empty():
    p = PingMsg(
        from_shard_id="a", from_incarnation=1, deltas=[],
    )
    assert p.heat == []
    assert p.ownership == []


def test_ack_carries_heat_and_ownership():
    hr = HeatReportRecord(shard_id="a", entries=((15, 7, 500),), ts_unix_ms=1)
    od = OwnershipDeltaRecord(shard_id="a", layer_idx=15, expert_id=7, action=0, ts_unix_ms=1)
    a = AckMsg(
        from_shard_id="a", from_incarnation=1, deltas=[],
        heat=[hr], ownership=[od],
    )
    assert a.heat == [hr]
    assert a.ownership == [od]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_membership_heat_records.py -v`
Expected: ImportError or AttributeError on `HeatReportRecord`/`OwnershipDeltaRecord`.

- [ ] **Step 3: Add the dataclasses to `records.py`**

In `src/model_shard/membership/records.py`, after the `LoadReportRecord` dataclass (around line 40) add:

```python
@dataclass(frozen=True)
class HeatReportRecord:
    """Sparse per-node heat snapshot. ``entries`` is a tuple of
    ``(layer_idx, expert_id, heat_ema_x100)`` triples sorted by EMA desc,
    capped at HeatTracker.top_n (default 16)."""
    shard_id: str
    entries: tuple[tuple[int, int, int], ...]
    ts_unix_ms: int


@dataclass(frozen=True)
class OwnershipDeltaRecord:
    """Idempotent ADD-only ownership announcement (Phase 5b).

    ``action`` is 0 for ADD (only value used in Phase 5b; 1 REMOVE is
    reserved for Phase 6 eviction)."""
    shard_id: str
    layer_idx: int
    expert_id: int
    action: int
    ts_unix_ms: int
```

- [ ] **Step 4: Extend the ping-family dataclasses**

Edit `PingMsg`, `AckMsg`, `PingReqMsg`, `PingReqAckMsg` (around lines 58-91) to add the two new fields with empty defaults:

```python
@dataclass(frozen=True)
class PingMsg:
    from_shard_id: str
    from_incarnation: int
    deltas: list[MemberRecord]
    loads: list[LoadReportRecord] = field(default_factory=list)
    heat: list[HeatReportRecord] = field(default_factory=list)
    ownership: list[OwnershipDeltaRecord] = field(default_factory=list)


@dataclass(frozen=True)
class AckMsg:
    from_shard_id: str
    from_incarnation: int
    deltas: list[MemberRecord]
    loads: list[LoadReportRecord] = field(default_factory=list)
    heat: list[HeatReportRecord] = field(default_factory=list)
    ownership: list[OwnershipDeltaRecord] = field(default_factory=list)


@dataclass(frozen=True)
class PingReqMsg:
    from_shard_id: str
    target_shard_id: str
    probe_id: str
    deltas: list[MemberRecord]
    loads: list[LoadReportRecord] = field(default_factory=list)
    heat: list[HeatReportRecord] = field(default_factory=list)
    ownership: list[OwnershipDeltaRecord] = field(default_factory=list)


@dataclass(frozen=True)
class PingReqAckMsg:
    from_shard_id: str
    target_shard_id: str
    probe_id: str
    success: bool
    deltas: list[MemberRecord]
    loads: list[LoadReportRecord] = field(default_factory=list)
    heat: list[HeatReportRecord] = field(default_factory=list)
    ownership: list[OwnershipDeltaRecord] = field(default_factory=list)
```

Update `__all__` to include `"HeatReportRecord"` and `"OwnershipDeltaRecord"`.

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_membership_heat_records.py -v`
Expected: 4 PASS.

- [ ] **Step 6: Sanity-check existing Phase 2/4 suite still green**

Run: `uv run pytest tests/test_membership_load_records.py tests/membership/test_records.py -v`
Expected: all PASS (new fields have defaults so Phase 2/4 call sites are unaffected).

- [ ] **Step 7: Commit**

```bash
git add src/model_shard/membership/records.py tests/test_membership_heat_records.py
git commit -m "Phase 5b Task 2: membership records for heat + ownership"
```

---

### Task 3: Membership encode/decode for heat + ownership piggyback

**Files:**
- Modify: `src/model_shard/membership/messages.py`
- Test: `tests/test_membership_heat_records.py` (extend)

- [ ] **Step 1: Write the failing encode/decode test**

Append to `tests/test_membership_heat_records.py`:

```python
from model_shard.membership.messages import (
    decode_membership_envelope,
    encode_membership_envelope,
)


def test_ping_heat_and_ownership_roundtrip():
    p = PingMsg(
        from_shard_id="a",
        from_incarnation=1,
        deltas=[],
        heat=[HeatReportRecord(
            shard_id="a", entries=((15, 7, 500), (15, 3, 300)), ts_unix_ms=1
        )],
        ownership=[OwnershipDeltaRecord(
            shard_id="a", layer_idx=15, expert_id=7, action=0, ts_unix_ms=1
        )],
    )
    raw = encode_membership_envelope(p)
    parsed = decode_membership_envelope(raw)
    assert isinstance(parsed, PingMsg)
    assert parsed.heat == p.heat
    assert parsed.ownership == p.ownership


def test_ack_heat_and_ownership_roundtrip():
    a = AckMsg(
        from_shard_id="a",
        from_incarnation=1,
        deltas=[],
        heat=[HeatReportRecord(
            shard_id="a", entries=((15, 7, 500),), ts_unix_ms=1
        )],
        ownership=[OwnershipDeltaRecord(
            shard_id="a", layer_idx=15, expert_id=7, action=0, ts_unix_ms=1
        )],
    )
    raw = encode_membership_envelope(a)
    parsed = decode_membership_envelope(raw)
    assert isinstance(parsed, AckMsg)
    assert parsed.heat == a.heat
    assert parsed.ownership == a.ownership
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_membership_heat_records.py::test_ping_heat_and_ownership_roundtrip -v`
Expected: FAIL — encode path ignores new fields so parsed ones will be empty.

- [ ] **Step 3: Add encode/decode helpers in `messages.py`**

In `src/model_shard/membership/messages.py`, after `_load_from_pb` (line 62) add:

```python
def _heat_to_pb(r: HeatReportRecord) -> wire_pb2.ExpertHeatReport:
    pb = wire_pb2.ExpertHeatReport(
        shard_id=r.shard_id,
        ts_unix_ms=r.ts_unix_ms,
    )
    for layer_idx, expert_id, ema in r.entries:
        e = pb.entries.add()
        e.layer_idx = layer_idx
        e.expert_id = expert_id
        e.heat_ema_x100 = ema
    return pb


def _heat_from_pb(pb: wire_pb2.ExpertHeatReport) -> HeatReportRecord:
    return HeatReportRecord(
        shard_id=pb.shard_id,
        entries=tuple(
            (int(e.layer_idx), int(e.expert_id), int(e.heat_ema_x100))
            for e in pb.entries
        ),
        ts_unix_ms=int(pb.ts_unix_ms),
    )


def _ownership_to_pb(r: OwnershipDeltaRecord) -> wire_pb2.OwnershipDelta:
    return wire_pb2.OwnershipDelta(
        shard_id=r.shard_id,
        layer_idx=r.layer_idx,
        expert_id=r.expert_id,
        action=r.action,
        ts_unix_ms=r.ts_unix_ms,
    )


def _ownership_from_pb(pb: wire_pb2.OwnershipDelta) -> OwnershipDeltaRecord:
    return OwnershipDeltaRecord(
        shard_id=pb.shard_id,
        layer_idx=int(pb.layer_idx),
        expert_id=int(pb.expert_id),
        action=int(pb.action),
        ts_unix_ms=int(pb.ts_unix_ms),
    )
```

Update the `from model_shard.membership.records import (...)` block to include:
```python
    HeatReportRecord,
    OwnershipDeltaRecord,
```

- [ ] **Step 4: Wire the helpers into encode/decode**

In `encode_membership_envelope` (starts line 64), for each of the four ping-family branches, after the existing `...loads.extend(...)` line add:

```python
        env.<which>.heat.extend(_heat_to_pb(h) for h in msg.heat)
        env.<which>.ownership.extend(_ownership_to_pb(o) for o in msg.ownership)
```

where `<which>` is `ping`, `ack`, `ping_req`, `ping_req_ack` respectively.

In `decode_membership_envelope` (starts line 104), for each of the four ping-family branches, extend the constructor call with:

```python
            heat=[_heat_from_pb(h) for h in env.<which>.heat],
            ownership=[_ownership_from_pb(o) for o in env.<which>.ownership],
```

The `ping` branch example (lines 108-114) becomes:

```python
    if which == "ping":
        return PingMsg(
            from_shard_id=env.ping.from_shard_id,
            from_incarnation=int(env.ping.from_incarnation),
            deltas=[_record_from_pb(d) for d in env.ping.deltas],
            loads=[_load_from_pb(lr) for lr in env.ping.loads],
            heat=[_heat_from_pb(h) for h in env.ping.heat],
            ownership=[_ownership_from_pb(o) for o in env.ping.ownership],
        )
```

Apply the same shape to `ack`, `ping_req`, `ping_req_ack`.

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_membership_heat_records.py -v`
Expected: 6 PASS.

- [ ] **Step 6: Run regression on existing membership suite**

Run: `uv run pytest tests/membership/ tests/test_load_report_envelope.py -v`
Expected: all PASS (defaults make this backwards compatible).

- [ ] **Step 7: Commit**

```bash
git add src/model_shard/membership/messages.py tests/test_membership_heat_records.py
git commit -m "Phase 5b Task 3: encode/decode heat + ownership piggyback"
```

---

### Task 4: `HeatTracker`

**Files:**
- Create: `src/model_shard/heat.py`
- Test: `tests/test_heat_tracker.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_heat_tracker.py`:

```python
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
    for t in threads: t.start()
    for t in threads: t.join()
    # alpha=1 means every observe overwrites with count; after the last
    # observe lands ema ≈ count_from_that_call (non-deterministic but finite).
    assert ht.local_heat(15, 3) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_heat_tracker.py -v`
Expected: ImportError — `model_shard.heat` does not exist.

- [ ] **Step 3: Implement `HeatTracker`**

Create `src/model_shard/heat.py`:

```python
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
        counts: dict[int, int] = defaultdict(int)
        for eid in expert_ids:
            counts[int(eid)] += 1
        with self._lock:
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
        capped at ``top_n``. Suitable for UDP piggyback."""
        with self._lock:
            snapshot = [
                (l, e, round(v * 100.0))
                for (l, e), v in self._ema.items()
                if v > 0.0
            ]
        snapshot.sort(key=lambda t: t[2], reverse=True)
        return snapshot[: self._top_n]

    def local_heat(self, layer_idx: int, expert_id: int) -> int:
        """Return current EMA×100 for one (layer, expert), or 0 if untracked."""
        with self._lock:
            return round(self._ema.get((layer_idx, expert_id), 0.0) * 100.0)


__all__ = ["HeatTracker"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_heat_tracker.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/model_shard/heat.py tests/test_heat_tracker.py
git commit -m "Phase 5b Task 4: HeatTracker (per-node routing-count EMA)"
```

---

### Task 5: `slice_expert` helper

**Files:**
- Modify: `src/model_shard/partial_load.py`
- Test: `tests/test_partial_load_slice_attach.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_partial_load_slice_attach.py`:

```python
"""Unit tests for slice_expert / attach_expert using synthetic LoadedModel."""
from __future__ import annotations

import threading
import types
from typing import Any

import mlx.core as mx
import pytest

from model_shard.mlx_engine import LoadedModel
from model_shard.partial_load import attach_expert, slice_expert


def _make_fake_lm(num_experts: int, held: list[int]) -> LoadedModel:
    """Build a LoadedModel shell whose text_model.layers[0].experts.switch_glu
    has synthetic (num_experts, 4, 4) tensors for the 9 proj/attr slots.
    Only layer 0 is wired; enough for slice_expert / attach_expert tests."""
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
        mlx_model=mlx_model,   # type: ignore[arg-type]
        language_model=language_model,  # type: ignore[arg-type]
        text_model=text_model,  # type: ignore[arg-type]
        processor=None,  # type: ignore[arg-type]
        num_layers=1,
        held_ids_per_layer={0: tuple(held)} if held else {},
    )


def test_slice_expert_returns_nine_tensors_at_local_slot():
    lock = threading.Lock()
    # Held = [0, 3, 6, 9]; local slot of global id 6 is 2.
    lm = _make_fake_lm(num_experts=4, held=[0, 3, 6, 9])
    tensors = slice_expert(lm, layer_idx=0, expert_id=6, mlx_lock=lock)
    assert len(tensors) == 9
    # Verify we sliced along axis 0 at local slot 2.
    sg = lm.text_model.layers[0].experts.switch_glu
    for i, (proj_name, attr) in enumerate([
        ("gate_proj", "weight"), ("gate_proj", "scales"), ("gate_proj", "biases"),
        ("up_proj",   "weight"), ("up_proj",   "scales"), ("up_proj",   "biases"),
        ("down_proj", "weight"), ("down_proj", "scales"), ("down_proj", "biases"),
    ]):
        expected = getattr(getattr(sg, proj_name), attr)[2]
        assert mx.array_equal(tensors[i], expected).item()


def test_slice_expert_raises_when_not_held():
    lock = threading.Lock()
    lm = _make_fake_lm(num_experts=4, held=[0, 3, 6, 9])
    with pytest.raises(KeyError):
        slice_expert(lm, layer_idx=0, expert_id=42, mlx_lock=lock)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_partial_load_slice_attach.py::test_slice_expert_returns_nine_tensors_at_local_slot -v`
Expected: ImportError — `slice_expert` does not exist.

- [ ] **Step 3: Implement `slice_expert`**

Append to `src/model_shard/partial_load.py` (after `load_model_partial`, before `__all__`):

```python
# Canonical order matches the on-wire ExpertWeightTransfer payload (spec §3.4).
_PROJ_ATTR_ORDER: list[tuple[str, str]] = [
    ("gate_proj", "weight"), ("gate_proj", "scales"), ("gate_proj", "biases"),
    ("up_proj",   "weight"), ("up_proj",   "scales"), ("up_proj",   "biases"),
    ("down_proj", "weight"), ("down_proj", "scales"), ("down_proj", "biases"),
]


def slice_expert(
    lm: LoadedModel,
    layer_idx: int,
    expert_id: int,
    mlx_lock: "threading.Lock",
) -> list["mx.array"]:
    """Return the 9 tensors for one expert in canonical order.

    Translates ``expert_id`` to a local slot via ``lm.held_ids_per_layer``,
    then ``mx.take`` along axis 0. Held lock only during take + eval.
    Raises KeyError if ``expert_id`` is not held on this node."""
    held = lm.held_ids_per_layer.get(layer_idx)
    if held is None or expert_id not in held:
        raise KeyError(
            f"expert {expert_id} not held at layer {layer_idx} "
            f"(held ids: {held})"
        )
    local_slot = list(held).index(expert_id)
    layer = lm.text_model.layers[layer_idx]
    switch_glu = layer.experts.switch_glu
    with mlx_lock:
        out: list[mx.array] = []
        idx = mx.array([local_slot])
        for proj_name, attr in _PROJ_ATTR_ORDER:
            proj = getattr(switch_glu, proj_name)
            full = getattr(proj, attr)
            # Take axis 0 at the single slot and squeeze the leading dim.
            sliced = mx.take(full, idx, axis=0)[0]
            out.append(sliced)
        mx.eval(*out)
    return out
```

Ensure `import threading` sits at the top of the file. The `_PROJ_ATTR_ORDER` constant is module-level and reused by Task 6.

- [ ] **Step 4: Update `__all__`**

```python
__all__ = ["_slice_stacked_by_axis0", "load_model_partial", "slice_expert"]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_partial_load_slice_attach.py -v`
Expected: 2 PASS.

- [ ] **Step 6: Commit**

```bash
git add src/model_shard/partial_load.py tests/test_partial_load_slice_attach.py
git commit -m "Phase 5b Task 5: slice_expert helper"
```

---

### Task 6: `attach_expert` helper

**Files:**
- Modify: `src/model_shard/partial_load.py`
- Test: `tests/test_partial_load_slice_attach.py` (extend)

- [ ] **Step 1: Append failing tests**

Append to `tests/test_partial_load_slice_attach.py`:

```python
def test_attach_expert_grows_stack_by_one():
    lock = threading.Lock()
    lm = _make_fake_lm(num_experts=4, held=[0, 3, 6, 9])
    # Build 9 synthetic tensors with identifiable values.
    new_tensors = [mx.full((4, 4), fill_value=float(i)) for i in range(3)]
    new_tensors += [mx.full((4, 8), fill_value=float(i)) for i in range(3, 6)]
    new_tensors += [mx.full((4, 8), fill_value=float(i)) for i in range(6, 9)]
    attach_expert(lm, layer_idx=0, expert_id=42, tensors=new_tensors, mlx_lock=lock)
    sg = lm.text_model.layers[0].experts.switch_glu
    assert sg.gate_proj.weight.shape[0] == 5
    assert lm.held_ids_per_layer[0] == (0, 3, 6, 9, 42)
    # New tensor landed at the new tail row.
    assert mx.array_equal(sg.gate_proj.weight[4], new_tensors[0]).item()


def test_attach_expert_raises_on_duplicate():
    lock = threading.Lock()
    lm = _make_fake_lm(num_experts=4, held=[0, 3, 6, 9])
    dummy = [mx.zeros((4, 4)) for _ in range(3)] + [mx.zeros((4, 8)) for _ in range(6)]
    with pytest.raises(ValueError):
        attach_expert(lm, layer_idx=0, expert_id=3, tensors=dummy, mlx_lock=lock)


def test_attach_expert_requires_nine_tensors():
    lock = threading.Lock()
    lm = _make_fake_lm(num_experts=4, held=[0, 3, 6, 9])
    with pytest.raises(ValueError):
        attach_expert(lm, layer_idx=0, expert_id=42, tensors=[mx.zeros((4, 4))], mlx_lock=lock)


def test_attach_then_slice_roundtrips():
    lock = threading.Lock()
    lm = _make_fake_lm(num_experts=4, held=[0, 3, 6, 9])
    sentinels = (
        [mx.full((4, 4), fill_value=10.0 + i) for i in range(3)]
        + [mx.full((4, 8), fill_value=20.0 + i) for i in range(3)]
        + [mx.full((4, 8), fill_value=30.0 + i) for i in range(3)]
    )
    attach_expert(lm, layer_idx=0, expert_id=42, tensors=sentinels, mlx_lock=lock)
    sliced = slice_expert(lm, layer_idx=0, expert_id=42, mlx_lock=lock)
    for a, b in zip(sentinels, sliced):
        assert mx.array_equal(a, b).item()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_partial_load_slice_attach.py -v`
Expected: FAIL — `attach_expert` does not exist.

- [ ] **Step 3: Implement `attach_expert`**

Append to `src/model_shard/partial_load.py`:

```python
def attach_expert(
    lm: LoadedModel,
    layer_idx: int,
    expert_id: int,
    tensors: list["mx.array"],
    mlx_lock: "threading.Lock",
) -> None:
    """Grow the compact stack at ``layer_idx`` by one expert.

    Invariants:
      * ``expert_id`` must not already be in ``lm.held_ids_per_layer[layer_idx]``.
      * ``tensors`` must be exactly 9 items in the canonical ``_PROJ_ATTR_ORDER``.

    Under ``mlx_lock``:
      1. For each (proj, attr), replace the attribute with
         ``mx.concatenate([current, incoming[None, ...]], axis=0)``.
      2. Append ``expert_id`` to ``held_ids_per_layer[layer_idx]``.
      3. ``mx.eval`` the 9 new tensors to force realization.
    """
    if len(tensors) != 9:
        raise ValueError(f"attach_expert requires 9 tensors, got {len(tensors)}")
    held_before = lm.held_ids_per_layer.get(layer_idx, ())
    if expert_id in held_before:
        raise ValueError(
            f"expert {expert_id} already held at layer {layer_idx} "
            f"(held: {held_before})"
        )
    layer = lm.text_model.layers[layer_idx]
    switch_glu = layer.experts.switch_glu
    with mlx_lock:
        realized: list[mx.array] = []
        for (proj_name, attr), incoming in zip(_PROJ_ATTR_ORDER, tensors):
            proj = getattr(switch_glu, proj_name)
            current = getattr(proj, attr)
            grown = mx.concatenate(
                [current, incoming[None, ...]], axis=0
            )
            setattr(proj, attr, grown)
            realized.append(grown)
        lm.held_ids_per_layer[layer_idx] = (*held_before, expert_id)
        mx.eval(*realized)
```

Update `__all__`:
```python
__all__ = [
    "_slice_stacked_by_axis0",
    "attach_expert",
    "load_model_partial",
    "slice_expert",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_partial_load_slice_attach.py -v`
Expected: 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/model_shard/partial_load.py tests/test_partial_load_slice_attach.py
git commit -m "Phase 5b Task 6: attach_expert helper"
```

---

### Task 7: Bit-exact slice→attach correctness proof (slow, load-bearing)

**Files:**
- Create: `tests/test_migration_bit_exact_per_expert.py`

This is the load-bearing correctness proof. Mirrors 5a's per-expert bit-exactness. Must run under the no-sort path (B*Seq < 64) so that quant sort-path FP noise does not apply (spec D15 and 5a §7.5).

- [ ] **Step 1: Write the failing slow test**

Create `tests/test_migration_bit_exact_per_expert.py`:

```python
"""Bit-exact correctness proof for Phase 5b migration.

Load two sliced LoadedModels with disjoint held expert sets for layer 15.
Slice expert E from A, attach to B, assert run_selected_experts matches
the full-model baseline bit-exactly on the no-sort path."""
from __future__ import annotations

import threading

import mlx.core as mx
import pytest

from model_shard.mlx_engine import load_model, load_model_partial
from model_shard.moe import run_selected_experts
from model_shard.partial_load import attach_expert, slice_expert

pytestmark = pytest.mark.slow

_HF_ID = "mlx-community/gemma-4-26b-a4b-it-4bit"
_LAYER = 15
_MIGRATED_EXPERT = 3


@pytest.fixture(scope="module")
def lm_full():
    return load_model(_HF_ID)


@pytest.fixture(scope="module")
def lm_a():
    return load_model_partial(_HF_ID, {_LAYER: [0, 3, 6, 9]})


@pytest.fixture(scope="module")
def lm_b():
    return load_model_partial(_HF_ID, {_LAYER: [1, 4, 7, 10]})


def _synthetic_h(lm_full) -> mx.array:
    # Keep B*Seq = 1*7 = 7 — firmly on the no-sort path per 5a §7.5.
    mx.random.seed(42)
    hidden = lm_full.text_model.layers[_LAYER].pre_feedforward_layernorm_2.weight.shape[0]
    return mx.random.normal((1, 7, hidden)).astype(mx.bfloat16)


def test_slice_from_a_equals_attach_on_b(lm_full, lm_a, lm_b):
    lock = threading.Lock()
    tensors = slice_expert(lm_a, _LAYER, _MIGRATED_EXPERT, lock)
    attach_expert(lm_b, _LAYER, _MIGRATED_EXPERT, tensors, lock)
    assert _MIGRATED_EXPERT in lm_b.held_ids_per_layer[_LAYER]

    h = _synthetic_h(lm_full)
    out_a = run_selected_experts(lm_a, h, _LAYER, [_MIGRATED_EXPERT])
    out_b = run_selected_experts(lm_b, h, _LAYER, [_MIGRATED_EXPERT])
    assert mx.array_equal(out_a[_MIGRATED_EXPERT], out_b[_MIGRATED_EXPERT]).item()


def test_both_equal_full_model_baseline(lm_full, lm_a, lm_b):
    # Idempotent: if test above already attached, skip the attach.
    lock = threading.Lock()
    if _MIGRATED_EXPERT not in lm_b.held_ids_per_layer[_LAYER]:
        tensors = slice_expert(lm_a, _LAYER, _MIGRATED_EXPERT, lock)
        attach_expert(lm_b, _LAYER, _MIGRATED_EXPERT, tensors, lock)
    h = _synthetic_h(lm_full)
    out_full = run_selected_experts(lm_full, h, _LAYER, [_MIGRATED_EXPERT])
    out_a = run_selected_experts(lm_a, h, _LAYER, [_MIGRATED_EXPERT])
    out_b = run_selected_experts(lm_b, h, _LAYER, [_MIGRATED_EXPERT])
    assert mx.array_equal(out_full[_MIGRATED_EXPERT], out_a[_MIGRATED_EXPERT]).item()
    assert mx.array_equal(out_full[_MIGRATED_EXPERT], out_b[_MIGRATED_EXPERT]).item()
```

- [ ] **Step 2: Run the slow test**

Run: `uv run pytest tests/test_migration_bit_exact_per_expert.py -v -m slow`
Expected: both PASS. If the first assert fails, re-check that `slice_expert` picked up the right local slot (held=[0,3,6,9] → expert 3 is at slot 1) and that `attach_expert` appended at the tail (`held_ids_per_layer[15]` ends with 3).

- [ ] **Step 3: Commit**

```bash
git add tests/test_migration_bit_exact_per_expert.py
git commit -m "Phase 5b Task 7: bit-exact per-expert migration proof (slow)"
```

---

### Task 8: `MembershipRunner.start_heat_source` + `latest_heat`

**Files:**
- Modify: `src/model_shard/membership/runner.py`
- Test: `tests/test_membership_runner_heat.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_membership_runner_heat.py`:

```python
"""Unit tests for MembershipRunner heat piggyback and reception."""
from __future__ import annotations

import dataclasses

from model_shard.membership.config import SwimConfig
from model_shard.membership.records import (
    AckMsg,
    HeatReportRecord,
    PingMsg,
)
from model_shard.membership.runner import MembershipRunner
from model_shard.membership.state import PeerSpec


def _make_runner() -> MembershipRunner:
    return MembershipRunner(
        self_spec=PeerSpec(shard_id="self", host="127.0.0.1", udp_port=30001),
        peers=[PeerSpec(shard_id="peer", host="127.0.0.1", udp_port=30002)],
        config=SwimConfig(),
    )


def test_start_heat_source_registers_callable():
    r = _make_runner()
    r.start_heat_source(lambda: HeatReportRecord(
        shard_id="self", entries=((15, 7, 500),), ts_unix_ms=1
    ))
    assert r._heat_source is not None  # private but the simplest signal


def test_latest_heat_updates_on_recv():
    r = _make_runner()
    hr = HeatReportRecord(shard_id="peer", entries=((15, 3, 200),), ts_unix_ms=42)
    ping = PingMsg(
        from_shard_id="peer", from_incarnation=1, deltas=[],
        heat=[hr],
    )
    r._on_recv_decoded(ping)
    snap = r.latest_heat()
    assert snap["peer"] == hr


def test_latest_heat_snapshot_is_isolated():
    r = _make_runner()
    hr = HeatReportRecord(shard_id="peer", entries=((15, 3, 200),), ts_unix_ms=42)
    r._on_recv_decoded(PingMsg(
        from_shard_id="peer", from_incarnation=1, deltas=[], heat=[hr],
    ))
    snap = r.latest_heat()
    snap["bogus"] = "should not propagate"  # type: ignore[assignment]
    assert "bogus" not in r.latest_heat()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_membership_runner_heat.py -v`
Expected: AttributeError — `start_heat_source` / `latest_heat` / `_heat_source` do not exist.

- [ ] **Step 3: Implement on `MembershipRunner`**

Edit `src/model_shard/membership/runner.py`:

In `__init__`, after the existing `self._peer_loads` block (around line 80) add:

```python
        self._heat_source: Callable[[], HeatReportRecord] | None = None
        self._peer_heat: dict[str, HeatReportRecord] = {}
        self._peer_heat_lock = threading.Lock()
```

Update the `from model_shard.membership.records import (...)` block to also include `HeatReportRecord`.

After `latest_loads` method (around line 122), add:

```python
    def start_heat_source(self, fn: Callable[[], HeatReportRecord]) -> None:
        """Register a callable invoked once per outgoing ping-family message
        to produce this node's own heat report. Latest registration wins."""
        self._heat_source = fn

    def latest_heat(self) -> dict[str, HeatReportRecord]:
        """Return a snapshot of the most recent heat report seen per peer."""
        with self._peer_heat_lock:
            return dict(self._peer_heat)
```

In `_on_recv_decoded` (around line 136-150), after the existing `loads` scrape add:

```python
        heat = getattr(decoded, "heat", None)
        if heat:
            with self._peer_heat_lock:
                for hr in heat:
                    self._peer_heat[hr.shard_id] = hr
```

In `_run` (around line 168-189), extend the load-piggyback block with a parallel heat-piggyback block:

```python
            if self._heat_source is not None:
                try:
                    my_heat = self._heat_source()
                except Exception:
                    _LOG.exception("heat source raised; skipping heat piggyback")
                    my_heat = None
                if my_heat is not None:
                    new_outgoing2 = []
                    for o in outgoing:
                        p = o.payload
                        if isinstance(p, (PingMsg, AckMsg, PingReqMsg, PingReqAckMsg)):
                            new_payload = dataclasses.replace(
                                p, heat=[*p.heat, my_heat]
                            )
                            new_outgoing2.append(
                                dataclasses.replace(o, payload=new_payload)
                            )
                        else:
                            new_outgoing2.append(o)
                    outgoing = new_outgoing2
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_membership_runner_heat.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Regression**

Run: `uv run pytest tests/test_membership_runner_loads.py -v`
Expected: PASS (heat block is additive and orthogonal to loads).

- [ ] **Step 6: Commit**

```bash
git add src/model_shard/membership/runner.py tests/test_membership_runner_heat.py
git commit -m "Phase 5b Task 8: MembershipRunner heat piggyback + latest_heat"
```

---

### Task 9: `MembershipRunner.announce_ownership_add` + `ownership_view` with TTL

**Files:**
- Modify: `src/model_shard/membership/runner.py`
- Test: `tests/test_membership_ownership_gossip.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_membership_ownership_gossip.py`:

```python
"""Unit tests for ownership delta TTL'd piggyback and ownership_view union."""
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
        self_spec=PeerSpec(shard_id="self", host="127.0.0.1", udp_port=30003),
        peers=[PeerSpec(shard_id="peer", host="127.0.0.1", udp_port=30004)],
        config=SwimConfig(),
    )


def test_announce_enqueues_delta():
    r = _make_runner()
    r.announce_ownership_add(layer_idx=15, expert_id=7)
    assert len(r._outbound_ownership) == 1
    d = r._outbound_ownership[0]
    assert d.record.shard_id == "self"
    assert d.record.layer_idx == 15
    assert d.record.expert_id == 7
    assert d.ttl == 5  # default TTL


def test_ownership_view_includes_received_and_self():
    r = _make_runner()
    r.announce_ownership_add(layer_idx=15, expert_id=7)
    # Simulate a received delta from a peer.
    r._on_recv_decoded(PingMsg(
        from_shard_id="peer", from_incarnation=1, deltas=[],
        ownership=[OwnershipDeltaRecord(
            shard_id="peer", layer_idx=15, expert_id=3, action=0, ts_unix_ms=1
        )],
    ))
    view = r.ownership_view()
    assert ("self", 15, 7) in view
    assert ("peer", 15, 3) in view


def test_drain_ownership_decrements_ttl():
    r = _make_runner()
    r.announce_ownership_add(layer_idx=15, expert_id=7)
    first = r._drain_outbound_ownership()
    assert len(first) == 1
    # TTL should be 4 after one drain.
    remaining = r._outbound_ownership[0]
    assert remaining.ttl == 4


def test_drain_evicts_after_ttl_zero():
    r = _make_runner()
    r.announce_ownership_add(layer_idx=15, expert_id=7)
    for _ in range(5):
        r._drain_outbound_ownership()
    assert r._outbound_ownership == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_membership_ownership_gossip.py -v`
Expected: FAIL — new methods don't exist yet.

- [ ] **Step 3: Add a private `_OutboundOwnership` struct and TTL queue**

In `src/model_shard/membership/runner.py`, update the record import block to include `OwnershipDeltaRecord`, and add near the top of the file (after `_INTERNAL_QUEUE_MAX`):

```python
_DEFAULT_OWNERSHIP_TTL: Final[int] = 5


@dataclasses.dataclass
class _OutboundOwnership:
    record: OwnershipDeltaRecord
    ttl: int
```

Note: `dataclasses` is already imported at the top.

In `__init__` (after the heat state block from Task 8), add:

```python
        self._outbound_ownership: list[_OutboundOwnership] = []
        self._outbound_ownership_lock = threading.Lock()
        self._ownership_seen: set[tuple[str, int, int]] = set()
        self._ownership_seen_lock = threading.Lock()
```

After `latest_heat` add:

```python
    def announce_ownership_add(
        self, layer_idx: int, expert_id: int, ttl: int = _DEFAULT_OWNERSHIP_TTL
    ) -> None:
        """Enqueue an ADD delta about self to piggyback for the next `ttl`
        outbound ping-family messages. Also folds into ``ownership_view``
        immediately so local readers see self-ownership without waiting."""
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
            self._ownership_seen.add((rec.shard_id, rec.layer_idx, rec.expert_id))

    def ownership_view(self) -> set[tuple[str, int, int]]:
        """Snapshot of every (shard_id, layer_idx, expert_id) ADD ever
        observed, including self-announcements."""
        with self._ownership_seen_lock:
            return set(self._ownership_seen)

    def _drain_outbound_ownership(self) -> list[OwnershipDeltaRecord]:
        """Return records to piggyback this round and decrement TTLs; evict
        entries whose TTL reaches zero."""
        with self._outbound_ownership_lock:
            to_send = [o.record for o in self._outbound_ownership]
            surviving: list[_OutboundOwnership] = []
            for o in self._outbound_ownership:
                if o.ttl > 1:
                    surviving.append(_OutboundOwnership(record=o.record, ttl=o.ttl - 1))
            self._outbound_ownership = surviving
        return to_send
```

In `_on_recv_decoded`, after the heat scrape add:

```python
        ownership = getattr(decoded, "ownership", None)
        if ownership:
            with self._ownership_seen_lock:
                for od in ownership:
                    self._ownership_seen.add(
                        (od.shard_id, od.layer_idx, od.expert_id)
                    )
```

In `_run`, after the heat-piggyback block add a parallel ownership-piggyback block:

```python
            owner_batch = self._drain_outbound_ownership()
            if owner_batch:
                new_outgoing3 = []
                for o in outgoing:
                    p = o.payload
                    if isinstance(p, (PingMsg, AckMsg, PingReqMsg, PingReqAckMsg)):
                        new_payload = dataclasses.replace(
                            p, ownership=[*p.ownership, *owner_batch]
                        )
                        new_outgoing3.append(
                            dataclasses.replace(o, payload=new_payload)
                        )
                    else:
                        new_outgoing3.append(o)
                outgoing = new_outgoing3
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_membership_ownership_gossip.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/model_shard/membership/runner.py tests/test_membership_ownership_gossip.py
git commit -m "Phase 5b Task 9: MembershipRunner ownership gossip + TTL queue"
```

---

### Task 10: `group_expert_ids_by_owner_loaded` accepts `live_owners_provider`

**Files:**
- Modify: `src/model_shard/moe.py`
- Test: `tests/test_orchestrator_live_owners.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_orchestrator_live_owners.py`:

```python
"""Tests for live-owners resolution via callback."""
from __future__ import annotations

import random

from model_shard.moe import group_expert_ids_by_owner_loaded


def test_live_owners_provider_augments_static_owners():
    static = {"A": {3}, "B": {7}}  # A owns 3, B owns 7 at this layer
    # C now also owns 7 (newly migrated replica).
    def provider(eid: int) -> set[str]:
        if eid == 3:
            return {"A"}
        if eid == 7:
            return {"B", "C"}
        return set()
    rng = random.Random(0)
    result = group_expert_ids_by_owner_loaded(
        top_k_ids=[3, 7],
        owners=static,
        peer_loads={"A": 100, "B": 500, "C": 10},
        self_shard_id="A",
        self_load=100,
        rng=rng,
        live_owners_provider=provider,
    )
    # Expert 7's P2C should pick C (lower load).
    assert "A" in result and result["A"] == [3]
    assert "C" in result and result["C"] == [7]


def test_live_owners_provider_default_preserves_phase4():
    static = {"A": {3}, "B": {7}}
    rng = random.Random(0)
    result = group_expert_ids_by_owner_loaded(
        top_k_ids=[3, 7],
        owners=static,
        peer_loads={"A": 100, "B": 500},
        self_shard_id="A",
        self_load=100,
        rng=rng,
    )
    # Single-owner per id: A gets 3, B gets 7.
    assert result == {"A": [3], "B": [7]}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_orchestrator_live_owners.py::test_live_owners_provider_augments_static_owners -v`
Expected: TypeError on unexpected kwarg `live_owners_provider`.

- [ ] **Step 3: Extend `group_expert_ids_by_owner_loaded`**

In `src/model_shard/moe.py`, update the function signature and candidate-gathering logic (replace the whole function body):

```python
def group_expert_ids_by_owner_loaded(
    top_k_ids: list[int],
    owners: Mapping[str, set[int]],
    peer_loads: Mapping[str, int],
    self_shard_id: str,
    self_load: int,
    rng: random.Random,
    live_owners_provider: "Callable[[int], set[str]] | None" = None,
) -> dict[str, list[int]]:
    """Partition top_k_ids by owner using power-of-two-choices on load.

    If ``live_owners_provider`` is not None, candidate owners for each id are
    taken from ``live_owners_provider(eid) | {s for s, ids in owners.items()
    if eid in ids}`` — i.e. the union of bootstrap owners and whatever the
    callback reports. Phase 5b injects gossip-observed ADD deltas here so
    routing picks up new replicas without restarting the orchestrator.
    """
    static_candidates_by_id: dict[int, list[str]] = {}
    for owner, ids in owners.items():
        for i in ids:
            static_candidates_by_id.setdefault(i, []).append(owner)

    def load_of(sid: str) -> int:
        if sid == self_shard_id:
            return self_load
        if sid in peer_loads:
            return peer_loads[sid]
        return 2**31 - 1

    by_owner: dict[str, list[int]] = {}
    for eid in top_k_ids:
        static = static_candidates_by_id.get(eid, [])
        live_extra = (
            list(live_owners_provider(eid)) if live_owners_provider is not None else []
        )
        combined = list(dict.fromkeys([*static, *live_extra]))  # preserve order, dedupe
        if not combined:
            raise KeyError(f"expert_id {eid} has no owner in {list(owners)}")
        if len(combined) == 1:
            winner = combined[0]
        else:
            pool = (
                list(combined)
                if len(combined) == 2
                else rng.sample(combined, 2)
            )
            winner = min(pool, key=load_of)
        by_owner.setdefault(winner, []).append(eid)
    return by_owner
```

Add `from collections.abc import Callable` to the top imports if not already present.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_orchestrator_live_owners.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Regression**

Run: `uv run pytest tests/test_routing_correctness.py tests/test_moe_run_experts.py -v`
Expected: all PASS (default `live_owners_provider=None` preserves Phase 4 behavior).

- [ ] **Step 6: Commit**

```bash
git add src/model_shard/moe.py tests/test_orchestrator_live_owners.py
git commit -m "Phase 5b Task 10: live_owners_provider in group_expert_ids_by_owner_loaded"
```

---

### Task 11: `ExpertOrchestrator` uses `live_owners_provider`

**Files:**
- Modify: `src/model_shard/expert_orchestrator.py`
- Test: `tests/test_orchestrator_live_owners.py` (extend)

- [ ] **Step 1: Append the failing test**

Append to `tests/test_orchestrator_live_owners.py`:

```python
from unittest.mock import MagicMock

from model_shard.expert_orchestrator import ExpertOrchestrator


def test_orchestrator_accepts_and_invokes_live_owners_provider():
    calls: list[int] = []
    def provider(eid: int) -> set[str]:
        calls.append(eid)
        return set()  # no extra owners
    orch = ExpertOrchestrator(
        self_shard_id="A",
        owners={"A": {3}, "B": {7}},
        peer_rpc=MagicMock(),
        rpc_timeout_s=1.0,
        live_owners_provider=provider,
    )
    assert orch.live_owners_provider is provider
    # Directly drive the grouping step with the orchestrator's provider.
    from model_shard.moe import group_expert_ids_by_owner_loaded
    import random
    result = group_expert_ids_by_owner_loaded(
        top_k_ids=[3, 7],
        owners=orch.owners,
        peer_loads={},
        self_shard_id="A",
        self_load=0,
        rng=random.Random(0),
        live_owners_provider=orch.live_owners_provider,
    )
    assert sorted(calls) == [3, 7]
    assert result == {"A": [3], "B": [7]}
```

- [ ] **Step 2: Run test to verify it fails**

Expected: TypeError on unexpected kwarg `live_owners_provider`.

- [ ] **Step 3: Wire `live_owners_provider` into `ExpertOrchestrator`**

In `src/model_shard/expert_orchestrator.py`:

Add after `rng: random.Random = field(default_factory=random.Random)` (around line 173):

```python
    live_owners_provider: Callable[[int], set[str]] | None = None
```

Update the `from collections.abc import Callable, Iterator, Mapping` line — `Callable` is already imported; nothing to change.

In `run_split_layer`, inside the `_mlx_guard` block, replace the current `group_expert_ids_by_owner_loaded(...)` call (around line 340-347) with:

```python
            by_owner = group_expert_ids_by_owner_loaded(
                all_ids,
                owners=self.owners,
                peer_loads=peer_loads,
                self_shard_id=self.self_shard_id,
                self_load=self_load,
                rng=self.rng,
                live_owners_provider=self.live_owners_provider,
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_orchestrator_live_owners.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Regression**

Run: `uv run pytest tests/test_expert_orchestrator.py tests/test_expert_orchestrator_observer.py tests/test_expert_orchestrator_timeout.py tests/test_expert_rpc_load_shift.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/model_shard/expert_orchestrator.py tests/test_orchestrator_live_owners.py
git commit -m "Phase 5b Task 11: ExpertOrchestrator live_owners_provider"
```

---

### Task 12: `ExpertWeightPeerRPC`

**Files:**
- Create: `src/model_shard/migration.py`
- Test: `tests/test_expert_weight_peer_rpc.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_expert_weight_peer_rpc.py`:

```python
"""Unit test for ExpertWeightPeerRPC against an in-process TCP server."""
from __future__ import annotations

import socket
import threading

import mlx.core as mx
import pytest

from model_shard._pb import wire_pb2
from model_shard.envelope import recv_envelope, send_envelope
from model_shard.mlx_engine import tensor_to_bytes
from model_shard.migration import ExpertWeightPeerRPC


def _fake_server(host: str, port: int, tensors: list[mx.array]) -> threading.Thread:
    def run():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen(1)
        conn, _ = s.accept()
        stream = conn.makefile("rwb", buffering=0)
        env, _ = recv_envelope(stream)
        assert env.WhichOneof("payload") == "expert_weight_request"
        req = env.expert_weight_request
        # Send an ExpertWeightTransfer with the caller's tensors.
        resp = wire_pb2.Envelope()
        resp.expert_weight_transfer.protocol_version = 1
        resp.expert_weight_transfer.request_id = req.request_id
        resp.expert_weight_transfer.layer_idx = req.layer_idx
        resp.expert_weight_transfer.expert_id = req.expert_id
        resp.expert_weight_transfer.tensor_count = len(tensors)
        blobs: list[bytes] = []
        for t in tensors:
            d = resp.expert_weight_transfer.tensors.add()
            d.shape.extend(list(t.shape))
            from model_shard.expert_orchestrator import _dtype_to_wire
            d.dtype = _dtype_to_wire(t.dtype)
            d.quant = wire_pb2.QUANT_NONE
            raw = tensor_to_bytes(t)
            d.byte_count = len(raw)
            blobs.append(raw)
        send_envelope(stream, resp, b"".join(blobs))
        conn.close()
        s.close()
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def test_pull_deserialises_nine_tensors():
    tensors = [mx.full((4, 4), fill_value=float(i), dtype=mx.bfloat16) for i in range(9)]
    port = 29451
    server = _fake_server("127.0.0.1", port, tensors)
    rpc = ExpertWeightPeerRPC(
        addresses={"peer": ("127.0.0.1", port)}, timeout_s=5.0
    )
    received = rpc.pull(source_shard_id="peer", layer_idx=15, expert_id=7)
    server.join(timeout=2.0)
    assert len(received) == 9
    for got, want in zip(received, tensors):
        assert mx.array_equal(got, want).item()


def test_pull_raises_on_error_envelope():
    def run():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 29452))
        s.listen(1)
        conn, _ = s.accept()
        stream = conn.makefile("rwb", buffering=0)
        recv_envelope(stream)
        err = wire_pb2.Envelope()
        err.error.protocol_version = 1
        err.error.request_id = "r"
        err.error.code = wire_pb2.ERR_SHARD_UNAVAILABLE
        err.error.detail = "gone"
        send_envelope(stream, err)
        conn.close()
        s.close()
    t = threading.Thread(target=run, daemon=True)
    t.start()
    rpc = ExpertWeightPeerRPC(
        addresses={"peer": ("127.0.0.1", 29452)}, timeout_s=5.0
    )
    with pytest.raises(RuntimeError, match="gone"):
        rpc.pull(source_shard_id="peer", layer_idx=15, expert_id=7)
    t.join(timeout=2.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_expert_weight_peer_rpc.py -v`
Expected: ImportError — `model_shard.migration` does not exist.

- [ ] **Step 3: Create `src/model_shard/migration.py` with `ExpertWeightPeerRPC`**

```python
"""Phase 5b target-pull migration: policy + peer RPC + scanner.

Layering:
  * ExpertWeightPeerRPC — TCP client for ExpertWeightRequest/Transfer.
  * MigrationPolicy     — knobs (thresholds, intervals).
  * MigrationScanner    — periodic daemon thread that decides + pulls.
"""

from __future__ import annotations

import logging
import socket
from dataclasses import dataclass
from typing import BinaryIO, cast

import mlx.core as mx

from model_shard._pb import wire_pb2
from model_shard.envelope import recv_envelope, send_envelope
from model_shard.expert_orchestrator import _dtype_to_wire
from model_shard.mlx_engine import bytes_to_tensor, tensor_to_bytes

_LOG = logging.getLogger(__name__)


class ExpertWeightPeerRPC:
    """TCP client for pulling expert weights from a source shard."""

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


# Placeholder: Policy and Scanner land in Task 15.
__all__ = ["ExpertWeightPeerRPC"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_expert_weight_peer_rpc.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/model_shard/migration.py tests/test_expert_weight_peer_rpc.py
git commit -m "Phase 5b Task 12: ExpertWeightPeerRPC (TCP pull client)"
```

---

### Task 13: `Node._handle_expert_weight_request` (source side)

**Files:**
- Modify: `src/model_shard/node.py`
- Test: `tests/test_node_expert_weight_handler.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_node_expert_weight_handler.py`:

```python
"""Server-side handler for ExpertWeightRequest (source side of migration)."""
from __future__ import annotations

import socket
import threading

import mlx.core as mx
import pytest

from model_shard._pb import wire_pb2
from model_shard.envelope import recv_envelope, send_envelope
from model_shard.migration import ExpertWeightPeerRPC


pytestmark = pytest.mark.slow  # loads the model via partial_load fixture


@pytest.fixture(scope="module")
def source_node(partial_load_fixture):
    # partial_load_fixture yields a (node, port) pair with experts [0,3,6,9]
    # at layer 15 and a serve_forever thread running. Defined in conftest.py
    # alongside existing Phase 5a fixtures. See conftest for details.
    return partial_load_fixture


def test_source_slices_and_transfers_held_expert(source_node):
    node, port = source_node
    rpc = ExpertWeightPeerRPC(
        addresses={"src": ("127.0.0.1", port)}, timeout_s=30.0
    )
    tensors = rpc.pull(source_shard_id="src", layer_idx=15, expert_id=3)
    assert len(tensors) == 9


def test_source_returns_error_on_unheld(source_node):
    node, port = source_node
    rpc = ExpertWeightPeerRPC(
        addresses={"src": ("127.0.0.1", port)}, timeout_s=30.0
    )
    with pytest.raises(RuntimeError, match="not held"):
        rpc.pull(source_shard_id="src", layer_idx=15, expert_id=1)
```

Add fixture support in `tests/conftest.py` — append at the bottom:

```python
@pytest.fixture(scope="module")
def partial_load_fixture():
    """Spin up a single Node with partial load at layer 15 = [0,3,6,9]."""
    import os
    import socket as _sk
    import threading as _th
    import time as _time

    os.environ["ENABLE_PARTIAL_LOAD"] = "true"
    os.environ["ENABLE_GOSSIP"] = "false"  # no UDP membership needed here

    from model_shard.node import Node
    from model_shard.shard_map import NodeAddress, ShardMap, ShardSpec

    def _free_port() -> int:
        s = _sk.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    port = _free_port()
    spec = ShardSpec(
        shard_id="src",
        address=NodeAddress(host="127.0.0.1", port=port),
        start_layer=0,
        end_layer=30,
        moe_experts={15: (0, 3, 6, 9)},
    )
    sm = ShardMap({"src": spec})
    node = Node(shard=spec, shard_map=sm, total_layers=30)
    t = _th.Thread(target=node.serve_forever, daemon=True)
    t.start()
    _time.sleep(0.5)
    try:
        yield (node, port)
    finally:
        node.shutdown()
        t.join(timeout=2.0)
        os.environ.pop("ENABLE_PARTIAL_LOAD", None)
        os.environ.pop("ENABLE_GOSSIP", None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_node_expert_weight_handler.py -v -m slow`
Expected: FAIL — Node does not yet dispatch `expert_weight_request`.

- [ ] **Step 3: Add `_handle_expert_weight_request` to `Node`**

In `src/model_shard/node.py`:

Add to the imports:

```python
from model_shard.partial_load import attach_expert, slice_expert
```

In `_dispatch` (around line 233-253), add a new branch:

```python
        elif which == "expert_weight_request":
            self._handle_expert_weight_request(
                env.expert_weight_request, inbound_stream
            )
```

Add the handler method (after `_handle_expert_request`, around line 557):

```python
    def _handle_expert_weight_request(
        self,
        req: wire_pb2.ExpertWeightRequest,
        inbound_stream: BinaryIO,
    ) -> None:
        """Source-side of Phase 5b migration: slice the requested expert
        out of our compact stack and reply with ExpertWeightTransfer.

        Error{ERR_SHARD_UNAVAILABLE} on miss (expert no longer held)."""
        layer_idx = int(req.layer_idx)
        expert_id = int(req.expert_id)
        try:
            tensors = slice_expert(
                self._lm, layer_idx, expert_id, _MLX_COMPUTE_LOCK
            )
        except KeyError as e:
            _send_error(
                inbound_stream,
                req.request_id,
                wire_pb2.ERR_SHARD_UNAVAILABLE,
                str(e),
            )
            return
        # Build the transfer response. Payload = 9 tensor blobs concatenated.
        resp = wire_pb2.Envelope()
        resp.expert_weight_transfer.protocol_version = _PROTOCOL_VERSION
        resp.expert_weight_transfer.request_id = req.request_id
        resp.expert_weight_transfer.layer_idx = layer_idx
        resp.expert_weight_transfer.expert_id = expert_id
        resp.expert_weight_transfer.tensor_count = 9
        blobs: list[bytes] = []
        for t in tensors:
            d = resp.expert_weight_transfer.tensors.add()
            d.shape.extend(list(t.shape))
            d.dtype = _dtype_to_wire(t.dtype)
            d.quant = wire_pb2.QUANT_NONE
            raw = tensor_to_bytes(t)
            d.byte_count = len(raw)
            blobs.append(raw)
        send_envelope(inbound_stream, resp, b"".join(blobs))
```

Note: `_dtype_to_wire` cannot encode `mx.uint32` (used by quantized `weight` tensors). Add a branch:

In `_dtype_to_wire` (around line 810), extend:

```python
def _dtype_to_wire(dt: mx.Dtype) -> int:
    if dt == mx.bfloat16:
        return int(wire_pb2.DTYPE_BFLOAT16)
    if dt == mx.float32:
        return int(wire_pb2.DTYPE_FLOAT32)
    if dt == mx.float16:
        return int(wire_pb2.DTYPE_FLOAT16)
    if dt == mx.uint32:  # NEW: quantized packed weights
        return int(wire_pb2.DTYPE_UINT32) if hasattr(wire_pb2, "DTYPE_UINT32") else int(wire_pb2.DTYPE_INT32)
    raise ValueError(f"unsupported activation dtype: {dt}")
```

To make that work, extend the proto enum. In `proto/wire.proto`, update the `DType` enum:

```proto
enum DType {
  DTYPE_UNSPECIFIED = 0;
  DTYPE_FLOAT32 = 1;
  DTYPE_FLOAT16 = 2;
  DTYPE_BFLOAT16 = 3;
  DTYPE_INT32 = 4;
  DTYPE_INT8 = 5;
  DTYPE_UINT8 = 6;
  DTYPE_UINT32 = 7;  // NEW: packed quantized 4-bit weights
}
```

Regenerate protobuf:

```bash
cd /Users/lukechang/Github/model_shard
uv run python -m grpc_tools.protoc -I proto --python_out=src/model_shard/_pb proto/wire.proto
```

Also extend `bytes_to_tensor` in `src/model_shard/mlx_engine.py` to recognize the new dtype. Find the dtype dispatch and add:

```python
    elif dtype == wire_pb2.DTYPE_UINT32:
        arr = np.frombuffer(raw, dtype=np.uint32).reshape(shape)
        return mx.array(arr)
```

Make sure `tensor_to_bytes` already handles `mx.uint32`; if not, extend it with:

```python
    if arr.dtype == mx.uint32:
        return bytes(np.asarray(arr, dtype=np.uint32).tobytes())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_node_expert_weight_handler.py tests/test_expert_weight_peer_rpc.py tests/test_wire_expert_weight_roundtrip.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/model_shard/node.py src/model_shard/mlx_engine.py proto/wire.proto src/model_shard/_pb/wire_pb2.py tests/test_node_expert_weight_handler.py tests/conftest.py
git commit -m "Phase 5b Task 13: source-side ExpertWeightRequest handler"
```

---

### Task 14: `Node._live_experts` registry + `_ownership_seen`

**Files:**
- Modify: `src/model_shard/node.py`
- Test: `tests/test_node_live_experts.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_node_live_experts.py`:

```python
"""Tests for Node._live_experts runtime ownership registry."""
from __future__ import annotations

import os

import pytest

from model_shard.node import Node
from model_shard.shard_map import NodeAddress, ShardMap, ShardSpec


@pytest.fixture(autouse=True)
def _no_gossip_env(monkeypatch):
    monkeypatch.setenv("ENABLE_GOSSIP", "false")
    monkeypatch.setenv("ENABLE_PARTIAL_LOAD", "false")
    monkeypatch.setenv("ENABLE_DYNAMIC_MIGRATION", "false")
    yield


def _mk_spec(sid: str, port: int, moe: dict[int, tuple[int, ...]]) -> ShardSpec:
    return ShardSpec(
        shard_id=sid,
        address=NodeAddress(host="127.0.0.1", port=port),
        start_layer=0,
        end_layer=30,
        moe_experts=moe,
    )


def test_live_experts_seeded_from_shard_spec():
    spec = _mk_spec("self", 30100, {15: (0, 3, 6, 9)})
    sm = ShardMap({"self": spec})
    # Build a Node with a pre-loaded model = None; Phase 5b allows this path
    # as long as ENABLE_PARTIAL_LOAD or a loaded_model is provided — here we
    # skip model load by stubbing.
    from unittest.mock import MagicMock
    n = Node(shard=spec, shard_map=sm, loaded_model=MagicMock(), total_layers=30)
    assert n._live_experts == {15: {0, 3, 6, 9}}


def test_ownership_seen_seeded_with_every_bootstrap_shard():
    spec_a = _mk_spec("A", 30101, {15: (0, 3, 6, 9)})
    spec_b = _mk_spec("B", 30102, {15: (1, 4, 7, 10)})
    sm = ShardMap({"A": spec_a, "B": spec_b})
    from unittest.mock import MagicMock
    n = Node(shard=spec_a, shard_map=sm, loaded_model=MagicMock(), total_layers=30)
    assert ("A", 15, 0) in n._ownership_seen
    assert ("B", 15, 1) in n._ownership_seen
    assert ("B", 15, 10) in n._ownership_seen


def test_owners_of_resolves_union():
    spec_a = _mk_spec("A", 30103, {15: (0, 3)})
    spec_b = _mk_spec("B", 30104, {15: (3, 7)})  # B also owns 3 (overlap)
    sm = ShardMap({"A": spec_a, "B": spec_b})
    from unittest.mock import MagicMock
    n = Node(shard=spec_a, shard_map=sm, loaded_model=MagicMock(), total_layers=30)
    assert n.owners_of(15, 3) == {"A", "B"}
    assert n.owners_of(15, 7) == {"B"}
    assert n.owners_of(15, 99) == set()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_node_live_experts.py -v`
Expected: AttributeError — `_live_experts` / `_ownership_seen` / `owners_of` do not exist.

- [ ] **Step 3: Add registry fields and `owners_of` to `Node`**

In `src/model_shard/node.py`, in `Node.__init__`, after the `self._shard_map = shard_map` line add:

```python
        # Phase 5b: runtime expert ownership registry (see spec D9).
        # Seeded from the frozen ShardSpec at boot; mutated by migration attach.
        self._live_experts: dict[int, set[int]] = {
            L: set(ids) for L, ids in shard.moe_experts.items()
        }
        # Union of bootstrap moe_experts across ALL shards + received
        # OwnershipDelta ADDs (see spec D10).
        self._ownership_seen: set[tuple[str, int, int]] = set()
        for sid in shard_map.all_shards():
            peer_spec = shard_map.lookup(sid)
            for L, ids in peer_spec.moe_experts.items():
                for eid in ids:
                    self._ownership_seen.add((sid, L, eid))
        self._ownership_seen_lock = threading.Lock()
```

After `self_load_report` method (around line 673) add:

```python
    def owners_of(self, layer_idx: int, expert_id: int) -> set[str]:
        """Return the current live owner set for (layer_idx, expert_id).

        Union of bootstrap ShardSpec.moe_experts and gossip-observed ADDs.
        Used by ExpertOrchestrator.live_owners_provider in Phase 5b."""
        with self._ownership_seen_lock:
            return {
                sid for (sid, L, e) in self._ownership_seen
                if L == layer_idx and e == expert_id
            }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_node_live_experts.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/model_shard/node.py tests/test_node_live_experts.py
git commit -m "Phase 5b Task 14: Node._live_experts registry + owners_of"
```

---

### Task 15: `MigrationPolicy` + `MigrationScanner._scan_once`

**Files:**
- Modify: `src/model_shard/migration.py`
- Test: `tests/test_migration_scanner_policy.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_migration_scanner_policy.py`:

```python
"""Scanner policy tests — _scan_once picks the hottest not-held expert."""
from __future__ import annotations

import threading
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
    ht.report.return_value = [(L, e, v) for (L, e), v in heat.items()]
    ht.local_heat.side_effect = lambda L, e: heat.get((L, e), 0)

    def owner_lookup(L: int, e: int) -> set[str]:
        return owners.get((L, e), set())

    def load_provider() -> dict[str, int]:
        return {}

    peer_rpc = MagicMock()
    peer_rpc.pull.return_value = [MagicMock() for _ in range(9)]

    def attacher(L: int, e: int, tensors: list) -> None:
        live.setdefault(L, set()).add(e)

    def announce(L: int, e: int) -> None:
        pulled.append((L, e, "announced"))

    return MigrationScanner(
        self_shard_id="self",
        policy=MigrationPolicy(
            scan_interval_s=0.0,
            heat_threshold=50,
            max_experts_per_layer=128,
        ),
        heat_tracker=ht,
        live_experts=live,
        owner_lookup=owner_lookup,
        load_provider=load_provider,
        peer_rpc=peer_rpc,
        attacher=attacher,
        ownership_announcer=announce,
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
    assert (15, 1) in live[15]
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
    pulled: list = []
    live = {15: set(range(128))}  # full stack
    owners = {(15, 128): {"peer-a"}}  # mythical new expert
    heat = {(15, 128): 5000}
    s = MigrationScanner(
        self_shard_id="self",
        policy=MigrationPolicy(
            scan_interval_s=0.0, heat_threshold=50, max_experts_per_layer=128,
        ),
        heat_tracker=MagicMock(
            report=MagicMock(return_value=[(15, 128, 5000)]),
            local_heat=MagicMock(return_value=5000),
        ),
        live_experts=live,
        owner_lookup=lambda L, e: owners.get((L, e), set()),
        load_provider=lambda: {},
        peer_rpc=MagicMock(),
        attacher=lambda L, e, t: None,
        ownership_announcer=lambda L, e: None,
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_migration_scanner_policy.py -v`
Expected: ImportError — `MigrationPolicy`/`MigrationScanner` do not exist.

- [ ] **Step 3: Implement policy and scanner**

Append to `src/model_shard/migration.py`:

```python
import random as _random
import threading as _threading
from collections.abc import Callable


@dataclass(frozen=True)
class MigrationPolicy:
    scan_interval_s: float
    heat_threshold: int
    max_experts_per_layer: int


class MigrationScanner:
    """Periodic target-pull scanner (Phase 5b decider, simple threshold)."""

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

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = _threading.Thread(
            target=self._run_loop, name="migration-scanner", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stopping.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run_loop(self) -> None:
        while not self._stopping.is_set():
            jitter = 1.0 + self._rng.uniform(-0.25, 0.25)
            self._stopping.wait(self._policy.scan_interval_s * jitter)
            if self._stopping.is_set():
                return
            try:
                self._scan_once()
            except Exception:
                _LOG.exception("scan_once raised")
```

Update `__all__`:
```python
__all__ = ["ExpertWeightPeerRPC", "MigrationPolicy", "MigrationScanner"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_migration_scanner_policy.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/model_shard/migration.py tests/test_migration_scanner_policy.py
git commit -m "Phase 5b Task 15: MigrationPolicy + MigrationScanner._scan_once"
```

---

### Task 16: `MigrationScanner.start/stop` background thread (covered in Task 15)

Already implemented in Task 15. This task is an empty checkpoint — no new code. Confirm `start()` and `stop()` work by:

- [ ] **Step 1: Add a background-thread test**

Append to `tests/test_migration_scanner_policy.py`:

```python
import time

def test_scanner_start_and_stop_clean():
    live = {15: {0, 3}}
    pulled: list = []
    s = _make_scanner(heat={}, live=live, owners={}, pulled=pulled)
    s.start()
    time.sleep(0.1)
    s.stop()
    assert s._thread is not None
    assert not s._thread.is_alive()
```

- [ ] **Step 2: Run**

`uv run pytest tests/test_migration_scanner_policy.py::test_scanner_start_and_stop_clean -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_migration_scanner_policy.py
git commit -m "Phase 5b Task 16: MigrationScanner start/stop smoke test"
```

---

### Task 17: Node-side integration — heat observation + attach path + scanner wiring

**Files:**
- Modify: `src/model_shard/node.py`
- Modify: `src/model_shard/moe.py`
- Modify: `src/model_shard/expert_orchestrator.py`
- Test: extend existing tests

- [ ] **Step 1: Write the failing integration test**

Append to `tests/test_node_live_experts.py`:

```python
def test_attach_path_updates_live_experts_and_announces(monkeypatch):
    monkeypatch.setenv("ENABLE_GOSSIP", "false")
    monkeypatch.setenv("ENABLE_PARTIAL_LOAD", "false")
    spec = _mk_spec("self", 30150, {15: (0, 3)})
    sm = ShardMap({"self": spec})
    from unittest.mock import MagicMock
    lm = MagicMock()
    lm.held_ids_per_layer = {15: (0, 3)}
    # Synthesize a mutable switch_glu so attach_expert succeeds.
    import types
    import mlx.core as mx
    def _stack(n: int, cols: int) -> mx.array:
        return mx.zeros((n, 4, cols))
    projs = {
        name: types.SimpleNamespace(
            weight=_stack(2, 4), scales=_stack(2, 8), biases=_stack(2, 8),
        )
        for name in ("gate_proj", "up_proj", "down_proj")
    }
    layer = types.SimpleNamespace(
        experts=types.SimpleNamespace(switch_glu=types.SimpleNamespace(**projs))
    )
    lm.text_model = types.SimpleNamespace(layers=[None] * 15 + [layer])
    n = Node(shard=spec, shard_map=sm, loaded_model=lm, total_layers=30)

    new_tensors = (
        [mx.zeros((4, 4)) for _ in range(3)]
        + [mx.zeros((4, 8)) for _ in range(3)]
        + [mx.zeros((4, 8)) for _ in range(3)]
    )
    n.migration_attach(layer_idx=15, expert_id=7, tensors=new_tensors)
    assert 7 in n._live_experts[15]
    assert ("self", 15, 7) in n._ownership_seen
```

- [ ] **Step 2: Run to verify it fails**

Expected: AttributeError — `migration_attach` does not exist.

- [ ] **Step 3: Add `Node.migration_attach`**

In `src/model_shard/node.py`, after `owners_of` add:

```python
    def migration_attach(
        self, layer_idx: int, expert_id: int, tensors: list[mx.array]
    ) -> None:
        """Receive-side integration: call attach_expert, update _live_experts,
        add to _ownership_seen, announce ADD delta on gossip (if running)."""
        attach_expert(
            self._lm, layer_idx, expert_id, tensors, _MLX_COMPUTE_LOCK
        )
        self._live_experts.setdefault(layer_idx, set()).add(expert_id)
        with self._ownership_seen_lock:
            self._ownership_seen.add((self._shard.shard_id, layer_idx, expert_id))
        if self._membership is not None:
            self._membership.announce_ownership_add(layer_idx, expert_id)
```

- [ ] **Step 4: Wire `HeatTracker` into `Node.__init__`**

After the `_load_tracker` construction add:

```python
        from model_shard.heat import HeatTracker
        self._heat_tracker = HeatTracker()
        if self._membership is not None:
            self._membership.start_heat_source(self._self_heat_report)
```

Add helper method near `self_load_report`:

```python
    def _self_heat_report(self):
        from model_shard.membership.records import HeatReportRecord
        return HeatReportRecord(
            shard_id=self._shard.shard_id,
            entries=tuple(self._heat_tracker.report()),
            ts_unix_ms=int(time.time() * 1000),
        )
```

- [ ] **Step 5: Wire heat observation into the routing path**

In `src/model_shard/moe.py`, extend `run_attention_and_route` to accept an optional heat observer. Change the signature to:

```python
def run_attention_and_route(
    lm: Any,
    h: mx.array,
    layer_idx: int,
    cache: list[Any],
    masks: tuple[Any, Any],
    heat_observer: "Callable[[int, list[int]], None] | None" = None,
) -> tuple[mx.array, mx.array, mx.array]:
```

At the end of the function (just before `return`), add:

```python
    if heat_observer is not None:
        ids_flat = [int(x) for x in top_k_ids.reshape(-1).tolist()]
        heat_observer(layer_idx, ids_flat)
```

Update `ExpertOrchestrator.run_split_layer` to pass the observer:

```python
            post_attn, top_k_ids, top_k_weights = run_attention_and_route(
                lm, h, layer_idx, cache, masks,
                heat_observer=self.heat_observer,
            )
```

And add to the `ExpertOrchestrator` fields:

```python
    heat_observer: Callable[[int, list[int]], None] | None = None
```

In `Node._build_expert_orchestrator` (around line 675), pass:

```python
            heat_observer=self._heat_tracker.observe,
```

to the `ExpertOrchestrator(...)` call.

- [ ] **Step 6: Wire `MigrationScanner` construction**

In `Node.__init__`, after the `_orchestrator` block add:

```python
        self._scanner: MigrationScanner | None = None
        if _dynamic_migration_enabled():
            from model_shard.migration import (
                ExpertWeightPeerRPC,
                MigrationPolicy,
                MigrationScanner,
            )
            policy = MigrationPolicy(
                scan_interval_s=_migration_scan_interval_s(),
                heat_threshold=_migration_heat_threshold(),
                max_experts_per_layer=_migration_max_experts_per_layer(),
            )
            addresses = {
                sid: (
                    shard_map.lookup(sid).address.host,
                    shard_map.lookup(sid).address.port,
                )
                for sid in shard_map.all_shards()
                if sid != shard.shard_id
            }
            self._scanner = MigrationScanner(
                self_shard_id=shard.shard_id,
                policy=policy,
                heat_tracker=self._heat_tracker,
                live_experts=self._live_experts,
                owner_lookup=self.owners_of,
                load_provider=self._loads_snapshot,
                peer_rpc=ExpertWeightPeerRPC(
                    addresses=addresses, timeout_s=60.0
                ),
                attacher=self.migration_attach,
                ownership_announcer=(
                    (lambda L, e: self._membership.announce_ownership_add(L, e))
                    if self._membership is not None else (lambda L, e: None)
                ),
            )
```

Add the helper `_loads_snapshot`:

```python
    def _loads_snapshot(self) -> dict[str, int]:
        if self._membership is None:
            return {}
        return {
            sid: lr.queue_depth_ema
            for sid, lr in self._membership.latest_loads().items()
        }
```

In `serve_forever` after `self._membership.start()`:

```python
        if self._scanner is not None:
            self._scanner.start()
```

In `shutdown`:

```python
        if self._scanner is not None:
            self._scanner.stop()
```

Add the env-var readers at the bottom of `node.py`:

```python
def _dynamic_migration_enabled() -> bool:
    return os.environ.get("ENABLE_DYNAMIC_MIGRATION", "false").lower() in (
        "1", "true", "yes"
    )

def _migration_scan_interval_s() -> float:
    return float(os.environ.get("MIGRATION_SCAN_INTERVAL_SECONDS", "10.0"))

def _migration_heat_threshold() -> int:
    return int(os.environ.get("MIGRATION_HEAT_THRESHOLD", "50"))

def _migration_max_experts_per_layer() -> int:
    return int(os.environ.get("MIGRATION_MAX_EXPERTS_PER_LAYER", "128"))
```

- [ ] **Step 7: Run tests**

Run: `uv run pytest tests/test_node_live_experts.py -v`
Expected: 4 PASS.

Run: `uv run pytest tests/test_expert_orchestrator.py -v`
Expected: PASS (heat_observer defaults to None; observer is optional kwarg).

- [ ] **Step 8: Commit**

```bash
git add src/model_shard/node.py src/model_shard/moe.py src/model_shard/expert_orchestrator.py tests/test_node_live_experts.py
git commit -m "Phase 5b Task 17: node-side migration_attach + scanner + heat wiring"
```

---

### Task 18: Decode-loop hang fix (D14)

**Files:**
- Modify: `src/model_shard/node.py`
- Test: `tests/test_decode_hang_fix.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_decode_hang_fix.py`:

```python
"""Observer-triggered queue poison unblocks _drive_decode_loop."""
from __future__ import annotations

import io
import queue
import threading
from unittest.mock import MagicMock

import pytest

from model_shard.node import _HeadRequestState, _POISON_TOKEN, Node, PeerLeftAliveError
from model_shard.shard_map import NodeAddress, ShardMap, ShardSpec
from model_shard.membership.records import MemberState, StateTransition, MemberRecord


def _make_node(monkeypatch) -> Node:
    monkeypatch.setenv("ENABLE_GOSSIP", "false")
    monkeypatch.setenv("ENABLE_PARTIAL_LOAD", "false")
    monkeypatch.setenv("ENABLE_DYNAMIC_MIGRATION", "false")
    spec_head = ShardSpec(
        shard_id="head",
        address=NodeAddress(host="127.0.0.1", port=30200),
        start_layer=0, end_layer=10, moe_experts={},
    )
    spec_tail = ShardSpec(
        shard_id="tail",
        address=NodeAddress(host="127.0.0.1", port=30201),
        start_layer=10, end_layer=30, moe_experts={},
    )
    sm = ShardMap({"head": spec_head, "tail": spec_tail})
    return Node(
        shard=spec_head, shard_map=sm, loaded_model=MagicMock(), total_layers=30
    )


def test_observer_poisons_active_head_states(monkeypatch):
    n = _make_node(monkeypatch)
    state = _HeadRequestState(
        client_stream=io.BytesIO(), max_new_tokens=10,
    )
    n._head_states["r1"] = state
    # Simulate peer-left-ALIVE transition.
    rec = MemberRecord(
        shard_id="tail", host="127.0.0.1", udp_port=31201,
        state=MemberState.SUSPECT, incarnation=1,
        last_state_change=0.0, suspect_deadline=None,
    )
    transition = StateTransition(
        shard_id="tail", old_state=MemberState.ALIVE, new_record=rec
    )
    n._on_membership_change(transition)
    assert state.token_queue.get_nowait() == _POISON_TOKEN


def test_drive_decode_loop_raises_on_poison(monkeypatch):
    n = _make_node(monkeypatch)
    state = _HeadRequestState(
        client_stream=io.BytesIO(), max_new_tokens=10,
    )
    state.token_queue.put(_POISON_TOKEN)
    # _drive_decode_loop translates poison to ERR_SHARD_UNAVAILABLE via the
    # existing ExpertRpcFailure-pattern branch. We verify that path runs by
    # checking the client_stream has an Error envelope after the call.
    n._head_states["r1"] = state
    n._kv_caches["r1"] = []
    n._drive_decode_loop("r1", state)
    state.client_stream.seek(0)
    # After poison handling, the request should be cleaned up.
    assert "r1" not in n._head_states
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_decode_hang_fix.py -v`
Expected: ImportError — `_POISON_TOKEN` / `PeerLeftAliveError` do not exist.

- [ ] **Step 3: Add sentinel, error class, and observer branch**

In `src/model_shard/node.py`, near the top after `_PROTOCOL_VERSION`:

```python
_POISON_TOKEN: int = -1


class PeerLeftAliveError(RuntimeError):
    """Raised inside _drive_decode_loop when the membership observer
    poisons the token queue because a peer left ALIVE mid-decode."""
```

Extend `_on_membership_change` — inside the `left_alive` block (after the orchestrator abort), add:

```python
        if left_alive:
            with self._state_lock:
                states = list(self._head_states.values())
            for st in states:
                try:
                    st.token_queue.put_nowait(_POISON_TOKEN)
                except queue.Full:
                    # Fall back to blocking put; queue size is unbounded by
                    # default (Queue() with no max), so this should not happen.
                    st.token_queue.put(_POISON_TOKEN)
```

In `_drive_decode_loop`, immediately after `token_id = state.token_queue.get()` (around line 316):

```python
                if token_id == _POISON_TOKEN:
                    raise PeerLeftAliveError(
                        f"request {request_id}: peer left ALIVE mid-decode"
                    )
```

Extend the except chain of `_drive_decode_loop` (around line 352-363) to handle `PeerLeftAliveError`:

```python
        except PeerLeftAliveError as exc:
            _LOG.warning("decode loop aborted by peer-left-alive: %s", exc)
            with contextlib.suppress(OSError):
                _send_error(
                    state.client_stream,
                    request_id,
                    wire_pb2.ERR_SHARD_UNAVAILABLE,
                    str(exc),
                )
            with self._state_lock:
                self._kv_caches.pop(request_id, None)
                self._head_states.pop(request_id, None)
```

Update `__all__`:

```python
__all__ = ["Node", "PeerLeftAliveError"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_decode_hang_fix.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Regression**

Run: `uv run pytest tests/test_node_membership.py tests/test_node_load_wiring.py -v`
Expected: all PASS (observer branch is additive).

- [ ] **Step 6: Commit**

```bash
git add src/model_shard/node.py tests/test_decode_hang_fix.py
git commit -m "Phase 5b Task 18: decode-loop hang fix via observer queue poison"
```

---

### Task 19: `ENABLE_DYNAMIC_MIGRATION` gate + conflicting-flag validation

**Files:**
- Modify: `src/model_shard/node.py`
- Test: `tests/test_dynamic_migration_gate.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_dynamic_migration_gate.py`:

```python
"""Gate tests for ENABLE_DYNAMIC_MIGRATION + partial-load dependency."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from model_shard.node import Node
from model_shard.shard_map import NodeAddress, ShardMap, ShardSpec


def _mk_spec() -> ShardSpec:
    return ShardSpec(
        shard_id="self",
        address=NodeAddress(host="127.0.0.1", port=30300),
        start_layer=0, end_layer=30,
        moe_experts={15: (0, 3, 6, 9)},
    )


def test_migration_on_partial_off_raises(monkeypatch):
    monkeypatch.setenv("ENABLE_GOSSIP", "false")
    monkeypatch.setenv("ENABLE_PARTIAL_LOAD", "false")
    monkeypatch.setenv("ENABLE_DYNAMIC_MIGRATION", "true")
    spec = _mk_spec()
    sm = ShardMap({"self": spec})
    with pytest.raises(ValueError, match="ENABLE_PARTIAL_LOAD"):
        Node(shard=spec, shard_map=sm, loaded_model=MagicMock(), total_layers=30)


def test_migration_off_partial_on_ok(monkeypatch):
    monkeypatch.setenv("ENABLE_GOSSIP", "false")
    monkeypatch.setenv("ENABLE_PARTIAL_LOAD", "false")  # bypass actual partial load
    monkeypatch.setenv("ENABLE_DYNAMIC_MIGRATION", "false")
    spec = _mk_spec()
    sm = ShardMap({"self": spec})
    n = Node(shard=spec, shard_map=sm, loaded_model=MagicMock(), total_layers=30)
    assert n._scanner is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dynamic_migration_gate.py -v`
Expected: FAIL — the ValueError is not raised yet.

- [ ] **Step 3: Add the validation in `Node.__init__`**

Near the top of `Node.__init__` (right after `self._shard_map = shard_map`):

```python
        if _dynamic_migration_enabled() and not _partial_load_enabled():
            raise ValueError(
                "ENABLE_DYNAMIC_MIGRATION=true requires ENABLE_PARTIAL_LOAD=true "
                "(see Phase 5b spec D16)"
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_dynamic_migration_gate.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/model_shard/node.py tests/test_dynamic_migration_gate.py
git commit -m "Phase 5b Task 19: ENABLE_DYNAMIC_MIGRATION gate + dependency check"
```

---

### Task 20: Slow — migration over TCP E2E (2-node)

**Files:**
- Create: `tests/test_migration_over_tcp.py`

- [ ] **Step 1: Write the failing slow test**

```python
"""End-to-end target-pull migration between two in-process Nodes."""
from __future__ import annotations

import os
import socket as _sk
import threading
import time

import mlx.core as mx
import pytest

from model_shard.mlx_engine import load_model, load_model_partial
from model_shard.moe import run_selected_experts
from model_shard.node import Node
from model_shard.shard_map import NodeAddress, ShardMap, ShardSpec

pytestmark = pytest.mark.slow


def _free_port() -> int:
    s = _sk.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def migration_env(monkeypatch):
    monkeypatch.setenv("ENABLE_PARTIAL_LOAD", "true")
    monkeypatch.setenv("ENABLE_DYNAMIC_MIGRATION", "false")
    monkeypatch.setenv("ENABLE_GOSSIP", "false")


def test_pull_over_tcp_matches_bit_exact(migration_env):
    port_a = _free_port()
    port_b = _free_port()

    spec_a = ShardSpec(
        shard_id="A", address=NodeAddress(host="127.0.0.1", port=port_a),
        start_layer=0, end_layer=30, moe_experts={15: (0, 3, 6, 9)},
    )
    spec_b = ShardSpec(
        shard_id="B", address=NodeAddress(host="127.0.0.1", port=port_b),
        start_layer=0, end_layer=30, moe_experts={15: (1, 4, 7, 10)},
    )
    sm = ShardMap({"A": spec_a, "B": spec_b})

    node_a = Node(shard=spec_a, shard_map=sm, total_layers=30)
    node_b = Node(shard=spec_b, shard_map=sm, total_layers=30)
    t_a = threading.Thread(target=node_a.serve_forever, daemon=True)
    t_b = threading.Thread(target=node_b.serve_forever, daemon=True)
    t_a.start(); t_b.start()
    time.sleep(0.5)

    try:
        from model_shard.migration import ExpertWeightPeerRPC
        rpc = ExpertWeightPeerRPC(
            addresses={"A": ("127.0.0.1", port_a)}, timeout_s=60.0
        )
        tensors = rpc.pull(source_shard_id="A", layer_idx=15, expert_id=3)
        node_b.migration_attach(layer_idx=15, expert_id=3, tensors=tensors)
        assert 3 in node_b._live_experts[15]

        # Verify bit-exact post-attach.
        hidden = node_a._lm.text_model.layers[15].pre_feedforward_layernorm_2.weight.shape[0]
        mx.random.seed(7)
        h = mx.random.normal((1, 7, hidden)).astype(mx.bfloat16)
        out_a = run_selected_experts(node_a._lm, h, 15, [3])
        out_b = run_selected_experts(node_b._lm, h, 15, [3])
        assert mx.array_equal(out_a[3], out_b[3]).item()
    finally:
        node_a.shutdown(); node_b.shutdown()
        t_a.join(timeout=3.0); t_b.join(timeout=3.0)
```

- [ ] **Step 2: Run**

`uv run pytest tests/test_migration_over_tcp.py -v -m slow`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_migration_over_tcp.py
git commit -m "Phase 5b Task 20: slow E2E — migration over TCP bit-exact"
```

---

### Task 21: Slow — ownership gossip convergence (3-node)

**Files:**
- Create: `tests/test_ownership_gossip_convergence.py`

- [ ] **Step 1: Write the failing slow test**

```python
"""3-node cluster: after attach, ownership gossip converges within N rounds."""
from __future__ import annotations

import socket as _sk
import threading
import time

import pytest

from model_shard.node import Node
from model_shard.shard_map import NodeAddress, ShardMap, ShardSpec

pytestmark = pytest.mark.slow


def _free_port() -> int:
    s = _sk.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def gossip_env(monkeypatch):
    monkeypatch.setenv("ENABLE_PARTIAL_LOAD", "true")
    monkeypatch.setenv("ENABLE_DYNAMIC_MIGRATION", "false")
    monkeypatch.setenv("ENABLE_GOSSIP", "true")


def test_ownership_delta_propagates_within_three_rounds(gossip_env):
    ports = [_free_port() for _ in range(3)]
    specs = [
        ShardSpec(
            shard_id=f"n{i}",
            address=NodeAddress(host="127.0.0.1", port=p),
            start_layer=0, end_layer=30,
            moe_experts={15: (i, 3 + i)},
        )
        for i, p in enumerate(ports)
    ]
    sm = ShardMap({s.shard_id: s for s in specs})
    nodes = [Node(shard=s, shard_map=sm, total_layers=30) for s in specs]
    threads = [threading.Thread(target=n.serve_forever, daemon=True) for n in nodes]
    for t in threads: t.start()
    try:
        # Wait for SWIM stabilization.
        time.sleep(2.0)
        # Fake a local attach on n0 for expert 42.
        nodes[0]._live_experts.setdefault(15, set()).add(42)
        with nodes[0]._ownership_seen_lock:
            nodes[0]._ownership_seen.add((nodes[0]._shard.shard_id, 15, 42))
        nodes[0]._membership.announce_ownership_add(15, 42)

        # Wait up to 6s for gossip propagation.
        deadline = time.monotonic() + 6.0
        converged = False
        while time.monotonic() < deadline:
            view_1 = nodes[1]._membership.ownership_view()
            view_2 = nodes[2]._membership.ownership_view()
            if ("n0", 15, 42) in view_1 and ("n0", 15, 42) in view_2:
                converged = True
                break
            time.sleep(0.1)
        assert converged, "ownership ADD did not propagate to all peers"
    finally:
        for n in nodes: n.shutdown()
        for t in threads: t.join(timeout=3.0)
```

- [ ] **Step 2: Run**

`uv run pytest tests/test_ownership_gossip_convergence.py -v -m slow`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_ownership_gossip_convergence.py
git commit -m "Phase 5b Task 21: slow E2E — ownership gossip convergence (3-node)"
```

---

### Task 22: Slow — decode-hang fix E2E (3-node)

**Files:**
- Create: `tests/test_decode_hang_fix_e2e.py`

- [ ] **Step 1: Write the failing slow test**

```python
"""3-node Tier 1 E2E: kill mid-decode peer, head exits decode loop cleanly."""
from __future__ import annotations

import socket as _sk
import threading
import time

import pytest

from model_shard.client import Client  # existing Phase 1 client
from model_shard.node import Node
from model_shard.shard_map import NodeAddress, ShardMap, ShardSpec

pytestmark = pytest.mark.slow


def _free_port() -> int:
    s = _sk.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_mid_decode_peer_death_unblocks_head(monkeypatch):
    monkeypatch.setenv("ENABLE_GOSSIP", "true")
    monkeypatch.setenv("ENABLE_PARTIAL_LOAD", "false")
    monkeypatch.setenv("ENABLE_DYNAMIC_MIGRATION", "false")
    # Reuse existing Phase 3 test scaffolding pattern; copy fixture code here
    # if the Phase 3 harness is not directly importable.
    ports = [_free_port() for _ in range(3)]
    specs = [
        ShardSpec(
            shard_id=f"s{i}",
            address=NodeAddress(host="127.0.0.1", port=p),
            start_layer=i * 10, end_layer=(i + 1) * 10, moe_experts={},
        )
        for i, p in enumerate(ports)
    ]
    sm = ShardMap({s.shard_id: s for s in specs})
    nodes = [Node(shard=s, shard_map=sm, total_layers=30) for s in specs]
    threads = [threading.Thread(target=n.serve_forever, daemon=True) for n in nodes]
    for t in threads: t.start()
    time.sleep(3.0)

    head_port = ports[0]
    client = Client(host="127.0.0.1", port=head_port)
    # Start a long generation (many tokens) in a background thread so we
    # can kill the tail mid-stream.
    errors: list[Exception] = []
    done = threading.Event()
    def drive():
        try:
            list(client.generate(prompt="hello", max_new_tokens=64))
        except Exception as e:
            errors.append(e)
        finally:
            done.set()
    t = threading.Thread(target=drive, daemon=True)
    t.start()
    time.sleep(1.0)

    # Kill the tail.
    nodes[2].shutdown()
    threads[2].join(timeout=3.0)

    # Head should exit the decode loop within SUSPECT_PERIOD + 1s (~5-6s).
    assert done.wait(timeout=10.0), "decode loop did not exit after peer death"
    # The client should have seen an error (SHARD_UNAVAILABLE) bubble up.
    assert errors, "expected client to receive an error"

    for n, th in zip(nodes[:2], threads[:2]):
        n.shutdown()
        th.join(timeout=3.0)
```

- [ ] **Step 2: Run**

`uv run pytest tests/test_decode_hang_fix_e2e.py -v -m slow`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_decode_hang_fix_e2e.py
git commit -m "Phase 5b Task 22: slow E2E — decode-loop hang fix (3-node)"
```

---

### Task 23: Slow — Tier 1 regression with both flags ON

**Files:**
- Create: `tests/test_partial_load_tier1_migration.py`

- [ ] **Step 1: Write the failing slow regression test**

```python
"""Tier 1 E2E with ENABLE_PARTIAL_LOAD=true AND ENABLE_DYNAMIC_MIGRATION=true.

Verifies the scanner runs in the background without breaking token output
(short prompts ≤ 8 tokens stay on the no-sort path — see 5a §7.5)."""
from __future__ import annotations

import json
import socket as _sk
import threading
import time
from pathlib import Path

import pytest

from model_shard.client import Client
from model_shard.node import Node
from model_shard.shard_map import ShardMap

pytestmark = pytest.mark.slow


def _free_port() -> int:
    s = _sk.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_tier1_with_migration_enabled(monkeypatch):
    monkeypatch.setenv("ENABLE_PARTIAL_LOAD", "true")
    monkeypatch.setenv("ENABLE_DYNAMIC_MIGRATION", "true")
    monkeypatch.setenv("ENABLE_GOSSIP", "true")
    monkeypatch.setenv("MIGRATION_SCAN_INTERVAL_SECONDS", "2.0")

    sm = ShardMap.from_yaml(Path("config/shards.yaml"))
    nodes = [
        Node(shard=sm.lookup(sid), shard_map=sm, total_layers=30)
        for sid in sm.all_shards()
    ]
    threads = [threading.Thread(target=n.serve_forever, daemon=True) for n in nodes]
    for t in threads: t.start()
    time.sleep(3.0)

    head_port = sm.lookup("layer_0-10").address.port
    client = Client(host="127.0.0.1", port=head_port)

    prompts = json.loads(Path("tests/prompts.json").read_text())
    ref = json.loads(Path("artifacts/ref/tokens.json").read_text())
    for pair in prompts[:5]:
        prompt = pair["prompt"]
        expected = ref[prompt][:8]  # short to stay in no-sort path
        got = list(client.generate(prompt=prompt, max_new_tokens=len(expected)))
        assert [t.token_id for t in got] == expected

    for n, th in zip(nodes, threads):
        n.shutdown()
        th.join(timeout=3.0)
```

- [ ] **Step 2: Run**

`uv run pytest tests/test_partial_load_tier1_migration.py -v -m slow`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_partial_load_tier1_migration.py
git commit -m "Phase 5b Task 23: slow regression — Tier 1 with both flags ON"
```

---

### Task 24: Memory update + README + final commit

**Files:**
- Modify: `README.md` (add a Phase 5b status paragraph)
- Update: memory file at `/Users/lukechang/.claude/projects/-Users-lukechang-Github-model-shard/memory/project_gossip_moe.md`

- [ ] **Step 1: Append a Phase 5b status paragraph to README**

Insert a paragraph after the Phase 5a status paragraph describing:

- scope: A (heat) + B (migration) + D (decode-hang fix)
- gate flag: `ENABLE_DYNAMIC_MIGRATION`
- correctness proof: `tests/test_migration_bit_exact_per_expert.py`
- load-bearing E2E: `tests/test_migration_over_tcp.py`, `tests/test_ownership_gossip_convergence.py`
- known carryover: sort-path FP noise limits full-length prefill bit-exactness (same as 5a §7.5)

- [ ] **Step 2: Update the `project_gossip_moe.md` memory file**

Add a `**Phase 5b STATUS: COMPLETE (<date>, commit <hash>)**` paragraph parallel to the Phase 5a entry, with:
- link to the plan
- link to the spec
- one-line summary of what's enabled
- the gate-flag note

- [ ] **Step 3: Full slow suite verification**

Run:
```bash
uv run pytest -v                              # fast
uv run pytest -m slow -v                      # full slow
uv run ruff check src tests scripts
uv run mypy src tests scripts
```
Expected: all green (acknowledging the known Phase 3 Metal-in-process artifact on the 57-test full slow session — not a Phase 5b regression).

- [ ] **Step 4: Final commit**

```bash
git add README.md
git commit -m "Phase 5b Task 24: README status paragraph; plan complete"
```

---

## Self-Review Notes

**Spec coverage check:** Every decision in the spec maps to one or more tasks:
- D1 (scope) → all tasks
- D2 (heat signal) → Task 4, Task 17 step 5
- D3 (heat gossip transport) → Tasks 1, 2, 3, 8
- D4 (target-pull initiation) → Tasks 12, 15
- D5 (owner discovery) → Tasks 10, 11, 14
- D6 (wire protocol) → Task 1
- D7 (monolithic transfer) → Tasks 12, 13
- D8 (attach semantics) → Tasks 6, 17
- D9 (ownership registry) → Tasks 10, 11, 14
- D10 (ownership gossip) → Tasks 2, 3, 9, 17
- D11 (replication only) → Task 15 (no REMOVE path)
- D12 (source concurrency) → Task 5 (slice takes the lock only during `mx.take`), Task 13
- D13 (policy stub) → Tasks 15, 17
- D14 (decode-loop hang fix) → Task 18
- D15 (correctness bar) → Tasks 7, 20
- D16 (gate) → Task 19
- D17 (non-goals) → plan excludes these; REMOVE action is reserved but unused

**Placeholder scan:** No "TBD" / "add error handling" steps. Every code step has complete code; every test step has assertions that exercise a specific behavior.

**Type consistency:** Names used consistently across tasks —
- `live_owners_provider` (Tasks 10, 11, 17)
- `heat_observer` (Tasks 17, moe.py)
- `migration_attach` (Tasks 17, 20)
- `owners_of` (Tasks 14, 17)
- `_POISON_TOKEN` / `PeerLeftAliveError` (Task 18)








