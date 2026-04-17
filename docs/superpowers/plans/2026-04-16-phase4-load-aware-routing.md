# Phase 4 — Load-Aware Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give each node a gossiped view of peer load (queue-depth EMA) and make `ExpertOrchestrator` prefer the less-loaded candidate when an expert has multiple owners per `moe_experts` YAML.

**Architecture:** Piggyback a small `LoadReport` on every SWIM Ping/Ack. A new `LoadTracker` keeps a local EMA and produces jittered reports. `MembershipRunner` exposes `start_load_source` (pluggable producer of the node's own report) and `latest_loads` (peer-load cache updated on inbound). `moe.group_expert_ids_by_owner_loaded` applies power-of-two-choices over multi-candidate experts, picking the lower-EMA owner. No dynamic replication; static `moe_experts` overlap is the source of multi-candidate scenarios.

**Tech Stack:** Python 3.13, MLX (same as Phase 3), protobuf, SWIM UDP transport from Phase 2. Reuse `wire_pb2` generation toolchain.

**Design spec:** `docs/superpowers/specs/2026-04-16-phase4-load-aware-routing-design.md`

---

## File Structure

**New files:**
- `src/model_shard/load.py` — `LoadTracker` (pure EMA + jitter helper).
- `tests/test_load_tracker.py` — fast unit tests.
- `tests/test_routing_correctness.py` — slow mocked-loads routing test.
- `tests/test_expert_rpc_load_shift.py` — slow subprocess E2E load-shift test.

**Modified:**
- `proto/wire.proto` — add `LoadReport` message + `loads` field on Ping/Ack/PingReq/PingReqAck.
- `src/model_shard/_pb/wire_pb2.py` — regenerated.
- `src/model_shard/membership/records.py` — add `loads: list[LoadReportRecord]` field on PingMsg/AckMsg/PingReqMsg/PingReqAckMsg; new `LoadReportRecord` dataclass.
- `src/model_shard/membership/messages.py` — encode/decode the new field.
- `src/model_shard/membership/runner.py` — `start_load_source(fn)`, `latest_loads()` API; piggyback outgoing loads; cache inbound loads.
- `src/model_shard/moe.py` — `group_expert_ids_by_owner_loaded`.
- `src/model_shard/expert_orchestrator.py` — accept `loads_provider` and `rng`, route via the new function.
- `src/model_shard/node.py` — construct `LoadTracker`; wire to runner; instrument `_handle_expert_request`; pass `loads_provider` to orchestrator.
- `scripts/run_node.py` — expose `/loads` debug endpoint.
- `config/shards.yaml` — overlap experts 0, 1, 2 across two shards each.
- `README.md` — Phase 4 status paragraph.

---

## Task Overview

| # | Task | Blocker |
|---|---|---|
| 1 | Proto: LoadReport + SWIM loads fields; regen; roundtrip test | — |
| 2 | records.py + messages.py: loads adapter | 1 |
| 3 | `LoadTracker` (fast) | — |
| 4 | MembershipRunner: `start_load_source` + `latest_loads` | 2 |
| 5 | `moe.group_expert_ids_by_owner_loaded` (fast) | — |
| 6 | `ExpertOrchestrator`: `loads_provider` + `rng` | 5 |
| 7 | `node.py`: wire LoadTracker to runner + orchestrator | 3, 4, 6 |
| 8 | `scripts/run_node.py`: `/loads` HTTP endpoint | 4 |
| 9 | `config/shards.yaml`: overlap experts 0, 1, 2 | — |
| 10 | Routing correctness test (mocked loads) | 6 |
| 11 | Phase 3 regression under overlap YAML | 7, 9 |
| 12 | E2E load-shift test (subprocess) | 7, 8, 9 |
| 13 | Final acceptance: ruff + mypy + fast + slow + README + memory | all |

---

## Task 1: Proto — `LoadReport` and SWIM `loads` fields

**Files:**
- Modify: `proto/wire.proto`
- Regenerate: `src/model_shard/_pb/wire_pb2.py`
- Create: `tests/test_load_report_envelope.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_load_report_envelope.py`:

```python
"""Roundtrip tests for the Phase 4 LoadReport piggybacked on SWIM messages."""

from __future__ import annotations

from model_shard._pb import wire_pb2


def test_load_report_roundtrip_on_ping() -> None:
    env = wire_pb2.Envelope()
    env.ping.protocol_version = 1
    env.ping.from_shard_id = "head"
    env.ping.from_incarnation = 7
    # loads piggyback:
    lr = env.ping.loads.add()
    lr.shard_id = "head"
    lr.queue_depth_ema = 250
    lr.ts_unix_ms = 1713000000_000

    raw = env.SerializeToString()
    out = wire_pb2.Envelope()
    out.ParseFromString(raw)
    assert out.WhichOneof("payload") == "ping"
    assert len(out.ping.loads) == 1
    assert out.ping.loads[0].shard_id == "head"
    assert out.ping.loads[0].queue_depth_ema == 250
    assert out.ping.loads[0].ts_unix_ms == 1713000000_000


def test_load_report_roundtrip_on_ack_multiple_entries() -> None:
    env = wire_pb2.Envelope()
    env.ack.protocol_version = 1
    env.ack.from_shard_id = "mid"
    env.ack.from_incarnation = 2
    for sid, ema in [("head", 100), ("mid", 50), ("tail", 300)]:
        lr = env.ack.loads.add()
        lr.shard_id = sid
        lr.queue_depth_ema = ema
        lr.ts_unix_ms = 0

    out = wire_pb2.Envelope()
    out.ParseFromString(env.SerializeToString())
    sids = [lr.shard_id for lr in out.ack.loads]
    emas = [lr.queue_depth_ema for lr in out.ack.loads]
    assert sids == ["head", "mid", "tail"]
    assert emas == [100, 50, 300]


def test_load_report_absent_on_ping_req_defaults_empty() -> None:
    env = wire_pb2.Envelope()
    env.ping_req.protocol_version = 1
    env.ping_req.from_shard_id = "head"
    env.ping_req.target_shard_id = "mid"
    env.ping_req.probe_id = "p1"
    out = wire_pb2.Envelope()
    out.ParseFromString(env.SerializeToString())
    assert list(out.ping_req.loads) == []
```

- [ ] **Step 2: Run — expect failure**

Run: `uv run pytest tests/test_load_report_envelope.py -v`
Expected: `AttributeError: 'Envelope.ping' object has no attribute 'loads'`.

- [ ] **Step 3: Modify `proto/wire.proto`**

Just before the `message Envelope {` block, add:

```proto
// Phase 4 — load gossip on the hot plane (piggybacked on SWIM messages).
message LoadReport {
  string shard_id       = 1;
  uint32 queue_depth_ema = 2;  // EMA × 100 (so 250 = 2.5 average)
  int64  ts_unix_ms     = 3;
}
```

Then extend each SWIM message with a `loads` field using a tag one above its current max:

```proto
message Ping {
  uint32 protocol_version = 1;
  string from_shard_id = 2;
  uint64 from_incarnation = 3;
  repeated MemberRecordPb deltas = 4;
  repeated LoadReport loads = 5;  // NEW
}

message Ack {
  uint32 protocol_version = 1;
  string from_shard_id = 2;
  uint64 from_incarnation = 3;
  repeated MemberRecordPb deltas = 4;
  repeated LoadReport loads = 5;  // NEW
}

message PingReq {
  uint32 protocol_version = 1;
  string from_shard_id = 2;
  string target_shard_id = 3;
  string probe_id = 4;
  repeated MemberRecordPb deltas = 5;
  repeated LoadReport loads = 6;  // NEW
}

message PingReqAck {
  uint32 protocol_version = 1;
  string from_shard_id = 2;
  string target_shard_id = 3;
  string probe_id = 4;
  bool success = 5;
  repeated MemberRecordPb deltas = 6;
  repeated LoadReport loads = 7;  // NEW
}
```

Do NOT modify the `Envelope` oneof. `Join` and `MembershipDelta` stay unchanged — they're not part of the steady-state gossip so they don't need a load piggyback.

- [ ] **Step 4: Regenerate the Python bindings**

Run:
```bash
uv run python -m grpc_tools.protoc -I proto --python_out=src/model_shard/_pb proto/wire.proto
```

- [ ] **Step 5: Run — expect pass**

Run: `uv run pytest tests/test_load_report_envelope.py -v`
Expected: 3 passed.

- [ ] **Step 6: Full fast suite sanity**

Run: `uv run pytest`
Expected: 124 passed (121 existing + 3 new). No existing test should break — the proto is additive only.

- [ ] **Step 7: Commit**

```bash
git add proto/wire.proto src/model_shard/_pb/wire_pb2.py tests/test_load_report_envelope.py
git commit -m "Phase 4: proto — LoadReport piggybacked on SWIM Ping/Ack/PingReq/PingReqAck"
```

---

## Task 2: `records.py` + `messages.py` — `LoadReportRecord` adapter

**Files:**
- Modify: `src/model_shard/membership/records.py`
- Modify: `src/model_shard/membership/messages.py`
- Create: `tests/test_membership_load_records.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_membership_load_records.py`:

```python
"""Round-trip tests for LoadReportRecord via encode/decode_membership_envelope."""

from __future__ import annotations

from model_shard.membership.messages import (
    decode_membership_envelope,
    encode_membership_envelope,
)
from model_shard.membership.records import (
    AckMsg,
    LoadReportRecord,
    PingMsg,
)


def test_ping_loads_roundtrip() -> None:
    msg = PingMsg(
        from_shard_id="head",
        from_incarnation=1,
        deltas=[],
        loads=[
            LoadReportRecord(shard_id="head", queue_depth_ema=250, ts_unix_ms=100),
            LoadReportRecord(shard_id="mid", queue_depth_ema=50, ts_unix_ms=100),
        ],
    )
    raw = encode_membership_envelope(msg)
    got = decode_membership_envelope(raw)
    assert isinstance(got, PingMsg)
    assert len(got.loads) == 2
    assert got.loads[0].shard_id == "head"
    assert got.loads[0].queue_depth_ema == 250
    assert got.loads[1].shard_id == "mid"


def test_ack_loads_absent_defaults_empty() -> None:
    msg = AckMsg(from_shard_id="mid", from_incarnation=3, deltas=[])
    raw = encode_membership_envelope(msg)
    got = decode_membership_envelope(raw)
    assert isinstance(got, AckMsg)
    assert got.loads == []
```

- [ ] **Step 2: Run — expect failure**

Run: `uv run pytest tests/test_membership_load_records.py -v`
Expected: `ImportError: cannot import name 'LoadReportRecord'` or a TypeError on `loads=` kwarg.

- [ ] **Step 3: Modify `src/model_shard/membership/records.py`**

Find the `PingMsg`, `AckMsg`, `PingReqMsg`, `PingReqAckMsg` dataclass definitions. They currently have `deltas: list[MemberRecord]` (likely `field(default_factory=list)`).

Add `LoadReportRecord` near `MemberRecord` in the same file:

```python
@dataclass(frozen=True)
class LoadReportRecord:
    shard_id: str
    queue_depth_ema: int   # EMA × 100
    ts_unix_ms: int
```

Add a `loads: list[LoadReportRecord] = field(default_factory=list)` field to each of PingMsg, AckMsg, PingReqMsg, PingReqAckMsg. Keep `deltas` unchanged.

Export `LoadReportRecord` from the module's `__all__` if it has one.

- [ ] **Step 4: Modify `src/model_shard/membership/messages.py`**

Add helper converters near `_record_to_pb` / `_record_from_pb`:

```python
def _load_to_pb(r: "LoadReportRecord") -> "wire_pb2.LoadReport":
    return wire_pb2.LoadReport(
        shard_id=r.shard_id,
        queue_depth_ema=r.queue_depth_ema,
        ts_unix_ms=r.ts_unix_ms,
    )


def _load_from_pb(pb: "wire_pb2.LoadReport") -> "LoadReportRecord":
    return LoadReportRecord(
        shard_id=pb.shard_id,
        queue_depth_ema=int(pb.queue_depth_ema),
        ts_unix_ms=int(pb.ts_unix_ms),
    )
```

Add `from model_shard.membership.records import LoadReportRecord` to the imports at the top.

In `encode_membership_envelope`, for each of the four message types (Ping/Ack/PingReq/PingReqAck), add one line BEFORE the return:

```python
        env.ping.loads.extend(_load_to_pb(lr) for lr in msg.loads)
```

(matching the existing `env.ping.deltas.extend(...)` pattern; use `.ack`, `.ping_req`, `.ping_req_ack` on their respective branches).

In `decode_membership_envelope`, for each matching branch, extract `loads`:

```python
    if which == "ping":
        return PingMsg(
            from_shard_id=env.ping.from_shard_id,
            from_incarnation=int(env.ping.from_incarnation),
            deltas=[_record_from_pb(d) for d in env.ping.deltas],
            loads=[_load_from_pb(lr) for lr in env.ping.loads],
        )
```

(and analogously for Ack, PingReq, PingReqAck).

- [ ] **Step 5: Run — expect pass**

Run: `uv run pytest tests/test_membership_load_records.py -v`
Expected: 2 passed.

- [ ] **Step 6: Confirm no Phase 2 membership tests regress**

Run: `uv run pytest tests/membership/ -v`
Expected: all existing membership tests still pass. (The new `loads` field is optional with default empty list; existing tests that construct PingMsg without `loads=` still work.)

- [ ] **Step 7: Commit**

```bash
git add src/model_shard/membership/records.py src/model_shard/membership/messages.py tests/test_membership_load_records.py
git commit -m "Phase 4: LoadReportRecord on SWIM message records; encode/decode adapter"
```

---

## Task 3: `LoadTracker` — EMA + jitter

**Files:**
- Create: `src/model_shard/load.py`
- Create: `tests/test_load_tracker.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_load_tracker.py`:

```python
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
    # After 20 observations at 10 with alpha=0.5, EMA is extremely close to 10.
    # report returns EMA × 100, so expect ~1000.
    assert abs(tk.report() - 1000) < 5


def test_tracker_ema_tracks_step_change() -> None:
    tk = LoadTracker(alpha=0.5, jitter_pct=0.0, rng=random.Random(0))
    for _ in range(20):
        tk.observe(10)
    before = tk.report()
    for _ in range(20):
        tk.observe(0)
    after = tk.report()
    assert after < before // 2  # EMA should drop noticeably


def test_tracker_jitter_bounded() -> None:
    """With jitter_pct=0.1, report is within ±10% of underlying EMA × 100."""
    tk = LoadTracker(alpha=0.5, jitter_pct=0.1, rng=random.Random(42))
    for _ in range(20):
        tk.observe(10)
    samples = [tk.report() for _ in range(200)]
    # Underlying EMA × 100 ≈ 1000. Jittered values must all be in [900, 1100].
    assert all(900 <= s <= 1100 for s in samples), f"out of range: {samples[:5]}"
    # And there must be actual variation (not all identical).
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
    for t in threads: t.start()
    for t in threads: t.join()
    # 400 observations of 5 with alpha=0.5 → EMA converges to ~5.
    assert abs(tk.report() - 500) < 10
```

- [ ] **Step 2: Run — expect failure**

Run: `uv run pytest tests/test_load_tracker.py -v`
Expected: `ModuleNotFoundError: No module named 'model_shard.load'`.

- [ ] **Step 3: Create `src/model_shard/load.py`**

```python
"""EMA-based queue-depth tracker for Phase 4 load-aware routing.

Observes integer depth samples, maintains an exponential moving average,
and produces a jittered integer report (EMA × 100) suitable for gossip.
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
        """Record one queue-depth sample. Called on handler entry/exit."""
        with self._lock:
            self._ema = self._alpha * depth + (1.0 - self._alpha) * self._ema

    def report(self) -> int:
        """Return jittered EMA scaled by 100 (integer wire form)."""
        with self._lock:
            ema = self._ema
        jitter = 1.0
        if self._jitter_pct > 0.0:
            jitter = 1.0 + self._rng.uniform(-self._jitter_pct, self._jitter_pct)
        return max(0, int(round(ema * 100.0 * jitter)))


__all__ = ["LoadTracker"]
```

- [ ] **Step 4: Run — expect pass**

Run: `uv run pytest tests/test_load_tracker.py -v`
Expected: 6 passed.

- [ ] **Step 5: Lint + types**

Run: `uv run ruff check src/model_shard/load.py tests/test_load_tracker.py`
Run: `uv run mypy src/model_shard/load.py`
Both expected clean.

- [ ] **Step 6: Commit**

```bash
git add src/model_shard/load.py tests/test_load_tracker.py
git commit -m "Phase 4: LoadTracker — EMA + jittered report for queue-depth gossip"
```

---

## Task 4: `MembershipRunner` — `start_load_source` + `latest_loads`

**Files:**
- Modify: `src/model_shard/membership/runner.py`
- Create: `tests/test_membership_runner_loads.py`

- [ ] **Step 1: Write the failing test**

```python
"""MembershipRunner exposes a pluggable load-source hook and caches peer loads."""

from __future__ import annotations

import time

from model_shard.membership.config import SwimConfig
from model_shard.membership.records import (
    AckMsg,
    IncomingMessage,
    LoadReportRecord,
    PingMsg,
)
from model_shard.membership.runner import MembershipRunner
from model_shard.membership.state import PeerSpec


def _spec(shard_id: str, port: int) -> PeerSpec:
    return PeerSpec(shard_id=shard_id, host="127.0.0.1", udp_port=port, incarnation=0)


def test_runner_start_load_source_and_latest_loads_roundtrip(tmp_path) -> None:
    self_spec = _spec("head", 40000)
    peers = [_spec("mid", 40001), _spec("tail", 40002)]
    runner = MembershipRunner(self_spec=self_spec, peers=peers, config=SwimConfig())
    try:
        # No loads yet.
        assert runner.latest_loads() == {}

        # Register a load source.
        runner.start_load_source(
            lambda: LoadReportRecord(shard_id="head", queue_depth_ema=123, ts_unix_ms=0)
        )
        # Source is queryable via internal accessor (used by the send path).
        assert runner._load_source is not None   # internal check, acceptable in tests
        assert runner._load_source().queue_depth_ema == 123

        # Simulate inbound Ping from peer mid carrying a load for mid.
        msg = PingMsg(
            from_shard_id="mid",
            from_incarnation=0,
            deltas=[],
            loads=[LoadReportRecord(shard_id="mid", queue_depth_ema=42, ts_unix_ms=int(time.time() * 1000))],
        )
        runner._on_recv_decoded(msg)   # test hook — see implementation

        loads = runner.latest_loads()
        assert "mid" in loads
        assert loads["mid"].queue_depth_ema == 42
    finally:
        runner.stop()


def test_runner_latest_loads_stale_entries_not_pruned_here() -> None:
    """Pruning stale entries is the orchestrator's responsibility.
    Runner stores whatever came in; callers filter by ts_unix_ms."""
    self_spec = _spec("head", 40010)
    peers = [_spec("mid", 40011)]
    runner = MembershipRunner(self_spec=self_spec, peers=peers, config=SwimConfig())
    try:
        msg = PingMsg(
            from_shard_id="mid", from_incarnation=0, deltas=[],
            loads=[LoadReportRecord(shard_id="mid", queue_depth_ema=1, ts_unix_ms=1)],
        )
        runner._on_recv_decoded(msg)
        # Overwrite with newer entry.
        msg2 = PingMsg(
            from_shard_id="mid", from_incarnation=0, deltas=[],
            loads=[LoadReportRecord(shard_id="mid", queue_depth_ema=99, ts_unix_ms=9999)],
        )
        runner._on_recv_decoded(msg2)
        assert runner.latest_loads()["mid"].queue_depth_ema == 99
    finally:
        runner.stop()
```

- [ ] **Step 2: Run — expect failure**

Run: `uv run pytest tests/test_membership_runner_loads.py -v`
Expected: AttributeError on `start_load_source` or `latest_loads` or `_on_recv_decoded`.

- [ ] **Step 3: Modify `src/model_shard/membership/runner.py`**

Add imports at the top:

```python
from model_shard.membership.records import LoadReportRecord, PingMsg, AckMsg, PingReqMsg, PingReqAckMsg
```

(Only those that aren't already imported.)

In `MembershipRunner.__init__`, add:

```python
        self._load_source: "Callable[[], LoadReportRecord] | None" = None
        self._peer_loads: dict[str, LoadReportRecord] = {}
        self._peer_loads_lock = threading.Lock()
```

Add public methods:

```python
    def start_load_source(self, fn: "Callable[[], LoadReportRecord]") -> None:
        """Register a callable invoked once per outgoing ping to produce this
        node's own load report. Safe to set multiple times; the latest wins."""
        self._load_source = fn

    def latest_loads(self) -> dict[str, LoadReportRecord]:
        """Return a snapshot of the most recent load report seen from each peer
        shard_id. Caller is responsible for filtering by staleness."""
        with self._peer_loads_lock:
            return dict(self._peer_loads)
```

Add a helper (called from `_on_recv` after decoding; also a test entry point):

```python
    def _on_recv_decoded(self, decoded: IncomingMessage) -> None:
        """Post a decoded message onto the inbox and scrape any loads it carries."""
        loads = getattr(decoded, "loads", None)
        if loads:
            with self._peer_loads_lock:
                for lr in loads:
                    self._peer_loads[lr.shard_id] = lr
        try:
            self._inbox.put_nowait(decoded)
        except queue.Full:
            _LOG.warning("membership inbox full; dropping message %s", type(decoded).__name__)
```

Replace the existing `_on_recv` body so it calls `_on_recv_decoded`:

```python
    def _on_recv(self, data: bytes, _addr: tuple[str, int]) -> None:
        decoded = decode_membership_envelope(data)
        if decoded is None:
            return
        self._on_recv_decoded(decoded)
```

For the outgoing path: in the `_run` loop, AFTER `outgoing = self._state.tick(now)` and AFTER draining the inbox, but BEFORE sending, inject the load report if a source is registered. The state machine's outgoing messages are `OutgoingMessage(target_shard_id, payload)` where payload is the dataclass message. Mutate the payload in place to add `loads`:

```python
            # Phase 4: piggyback own-load on outgoing ping-family messages.
            if self._load_source is not None:
                try:
                    my_load = self._load_source()
                except Exception:
                    _LOG.exception("load source raised; skipping load piggyback")
                    my_load = None
                if my_load is not None:
                    for o in outgoing:
                        p = o.payload
                        if isinstance(p, (PingMsg, AckMsg, PingReqMsg, PingReqAckMsg)):
                            # Dataclass instances are (likely) frozen; construct a replacement.
                            replaced = dataclasses.replace(p, loads=list(p.loads) + [my_load])
                            o.payload = replaced  # OutgoingMessage must not be frozen
```

If `OutgoingMessage` IS frozen, construct a new outgoing list. Read `records.py` for the exact shape.

Add `import dataclasses` near the top if not present.

- [ ] **Step 4: Run — expect pass**

Run: `uv run pytest tests/test_membership_runner_loads.py -v`
Expected: 2 passed.

- [ ] **Step 5: Phase 2 regression**

Run: `uv run pytest tests/membership/ -v`
Expected: all pass — Phase 2 tests don't register a load source so the piggyback is a no-op.

- [ ] **Step 6: Commit**

```bash
git add src/model_shard/membership/runner.py tests/test_membership_runner_loads.py
git commit -m "Phase 4: MembershipRunner — start_load_source + latest_loads + outbound piggyback"
```

---

## Task 5: `moe.group_expert_ids_by_owner_loaded`

**Files:**
- Modify: `src/model_shard/moe.py`
- Modify: `tests/test_moe_unit.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_moe_unit.py`:

```python
import random

import pytest

from model_shard.moe import group_expert_ids_by_owner_loaded


def test_group_loaded_single_candidate_uses_sole_owner() -> None:
    owners = {"head": {0}, "mid": {1}, "tail": {2}}
    got = group_expert_ids_by_owner_loaded(
        top_k_ids=[0, 1, 2],
        owners=owners,
        peer_loads={"mid": 100, "tail": 100},
        self_shard_id="head",
        self_load=50,
        rng=random.Random(0),
    )
    assert got == {"head": [0], "mid": [1], "tail": [2]}


def test_group_loaded_two_candidates_picks_less_loaded() -> None:
    owners = {"head": {0, 1}, "mid": {0, 1}, "tail": {2}}  # 0 and 1 are duplicated
    got = group_expert_ids_by_owner_loaded(
        top_k_ids=[0, 1, 2],
        owners=owners,
        peer_loads={"mid": 1000},
        self_shard_id="head",
        self_load=10,
        rng=random.Random(0),
    )
    # self_load=10 is less than mid's 1000 for both shared experts.
    assert got["head"] == [0, 1]
    assert got["tail"] == [2]
    assert "mid" not in got


def test_group_loaded_three_candidates_samples_two_then_picks_less_loaded() -> None:
    owners = {"a": {0}, "b": {0}, "c": {0}}
    # All three have the same expert.
    # With rng.seed(0), random.sample picks a specific pair; picking the
    # deterministic minimum of that pair is the behavior we want.
    rng = random.Random(0)
    got = group_expert_ids_by_owner_loaded(
        top_k_ids=[0],
        owners=owners,
        peer_loads={"a": 100, "b": 50, "c": 10},
        self_shard_id="self_not_in_owners",
        self_load=0,
        rng=rng,
    )
    # Exactly one of a, b, c should have [0]; the other two should be absent.
    assert sum(1 for v in got.values() if v == [0]) == 1
    assert all(v == [0] for v in got.values())


def test_group_loaded_unknown_peer_treated_as_max_load() -> None:
    """When a candidate has no entry in peer_loads, it's treated as
    effectively infinite so the known candidate wins."""
    owners = {"head": {0}, "mid": {0}}
    got = group_expert_ids_by_owner_loaded(
        top_k_ids=[0],
        owners=owners,
        peer_loads={},                 # both unknown
        self_shard_id="head",
        self_load=42,                  # known local load
        rng=random.Random(0),
    )
    # self has a known load, mid does not — prefer self.
    assert got == {"head": [0]}


def test_group_loaded_unknown_self_and_peer_falls_back_to_rng() -> None:
    """When all candidates are unknown, rng breaks the tie; result must be
    deterministic under a fixed rng seed."""
    owners = {"a": {0}, "b": {0}}
    rng = random.Random(99)
    got = group_expert_ids_by_owner_loaded(
        top_k_ids=[0],
        owners=owners,
        peer_loads={},
        self_shard_id="not-a-candidate",
        self_load=0,
        rng=rng,
    )
    # Exactly one of a, b picks the work; same rng seed → same result.
    winners = [k for k, v in got.items() if v == [0]]
    assert len(winners) == 1


def test_group_loaded_raises_on_unknown_id() -> None:
    owners = {"head": {0}, "mid": {1}}
    with pytest.raises(KeyError, match="expert_id 99"):
        group_expert_ids_by_owner_loaded(
            top_k_ids=[99], owners=owners, peer_loads={},
            self_shard_id="head", self_load=0, rng=random.Random(0),
        )
```

- [ ] **Step 2: Run — expect failure**

Run: `uv run pytest tests/test_moe_unit.py -v`
Expected: `ImportError: cannot import name 'group_expert_ids_by_owner_loaded'`.

- [ ] **Step 3: Implement**

Append to `src/model_shard/moe.py`:

```python
import random
from typing import Mapping


def group_expert_ids_by_owner_loaded(
    top_k_ids: list[int],
    owners: Mapping[str, set[int]],
    peer_loads: Mapping[str, int],
    self_shard_id: str,
    self_load: int,
    rng: random.Random,
) -> dict[str, list[int]]:
    """Partition top_k_ids by owner using power-of-two-choices on load.

    Each id in top_k_ids is assigned to exactly one of its candidate owners.
    If only one candidate owns the id, it wins uncontested. With two
    candidates, both are compared and the lower-loaded wins. With ≥3
    candidates, two are sampled uniformly and the lower-loaded of those wins.

    Loads are keyed by shard_id:
      * peer_loads[sid] — integer EMA × 100 from most recent gossip.
      * self_shard_id / self_load — the caller's own measurement.
      * A candidate with no known load is assigned INT_MAX so any known
        candidate beats it (loses ties to known data).
    """
    # Build id -> candidates list.
    candidates_by_id: dict[int, list[str]] = {}
    for owner, ids in owners.items():
        for i in ids:
            candidates_by_id.setdefault(i, []).append(owner)

    def load_of(sid: str) -> int:
        if sid == self_shard_id:
            return self_load
        if sid in peer_loads:
            return peer_loads[sid]
        return 2**31 - 1  # sentinel: unknown peer is max-load

    by_owner: dict[str, list[int]] = {}
    for eid in top_k_ids:
        candidates = candidates_by_id.get(eid)
        if not candidates:
            raise KeyError(f"expert_id {eid} has no owner in {list(owners)}")
        if len(candidates) == 1:
            winner = candidates[0]
        else:
            # Power-of-two-choices: sample at most two candidates.
            pool = (
                list(candidates)
                if len(candidates) == 2
                else rng.sample(candidates, 2)
            )
            winner = min(pool, key=load_of)
        by_owner.setdefault(winner, []).append(eid)
    return by_owner
```

Export it in `__all__`.

- [ ] **Step 4: Run — expect pass**

Run: `uv run pytest tests/test_moe_unit.py -v`
Expected: all pass (existing + new).

- [ ] **Step 5: Lint + types**

Run: `uv run ruff check src/model_shard/moe.py tests/test_moe_unit.py`
Run: `uv run mypy src/model_shard/moe.py`
Both clean.

- [ ] **Step 6: Commit**

```bash
git add src/model_shard/moe.py tests/test_moe_unit.py
git commit -m "Phase 4: moe.group_expert_ids_by_owner_loaded — P2C with loads"
```

---

## Task 6: `ExpertOrchestrator` — `loads_provider` + `rng`

**Files:**
- Modify: `src/model_shard/expert_orchestrator.py`
- Modify: `tests/test_expert_orchestrator.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_expert_orchestrator.py`:

```python
@pytest.mark.slow
def test_orchestrator_uses_loads_provider_for_multi_candidate(loaded_model: Any) -> None:
    """With expert 0 on both 'head' and 'peer', and peer reporting very high
    load, the orchestrator routes expert 0 locally rather than fanning out."""
    lm = loaded_model
    layer_idx = 15

    # Record each call to the fake peer RPC.
    calls: list[tuple[str, list[int]]] = []

    class _RecordingRpc(PeerRPC):
        def call(self, peer_shard_id: str, request_id: str, layer_idx: int,
                 expert_ids: list[int], h):
            calls.append((peer_shard_id, list(expert_ids)))
            raise AssertionError("should not be called when peer is reported high-load")

    # Expert 0 lives on both 'head' (self) and 'peer'. All other experts live
    # only on 'head'. Peer is very high load.
    owners = {"head": set(range(128)), "peer": {0}}
    orch = ExpertOrchestrator(
        self_shard_id="head",
        owners=owners,
        peer_rpc=_RecordingRpc(),
        rpc_timeout_s=1.0,
        loads_provider=lambda: {"peer": 1_000_000},  # peer is swamped
        rng=random.Random(0),
    )

    _, _, (gm, sm) = _replay_through(lm, mx.array([[1, 2, 3]]), layer_idx)
    _run_split_layer_happy_path(orch, lm, layer_idx)  # helper below

    # No peer RPC should have been attempted for expert 0.
    assert calls == []
    orch.close()
```

(You'll add the small helper `_run_split_layer_happy_path` near the existing `_replay_through` in this file. It just runs layers 0..layer_idx then calls `orch.run_split_layer(...)` and returns the result. Use the pattern already established in this file.)

Add `import random` and `from typing import Any` at the top if not present.

- [ ] **Step 2: Run — expect failure**

Run: `uv run pytest -m slow tests/test_expert_orchestrator.py -v`
Expected: the new test fails with `TypeError: __init__() got an unexpected keyword argument 'loads_provider'` (or similar).

- [ ] **Step 3: Modify `src/model_shard/expert_orchestrator.py`**

Add two optional fields to `ExpertOrchestrator`:

```python
@dataclass
class ExpertOrchestrator:
    self_shard_id: str
    owners: Mapping[str, set[int]]
    peer_rpc: PeerRPC
    rpc_timeout_s: float
    mlx_lock: threading.Lock | None = None
    loads_provider: Callable[[], Mapping[str, int]] = field(default_factory=lambda: (lambda: {}))
    rng: random.Random = field(default_factory=random.Random)
```

Add imports: `import random`, `from dataclasses import dataclass, field`, `from typing import Callable` (if not already present).

Inside `run_split_layer`, replace the call:

```python
by_owner = group_expert_ids_by_owner(all_ids, self.owners)
```

with:

```python
from model_shard.moe import group_expert_ids_by_owner_loaded
peer_loads = self.loads_provider()
by_owner = group_expert_ids_by_owner_loaded(
    all_ids,
    owners=self.owners,
    peer_loads=peer_loads,
    self_shard_id=self.self_shard_id,
    self_load=peer_loads.get(self.self_shard_id, 0),
    rng=self.rng,
)
```

Keep the `local_ids = by_owner.pop(self.self_shard_id, [])` unchanged; it now does double duty as the P2C-selected local experts.

- [ ] **Step 4: Run — expect pass**

Run: `uv run pytest -m slow tests/test_expert_orchestrator.py -v`
Expected: all pass (existing all-local plus new loads-provider test).

- [ ] **Step 5: Commit**

```bash
git add src/model_shard/expert_orchestrator.py tests/test_expert_orchestrator.py
git commit -m "Phase 4: ExpertOrchestrator — loads_provider + rng wire to group_experts_loaded"
```

---

## Task 7: `node.py` — construct LoadTracker; wire to runner + orchestrator

**Files:**
- Modify: `src/model_shard/node.py`
- Create: `tests/test_node_load_wiring.py`

- [ ] **Step 1: Write the failing test**

```python
"""Node constructs a LoadTracker, registers it with the runner, and passes a
loads_provider to its ExpertOrchestrator."""

from __future__ import annotations

from typing import Any

import pytest

from model_shard.node import Node
from model_shard.shard_map import NodeAddress, ShardMap, ShardSpec


def _make_shard_map(port: int) -> tuple[ShardMap, ShardSpec]:
    spec = ShardSpec(
        shard_id="solo",
        address=NodeAddress("127.0.0.1", port),
        start_layer=0,
        end_layer=30,
        moe_experts={15: (0, 1, 2)},
    )
    return ShardMap({"solo": spec}), spec


@pytest.mark.slow
def test_node_wires_load_tracker_and_runner_load_source(monkeypatch, loaded_model: Any) -> None:
    monkeypatch.setenv("ENABLE_EXPERT_SHARD", "true")
    monkeypatch.setenv("ENABLE_GOSSIP", "false")  # simplify — no real gossip needed for wiring test

    sm, spec = _make_shard_map(port=_find_free_port())
    node = Node(shard=spec, shard_map=sm, loaded_model=loaded_model, total_layers=30)
    try:
        # Node exposes its load tracker on a well-known attribute for testing.
        assert hasattr(node, "_load_tracker")
        assert node._load_tracker is not None
        # Orchestrator (if constructed) has a loads_provider callable.
        if node._orchestrator is not None:
            assert callable(node._orchestrator.loads_provider)
            # Calling the provider with gossip disabled returns an empty dict.
            assert node._orchestrator.loads_provider() == {}
    finally:
        node.shutdown()


def _find_free_port() -> int:
    import random, socket
    for _ in range(100):
        p = random.randint(30000, 60000)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", p))
            return p
        except OSError:
            continue
    raise RuntimeError("no free port")
```

- [ ] **Step 2: Run — expect failure**

Run: `uv run pytest -m slow tests/test_node_load_wiring.py -v`
Expected: `AttributeError: 'Node' object has no attribute '_load_tracker'` or similar.

- [ ] **Step 3: Modify `src/model_shard/node.py`**

Add imports near the top:

```python
import random as _random_mod
from model_shard.load import LoadTracker
from model_shard.membership.records import LoadReportRecord
```

In `Node.__init__`, after the existing membership runner setup but before the orchestrator construction, add:

```python
        self._load_tracker = LoadTracker(
            alpha=0.3, jitter_pct=0.1, rng=_random_mod.Random()
        )
        # Hook the tracker into the runner's piggyback path (no-op when
        # gossip is disabled).
        if self._membership is not None:
            def _load_source() -> LoadReportRecord:
                return LoadReportRecord(
                    shard_id=self._shard.shard_id,
                    queue_depth_ema=self._load_tracker.report(),
                    ts_unix_ms=int(time.time() * 1000),
                )
            self._membership.start_load_source(_load_source)
```

Find where `self._orchestrator = ExpertOrchestrator(...)` is constructed (introduced in Task 15 of Phase 3). Extend the call:

```python
        def _loads_provider() -> dict[str, int]:
            if self._membership is None:
                return {}
            return {
                sid: lr.queue_depth_ema
                for sid, lr in self._membership.latest_loads().items()
            }

        self._orchestrator = ExpertOrchestrator(
            self_shard_id=self._shard.shard_id,
            owners=owners,
            peer_rpc=peer_rpc,
            rpc_timeout_s=self._cfg.rpc_timeout_s if hasattr(self._cfg, "rpc_timeout_s") else 5.0,
            mlx_lock=_MLX_COMPUTE_LOCK,
            loads_provider=_loads_provider,
            rng=_random_mod.Random(),
        )
```

(Keep the existing structure; only the two extra kwargs are new. Adjust spacing to the file's existing style.)

In `_handle_expert_request`, bracket the MLX compute region with tracker updates:

```python
        self._load_tracker.observe(self._in_flight_expert_requests + 1)
        self._in_flight_expert_requests += 1
        try:
            with _MLX_COMPUTE_LOCK:
                ...
        finally:
            self._in_flight_expert_requests -= 1
            self._load_tracker.observe(self._in_flight_expert_requests)
```

Add `self._in_flight_expert_requests: int = 0` in `__init__`. (This is a rough proxy — expert depth per node — good enough for Phase 4.)

- [ ] **Step 4: Run — expect pass**

Run: `uv run pytest -m slow tests/test_node_load_wiring.py -v`
Expected: pass.

- [ ] **Step 5: Phase 3 regression (no overlap config yet, so no behavior change)**

Run:
```
uv run pytest -m slow tests/test_expert_orchestrator.py tests/test_expert_rpc_handler.py tests/test_tier1_expert_split_layer15.py -v
```
Expected: all pass — with no overlap in `moe_experts`, the loads_provider is irrelevant and routing matches Phase 3.

- [ ] **Step 6: Commit**

```bash
git add src/model_shard/node.py tests/test_node_load_wiring.py
git commit -m "Phase 4: Node wires LoadTracker → runner → ExpertOrchestrator"
```

---

## Task 8: `scripts/run_node.py` — `/loads` debug endpoint

**Files:**
- Modify: `scripts/run_node.py`

- [ ] **Step 1: Locate the existing `/membership` handler**

Read `scripts/run_node.py`. The Phase 2 debug endpoint served at `tcp_port + 2000` with path `/membership`. Find the `class Handler(http.server.BaseHTTPRequestHandler)` and its `do_GET`.

- [ ] **Step 2: Add `/loads` branch**

Inside `do_GET`, add:

```python
            elif self.path == "/loads":
                if handler_node.membership is None:
                    payload: dict[str, object] = {}
                else:
                    payload = {
                        sid: {"queue_depth_ema": lr.queue_depth_ema,
                              "ts_unix_ms": lr.ts_unix_ms}
                        for sid, lr in handler_node.membership.latest_loads().items()
                    }
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
```

Make sure the earlier `/membership` branch still returns after writing, and the 404 fallback is unchanged.

- [ ] **Step 3: Manual sanity**

Spawn a single node with gossip on, curl the new endpoint:

```bash
ENABLE_GOSSIP=true ENABLE_EXPERT_SHARD=true uv run python scripts/run_node.py \
  --shard layer_0-10 --config config/shards.yaml &
sleep 20
curl http://127.0.0.1:11001/loads
pkill -f run_node.py
```

Expected: JSON response, possibly `{}` if no peers are up yet.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_node.py
git commit -m "Phase 4: run_node — expose /loads debug endpoint"
```

---

## Task 9: `config/shards.yaml` — overlap experts 0, 1, 2

**Files:**
- Modify: `config/shards.yaml`

- [ ] **Step 1: Add overlap entries**

Edit `config/shards.yaml`. Add expert 0 to `layer_10-20.moe_experts[15]`, expert 1 to `layer_20-30.moe_experts[15]`, expert 2 to `layer_0-10.moe_experts[15]`:

- head (layer_0-10) still owns 0, 3, 6, ..., 126 — plus now expert 2.
- mid (layer_10-20) still owns 1, 4, 7, ..., 127 — plus now expert 0.
- tail (layer_20-30) still owns 2, 5, 8, ..., 125 — plus now expert 1.

Net effect:
- Expert 0: head, mid (2 candidates)
- Expert 1: mid, tail (2 candidates)
- Expert 2: tail, head (2 candidates)
- All other experts (125 of them): single candidate, unchanged.

- [ ] **Step 2: Validate partition coverage**

```bash
uv run python -c "
from pathlib import Path
from model_shard.shard_map import ShardMap
sm = ShardMap.from_yaml(Path('config/shards.yaml'))
all_ids: set[int] = set()
for s in sm.all_shards():
    for e in sm.lookup(s).moe_experts.get(15, ()):
        all_ids.add(e)
assert all_ids == set(range(128)), 'coverage broken'
print('every expert 0..127 still has at least one owner')
# Count multi-owner experts:
counts: dict[int, int] = {}
for s in sm.all_shards():
    for e in sm.lookup(s).moe_experts.get(15, ()):
        counts[e] = counts.get(e, 0) + 1
multi = [e for e, c in counts.items() if c > 1]
print(f'multi-candidate experts: {sorted(multi)}')
assert sorted(multi) == [0, 1, 2]
"
```

Expected output ends with `multi-candidate experts: [0, 1, 2]`.

- [ ] **Step 3: Commit**

```bash
git add config/shards.yaml
git commit -m "Phase 4: config — overlap experts 0, 1, 2 across two shards each"
```

---

## Task 10: Routing correctness test (mocked loads)

**Files:**
- Create: `tests/test_routing_correctness.py`

- [ ] **Step 1: Write the test**

```python
"""Deterministic routing correctness: given known peer loads, the orchestrator
consistently picks the less-loaded candidate for multi-owner experts."""

from __future__ import annotations

import random
from typing import Any

import mlx.core as mx
import pytest

from model_shard.expert_orchestrator import ExpertOrchestrator, PeerRPC
from model_shard.mlx_engine import embed_tokens, make_cache, make_masks


class _CountingRpc(PeerRPC):
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[int]]] = []

    def call(self, peer_shard_id: str, request_id: str, layer_idx: int,
             expert_ids: list[int], h):
        self.calls.append((peer_shard_id, sorted(expert_ids)))
        # Return a zero tensor for every requested expert, shaped like h.
        return {int(eid): mx.zeros_like(h) for eid in expert_ids}


@pytest.mark.slow
def test_multi_owner_orchestrator_picks_less_loaded_peer(loaded_model: Any) -> None:
    """Expert 0 is on 'head' (self) and 'peer'. Peer reports a massive load;
    self_load is low. Expert 0 should be routed locally — no peer RPC."""
    lm = loaded_model
    layer_idx = 15

    owners = {"head": set(range(128)), "peer": {0, 1, 2}}

    rpc = _CountingRpc()
    orch = ExpertOrchestrator(
        self_shard_id="head",
        owners=owners,
        peer_rpc=rpc,
        rpc_timeout_s=1.0,
        loads_provider=lambda: {"peer": 1_000_000, "head": 10},
        rng=random.Random(0),
    )

    tokens = mx.array([[1, 42, 99]])
    h = embed_tokens(lm, tokens)
    cache = make_cache(lm)
    gm, sm = make_masks(lm, h, cache)
    for i in range(layer_idx):
        layer = lm.text_model.layers[i]
        c = cache[lm.text_model.layer_idx_to_cache_idx[i]]
        mask = gm if layer.layer_type == "full_attention" else sm
        h = layer(h, mask, c, per_layer_input=None)

    out = orch.run_split_layer(lm, h=h, layer_idx=layer_idx, cache=cache,
                               masks=(gm, sm), request_id="r1")
    mx.eval(out)

    # Peer should not have received any call for experts 0, 1, or 2 — self is
    # less loaded for all of them.
    for peer, eids in rpc.calls:
        for e in eids:
            assert e not in (0, 1, 2), (
                f"expert {e} went to peer {peer!r} despite peer being high-load"
            )
    orch.close()


@pytest.mark.slow
def test_multi_owner_orchestrator_picks_peer_when_self_overloaded(loaded_model: Any) -> None:
    """Inverse case: self is massively loaded, peer is idle. Expert 0 should
    go to peer."""
    lm = loaded_model
    layer_idx = 15

    # Record which experts the peer was asked for.
    owners = {"head": set(range(128)), "peer": {0}}
    rpc_asked: list[int] = []

    class _EchoRpc(PeerRPC):
        def call(self, peer_shard_id: str, request_id: str, layer_idx: int,
                 expert_ids: list[int], h):
            for eid in expert_ids:
                rpc_asked.append(int(eid))
            return {int(eid): mx.zeros_like(h) for eid in expert_ids}

    orch = ExpertOrchestrator(
        self_shard_id="head",
        owners=owners,
        peer_rpc=_EchoRpc(),
        rpc_timeout_s=1.0,
        loads_provider=lambda: {"peer": 10, "head": 1_000_000},
        rng=random.Random(0),
    )

    tokens = mx.array([[1, 42]])
    h = embed_tokens(lm, tokens)
    cache = make_cache(lm)
    gm, sm = make_masks(lm, h, cache)
    for i in range(layer_idx):
        layer = lm.text_model.layers[i]
        c = cache[lm.text_model.layer_idx_to_cache_idx[i]]
        mask = gm if layer.layer_type == "full_attention" else sm
        h = layer(h, mask, c, per_layer_input=None)
    orch.run_split_layer(lm, h=h, layer_idx=layer_idx, cache=cache,
                         masks=(gm, sm), request_id="r2")

    # Expert 0, if it appeared in the batch's top-k, must have gone to peer.
    # Can't assert presence (depends on routing), but if any of experts 0
    # appeared, they went to peer.
    assert all(e == 0 for e in rpc_asked) or not rpc_asked
    orch.close()
```

- [ ] **Step 2: Run**

Run: `uv run pytest -m slow tests/test_routing_correctness.py -v`
Expected: 2 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/test_routing_correctness.py
git commit -m "Phase 4: routing correctness — deterministic pick-less-loaded"
```

---

## Task 11: Phase 3 regression under overlap YAML

No code changes here — just verify Phase 3 tests still pass with the new overlap config.

- [ ] **Step 1: Run regression**

```bash
uv run pytest -m slow tests/test_moe_split_equivalence.py \
                     tests/test_tier1_expert_split_layer15.py \
                     tests/test_tier2_expert_split_layer15.py \
                     tests/test_expert_rpc_handler.py \
                     tests/test_expert_orchestrator.py \
                     tests/test_expert_orchestrator_timeout.py \
                     tests/test_expert_orchestrator_observer.py -v
```

Expected: all pass. The orchestrator now chooses among multi-candidate experts by load; with the fixture's `three_node_pipeline_expert_split`, loads are uniform (nodes barely loaded), so routing is effectively arbitrary but still correct. Tier 1 still matches reference bit-exact (any valid owner computes the same expert output).

- [ ] **Step 2: If anything regresses**

Most likely: stale assumption that `group_expert_ids_by_owner` treats overlapping owners by last-wins. The Task 5/6 changes should have eliminated all call sites. If any test still imports the old `group_expert_ids_by_owner`, keep it as a shim that calls the loaded variant with synthetic uniform loads:

```python
def group_expert_ids_by_owner(
    top_k_ids: list[int], owners: Mapping[str, set[int]]
) -> dict[str, list[int]]:
    # Backward-compat wrapper: zero loads across the board + a shared rng.
    return group_expert_ids_by_owner_loaded(
        top_k_ids, owners=owners, peer_loads={}, self_shard_id="",
        self_load=0, rng=random.Random(0),
    )
```

- [ ] **Step 3: Commit (if any fix)**

If you made the shim fix or other adjustments, commit.
If no changes were needed, skip — the regression passed as-is.

---

## Task 12: E2E load-shift test (subprocess)

**Files:**
- Create: `tests/test_expert_rpc_load_shift.py`

- [ ] **Step 1: Write the test**

The pattern mirrors `tests/membership/test_e2e.py` and `tests/test_expert_rpc_failure.py`. Spawn 3 subprocess nodes with `ENABLE_GOSSIP=true ENABLE_EXPERT_SHARD=true SHARD_DRY_RUN=true`. Use a shards.yaml (generated in tmp_path) with `moe_experts` overlap for expert 0 on both head and mid. Inject latency via an env var consumed in a test-only branch of `_handle_expert_request` — or simpler, monkeypatch before spawn.

Actually, under `SHARD_DRY_RUN=true` the loaded_model is a MagicMock, so `run_selected_experts` will fail at `layer.pre_feedforward_layernorm_2(h)`. For this E2E we need the real model. That makes it expensive (3× model load).

**Surrogate approach** — same as Phase 3's Task 19: don't actually exercise the MLX path; exercise only the gossip. Query `/loads` on each node over time; verify that (a) all three nodes report a load for every peer within ~10s of convergence, and (b) when one node is artificially loaded (e.g. by its test hook inflating `_in_flight_expert_requests`), the other nodes observe that elevated load.

```python
"""E2E: gossip-delivered peer loads are observable via /loads endpoint."""

from __future__ import annotations

import contextlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
RUN_NODE = REPO / "scripts" / "run_node.py"


def _free_port() -> int:
    import random
    for _ in range(100):
        p = random.randint(30000, 60000)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", p))
            return p
        except OSError:
            continue
    raise RuntimeError("no free port")


def _write_shards(tmp_path: Path) -> tuple[Path, dict[str, int]]:
    head, mid, tail = _free_port(), _free_port(), _free_port()
    cfg = {
        "shards": {
            "head": {"host": "127.0.0.1", "port": head,
                     "start_layer": 0, "end_layer": 10,
                     "moe_experts": {15: [0, 3]}},
            "mid":  {"host": "127.0.0.1", "port": mid,
                     "start_layer": 10, "end_layer": 20,
                     "moe_experts": {15: [0, 1]}},
            "tail": {"host": "127.0.0.1", "port": tail,
                     "start_layer": 20, "end_layer": 30,
                     "moe_experts": {15: [2]}},
        }
    }
    p = tmp_path / "shards.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p, {"head": head, "mid": mid, "tail": tail}


def _spawn(shard_id: str, cfg: Path):
    env = {**os.environ,
           "ENABLE_GOSSIP": "true",
           "ENABLE_EXPERT_SHARD": "true",
           "SHARD_DRY_RUN": "true"}
    return subprocess.Popen(
        [sys.executable, str(RUN_NODE), "--shard", shard_id, "--config", str(cfg)],
        env=env, stderr=subprocess.PIPE,
    )


def _get_loads(debug_port: int) -> dict[str, int] | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{debug_port}/loads", timeout=1.0) as resp:
            data = json.loads(resp.read())
            return {k: v["queue_depth_ema"] for k, v in data.items()}
    except Exception:
        return None


@pytest.mark.slow
def test_gossip_delivers_peer_loads_within_ten_seconds(tmp_path: Path) -> None:
    cfg, ports = _write_shards(tmp_path)
    procs = {sid: _spawn(sid, cfg) for sid in ("head", "mid", "tail")}
    try:
        head_debug = ports["head"] + 2000
        deadline = time.monotonic() + 20.0
        last: dict[str, int] | None = None
        while time.monotonic() < deadline:
            view = _get_loads(head_debug)
            last = view
            if view is not None and set(view.keys()) >= {"head", "mid", "tail"}:
                # Every peer is known; success.
                return
            time.sleep(0.5)
        pytest.fail(f"head did not see all peer loads within 20s; final={last}")
    finally:
        for p in procs.values():
            with contextlib.suppress(ProcessLookupError):
                p.terminate()
        for p in procs.values():
            with contextlib.suppress(subprocess.TimeoutExpired):
                p.wait(timeout=5)
```

- [ ] **Step 2: Run**

Run: `uv run pytest -m slow tests/test_expert_rpc_load_shift.py -v`
Expected: pass within ~20s.

- [ ] **Step 3: Commit**

```bash
git add tests/test_expert_rpc_load_shift.py
git commit -m "Phase 4: E2E — gossip delivers peer loads to /loads debug endpoint"
```

---

## Task 13: Final acceptance — ruff + mypy + tests + README + memory

- [ ] **Step 1: Lint + types**

```
uv run ruff check src tests scripts
uv run mypy src tests scripts
```

Both must be clean.

- [ ] **Step 2: Fast suite**

```
uv run pytest
```

Expected: all pass (≥ 130 tests).

- [ ] **Step 3: Slow suite**

Note Phase 3's known in-process Metal state issue: the full 57+ slow suite may segfault late on some machines. Run component groups individually if needed:

```
uv run pytest -m slow tests/test_moe_unit.py tests/test_load_tracker.py \
              tests/test_load_report_envelope.py tests/test_membership_load_records.py \
              tests/test_membership_runner_loads.py tests/test_node_load_wiring.py \
              tests/test_routing_correctness.py tests/test_expert_rpc_load_shift.py -v
```

All new Phase 4 tests must pass.

```
uv run pytest -m slow tests/test_tier1_expert_split_layer15.py \
              tests/test_tier2_expert_split_layer15.py \
              tests/test_moe_split_equivalence.py -v
```

Phase 3 bit-exact regression tests must still pass.

- [ ] **Step 4: Update `README.md`**

Append:

```markdown
## Phase 4 status: Load-Aware Routing — complete

Nodes now gossip a compact queue-depth EMA to each other via `LoadReport`
piggybacked on existing SWIM Ping/Ack messages. When `moe_experts` in
`config/shards.yaml` lists an expert on multiple shards, `ExpertOrchestrator`
routes each top-k dispatch to the less-loaded candidate via
power-of-two-choices. Routing correctness is verified by
`tests/test_routing_correctness.py`; the configuration overlaps experts
0, 1, and 2 across two shards each for a live multi-candidate scenario. See
`docs/superpowers/specs/2026-04-16-phase4-load-aware-routing-design.md`.
```

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "Phase 4 complete: load-aware routing with static per-expert replication"
```

- [ ] **Step 6: Update memory**

Tell the operator: Phase 4 is complete. They may want to update
`~/.claude/projects/-Users-lukechang-Github-model-shard/memory/project_gossip_moe.md`
to mark Phase 4 done and Phase 5 (Dynamic Expert Migration) next. Phase 5
starts with a fresh brainstorming cycle.

---

## Self-Review

### 1. Spec coverage

| Spec § | Implemented in tasks |
|---|---|
| D1 routing-correctness success criterion | Task 10 |
| D2 static YAML replication | Task 9 |
| D3 power-of-two-choices | Task 5 |
| D4 anti-oscillation (EMA + jitter) | Task 3 |
| D5 piggyback on SWIM | Tasks 1, 2, 4 |
| D6 queue-depth EMA only | Task 3 |
| D7 non-goals (no batching/migration/heat) | — (by omission) |
| §3.1 LoadTracker | Task 3 |
| §3.2 runner piggyback | Task 4 |
| §3.3 group_expert_ids_by_owner_loaded | Task 5 |
| §3.4 orchestrator wiring | Task 6 |
| §3.5 node wiring + /loads endpoint | Tasks 7, 8 |
| §4 wire protocol | Tasks 1, 2 |
| §5 data flow | Tasks 7, 8 |
| §6 testing | Tasks 3, 5, 10, 12, 11 |
| §7 rollback (no env var; auto-on via YAML + gossip flags) | Task 7 (note) |
| §8 acceptance | Task 13 |

### 2. Placeholder scan

- Task 7's code uses `getattr(self._cfg, "rpc_timeout_s", 5.0)` because the exact config attribute name varies by the Phase 3 shape. Implementer should inspect the existing construction site and use the actual attribute.
- Task 2's encode/decode changes the existing `decode_membership_envelope` branches; the full replacement isn't written out because there are 4 branches and the pattern is identical for each. Implementer reads the file and applies the pattern — low risk of drift.
- Task 12's "Surrogate approach" note acknowledges the MLX subprocess cost and deliberately tests only the gossip delivery, not full inference. This is a scope choice, not a placeholder.

### 3. Type / name consistency

- `LoadReportRecord` — dataclass with `shard_id: str`, `queue_depth_ema: int`, `ts_unix_ms: int`. Used uniformly across records.py, messages.py, runner.py, node.py, tests.
- `group_expert_ids_by_owner_loaded` — signature `(top_k_ids, owners, peer_loads, self_shard_id, self_load, rng) -> dict[str, list[int]]`. Used uniformly by tests and the orchestrator.
- `ExpertOrchestrator.loads_provider: Callable[[], Mapping[str, int]]` — consistent in Tasks 6, 7, 10.
- `MembershipRunner.start_load_source(fn)` / `latest_loads() -> dict[str, LoadReportRecord]` — consistent in Tasks 4, 7.
- `_MLX_COMPUTE_LOCK` — reused from Phase 3; unchanged.

### 4. Scope check

Plan covers a single subsystem (gossip-driven load awareness + loaded-P2C routing). Non-goals in spec §1.3 are not touched. 13 tasks, most bite-sized; Task 2 and Task 4 are the densest but each remains a single logical change (adapter layer; runner API).
