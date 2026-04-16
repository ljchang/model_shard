# Phase 2: Gossip Discovery — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SWIM-style membership discovery to the Phase 1 distributed pipeline so the head admits requests only when all required shards are `alive`, and so in-flight requests fail cleanly when a peer transitions out of `alive`.

**Architecture:** A new `src/model_shard/membership/` package containing a pure SWIM state machine (no I/O), a UDP transport sidecar, a sequential-seed bootstrap, and a runner thread that ties them together. `node.py` gains a small observer hook that closes/redials TCP peer connections on membership changes and rejects new `BeginRequest`s when any shard is not `alive`. No Phase 1 file is restructured.

**Tech Stack:** Python 3.11, MLX, protobuf, raw UDP sockets (`socket.SOCK_DGRAM`), `threading`, pytest. All new code passes `ruff` and `mypy --strict`.

**Spec:** [`docs/superpowers/specs/2026-04-16-phase2-gossip-discovery-design.md`](../specs/2026-04-16-phase2-gossip-discovery-design.md).

**Working branch:** `main` (Phase 1 was developed directly on `main`; matching the established pattern).

---

## File map

### Created

| Path | Responsibility |
|---|---|
| `src/model_shard/membership/__init__.py` | Package marker; re-exports `MembershipRunner`, `MembershipState`, `SwimConfig`. |
| `src/model_shard/membership/config.py` | `SwimConfig` dataclass: timing constants and fanouts. |
| `src/model_shard/membership/records.py` | `MemberRecord`, `StateTransition`, message tagged-union dataclasses. Pure data. |
| `src/model_shard/membership/state.py` | `MembershipState` — the pure SWIM core. No I/O, no time, no threads. |
| `src/model_shard/membership/messages.py` | Protobuf ↔ dataclass adapters for the wire messages. |
| `src/model_shard/membership/transport.py` | `UDPTransport`: bind, `send_to`, `recv_loop` with MTU guard. |
| `src/model_shard/membership/bootstrap.py` | Sequential seed contact; reads peers from `ShardMap`. |
| `src/model_shard/membership/runner.py` | `MembershipRunner` thread; observer pattern; routes transport → state → transport. |
| `tests/membership/__init__.py` | Package marker. |
| `tests/membership/test_records.py` | Pure data tests. |
| `tests/membership/test_state.py` | Fast SWIM state machine tests (~40 cases, virtual clock). |
| `tests/membership/test_messages.py` | Protobuf round-trip tests. |
| `tests/membership/test_transport.py` | UDP transport tests with loopback. |
| `tests/membership/test_bootstrap.py` | Bootstrap logic with mocked transport. |
| `tests/membership/test_runner.py` | Runner unit tests with fake state + transport. |
| `tests/membership/test_e2e.py` | Slow behavioral cluster tests (~10 cases, marked `slow`). |

### Modified

| Path | Change |
|---|---|
| `proto/wire.proto` | Add 6 new message types (`Ping`, `Ack`, `PingReq`, `PingReqAck`, `Join`, `MembershipDelta`); add `ERR_SHARD_UNAVAILABLE` enum value. |
| `src/model_shard/_pb/wire_pb2.py` | Regenerated from proto. Auto-generated; do not hand-edit. |
| `src/model_shard/node.py` | Add `MembershipRunner` construction (gated on `ENABLE_GOSSIP`), admission-control check in `_handle_begin`, observer hook for TCP close/redial, structured error on broken pipe. |

### Unchanged

`mlx_engine.py`, `transport.py`, `envelope.py`, `request.py`, `shard.py`, `shard_map.py`, `client.py`, `reference.py`, `config/shards.yaml`. The Phase 1 `Tier 1` and `Tier 2` test suites are not edited.

---

## Tasks

### Task 1: Wire protocol — add SWIM messages and `ERR_SHARD_UNAVAILABLE`

**Files:**
- Modify: `proto/wire.proto`
- Regenerate: `src/model_shard/_pb/wire_pb2.py`
- Test: (deferred to Task 19; this task is structural)

- [ ] **Step 1: Edit `proto/wire.proto`** — append after the existing `Error` message (around line 117), and extend the `Envelope` oneof:

```protobuf
// ---------------------------------------------------------------------------
// Phase 2 — SWIM membership messages.
// All membership messages travel over a separate UDP transport (not the TCP
// envelope used for activations). The framing is one protobuf-encoded
// Envelope per UDP datagram. Datagrams must fit in a single 1400-byte MTU
// budget. See src/model_shard/membership/transport.py.
// ---------------------------------------------------------------------------

message MemberRecordPb {
  string shard_id = 1;
  string host = 2;
  uint32 udp_port = 3;
  // 0 = alive, 1 = suspect, 2 = dead. Must match records.MemberState ordering.
  uint32 state = 4;
  uint64 incarnation = 5;
}

message Ping {
  uint32 protocol_version = 1;
  string from_shard_id = 2;
  uint64 from_incarnation = 3;
  // Up to K_GOSSIP recent membership transitions piggybacked.
  repeated MemberRecordPb deltas = 4;
}

message Ack {
  uint32 protocol_version = 1;
  string from_shard_id = 2;
  uint64 from_incarnation = 3;
  repeated MemberRecordPb deltas = 4;
}

message PingReq {
  uint32 protocol_version = 1;
  string from_shard_id = 2;
  string target_shard_id = 3;
  // A correlation id so the requester can match the eventual PingReqAck.
  string probe_id = 4;
  repeated MemberRecordPb deltas = 5;
}

message PingReqAck {
  uint32 protocol_version = 1;
  string from_shard_id = 2;
  string target_shard_id = 3;
  string probe_id = 4;
  bool success = 5;
  repeated MemberRecordPb deltas = 6;
}

message Join {
  uint32 protocol_version = 1;
  MemberRecordPb self_record = 2;
}

message MembershipDelta {
  uint32 protocol_version = 1;
  // Full snapshot of the sender's view, used as one-shot bootstrap response.
  repeated MemberRecordPb members = 2;
}
```

Then extend the `Envelope` `oneof payload` block (was tags 1–7; add 8–13):

```protobuf
message Envelope {
  oneof payload {
    BeginRequest begin = 1;
    ContinueRequest cont = 2;
    Activation activation = 3;
    Logits logits = 4;
    EndRequest end = 5;
    Error error = 6;
    SampledToken sampled_token = 7;
    // Phase 2 SWIM membership messages.
    Ping ping = 8;
    Ack ack = 9;
    PingReq ping_req = 10;
    PingReqAck ping_req_ack = 11;
    Join join = 12;
    MembershipDelta membership_delta = 13;
  }
}
```

And extend the `ErrorCode` enum:

```protobuf
enum ErrorCode {
  ERR_UNSPECIFIED = 0;
  ERR_UNKNOWN_REQUEST = 1;
  ERR_WRONG_SHARD = 2;
  ERR_PROTOCOL_VERSION = 3;
  ERR_INTERNAL = 4;
  ERR_SHARD_UNAVAILABLE = 5;  // Phase 2: a required shard is suspect/dead.
}
```

- [ ] **Step 2: Regenerate `wire_pb2.py`**

Run: `uv run python -m grpc_tools.protoc -I proto --python_out=src/model_shard/_pb proto/wire.proto`
Expected: command succeeds silently. `git diff --stat src/model_shard/_pb/wire_pb2.py` shows additions.

- [ ] **Step 3: Smoke-check the regenerated bindings load**

Run: `uv run python -c "from model_shard._pb import wire_pb2; e = wire_pb2.Envelope(); e.ping.from_shard_id = 'x'; assert e.WhichOneof('payload') == 'ping'; print('ok')"`
Expected output: `ok`

- [ ] **Step 4: Commit**

```bash
git add proto/wire.proto src/model_shard/_pb/wire_pb2.py
git commit -m "Phase 2: add SWIM wire messages + ERR_SHARD_UNAVAILABLE"
```

---

### Task 2: Package scaffolding + `SwimConfig`

**Files:**
- Create: `src/model_shard/membership/__init__.py`
- Create: `src/model_shard/membership/config.py`
- Create: `tests/membership/__init__.py`
- Create: `tests/membership/test_config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/membership/__init__.py` empty. Then create `tests/membership/test_config.py`:

```python
from model_shard.membership.config import SwimConfig


def test_swim_config_has_spec_default_values() -> None:
    cfg = SwimConfig()
    assert cfg.t_ping_ms == 1000
    assert cfg.t_tick_ms == 100
    assert cfg.t_timeout_ms == 500
    assert cfg.k_indirect == 2
    assert cfg.t_suspect_ms == 4 * cfg.t_ping_ms
    assert cfg.k_gossip == 3
    assert cfg.mtu_bytes == 1400


def test_swim_config_is_frozen() -> None:
    cfg = SwimConfig()
    import dataclasses
    assert dataclasses.is_dataclass(cfg)
    try:
        cfg.t_ping_ms = 999  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("SwimConfig should be frozen")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/membership/test_config.py -v`
Expected: `ModuleNotFoundError: No module named 'model_shard.membership'`.

- [ ] **Step 3: Create the package and config**

Create `src/model_shard/membership/__init__.py`:

```python
"""SWIM-style membership discovery for the model_shard cluster.

Public surface re-exported here is what `node.py` and tests should import.
Internal modules (state, messages, transport, runner, bootstrap) may be
imported directly when writing tests against a single layer.
"""

from model_shard.membership.config import SwimConfig

__all__ = ["SwimConfig"]
```

Create `src/model_shard/membership/config.py`:

```python
"""Tunable timing/fanout constants for the SWIM membership protocol.

Defaults match the Phase 2 design spec (`docs/superpowers/specs/...`). They are
chosen for the localhost-3-node prototype but remain reasonable up to ~30
nodes. All time values are in milliseconds; the state machine and runner
convert to seconds internally where useful.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SwimConfig:
    t_ping_ms: int = 1000        # interval between protocol-period pings per peer
    t_tick_ms: int = 100         # state machine clock granularity
    t_timeout_ms: int = 500      # direct ping ack deadline (half of t_ping_ms)
    k_indirect: int = 2          # ping-req fanout when a direct ping times out
    k_gossip: int = 3            # max membership deltas piggybacked per message
    mtu_bytes: int = 1400        # safe single-datagram size; messages exceeding it are dropped
    t_suspect_ms: int = 4000     # suspect-deadline window; default = 4 * t_ping_ms
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/membership/test_config.py -v`
Expected: 2 passed.

- [ ] **Step 5: Run lint + types on new files**

Run: `uv run ruff check src/model_shard/membership tests/membership && uv run mypy src/model_shard/membership tests/membership`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/model_shard/membership/__init__.py src/model_shard/membership/config.py tests/membership/__init__.py tests/membership/test_config.py
git commit -m "Phase 2: membership package scaffolding + SwimConfig"
```

---

### Task 3: `MemberRecord` and message tagged unions

**Files:**
- Create: `src/model_shard/membership/records.py`
- Create: `tests/membership/test_records.py`

- [ ] **Step 1: Write the failing test**

Create `tests/membership/test_records.py`:

```python
import dataclasses

from model_shard.membership.records import (
    AckMsg,
    JoinMsg,
    MemberRecord,
    MembershipDeltaMsg,
    MemberState,
    PingMsg,
    PingReqAckMsg,
    PingReqMsg,
    StateTransition,
)


def test_member_state_ordering_is_dead_gt_suspect_gt_alive() -> None:
    # The numeric values must match the wire MemberRecordPb.state encoding.
    assert MemberState.ALIVE.value == 0
    assert MemberState.SUSPECT.value == 1
    assert MemberState.DEAD.value == 2
    # Severity ordering is used by the same-incarnation tiebreaker.
    assert MemberState.DEAD > MemberState.SUSPECT > MemberState.ALIVE


def test_member_record_is_immutable_dataclass() -> None:
    rec = MemberRecord(
        shard_id="x",
        host="127.0.0.1",
        udp_port=10001,
        state=MemberState.ALIVE,
        incarnation=0,
        last_state_change=0.0,
        suspect_deadline=None,
    )
    assert dataclasses.is_dataclass(rec)
    try:
        rec.incarnation = 1  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("MemberRecord should be frozen")


def test_state_transition_carries_old_and_new() -> None:
    rec = MemberRecord("x", "127.0.0.1", 10001, MemberState.SUSPECT, 3, 1.0, 5.0)
    t = StateTransition(
        shard_id="x",
        old_state=MemberState.ALIVE,
        new_record=rec,
    )
    assert t.shard_id == "x"
    assert t.old_state == MemberState.ALIVE
    assert t.new_record.state == MemberState.SUSPECT


def test_message_dataclasses_construct() -> None:
    rec = MemberRecord("x", "127.0.0.1", 10001, MemberState.ALIVE, 0, 0.0, None)
    PingMsg(from_shard_id="a", from_incarnation=2, deltas=[rec])
    AckMsg(from_shard_id="a", from_incarnation=2, deltas=[rec])
    PingReqMsg(from_shard_id="a", target_shard_id="b", probe_id="p1", deltas=[])
    PingReqAckMsg(from_shard_id="a", target_shard_id="b", probe_id="p1", success=True, deltas=[])
    JoinMsg(self_record=rec)
    MembershipDeltaMsg(members=[rec])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/membership/test_records.py -v`
Expected: `ModuleNotFoundError: No module named 'model_shard.membership.records'`.

- [ ] **Step 3: Implement `records.py`**

Create `src/model_shard/membership/records.py`:

```python
"""Pure data types for the SWIM membership layer.

Every type here is frozen and free of I/O imports. The state machine
(`state.py`) operates exclusively on these types; conversion to/from
protobuf lives in `messages.py`.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Union


class MemberState(IntEnum):
    """Membership states. Ordering is meaningful: severity ALIVE < SUSPECT < DEAD.

    Used by the same-incarnation tiebreaker in `state.py`. Numeric values
    must match the `state` field of wire `MemberRecordPb`.
    """

    ALIVE = 0
    SUSPECT = 1
    DEAD = 2


@dataclass(frozen=True)
class MemberRecord:
    shard_id: str
    host: str
    udp_port: int
    state: MemberState
    incarnation: int
    last_state_change: float
    suspect_deadline: float | None  # set iff state == SUSPECT


@dataclass(frozen=True)
class StateTransition:
    """Emitted from `MembershipState` whenever a member's recorded state changes.

    `new_record` is the post-transition record; `old_state` is the prior state
    (or None if this is a brand-new member entering the view).
    """

    shard_id: str
    old_state: MemberState | None
    new_record: "MemberRecord"


# ----- Wire-level message dataclasses (mirror the protobuf shapes) -----------


@dataclass(frozen=True)
class PingMsg:
    from_shard_id: str
    from_incarnation: int
    deltas: list[MemberRecord]


@dataclass(frozen=True)
class AckMsg:
    from_shard_id: str
    from_incarnation: int
    deltas: list[MemberRecord]


@dataclass(frozen=True)
class PingReqMsg:
    from_shard_id: str
    target_shard_id: str
    probe_id: str
    deltas: list[MemberRecord]


@dataclass(frozen=True)
class PingReqAckMsg:
    from_shard_id: str
    target_shard_id: str
    probe_id: str
    success: bool
    deltas: list[MemberRecord]


@dataclass(frozen=True)
class JoinMsg:
    self_record: MemberRecord


@dataclass(frozen=True)
class MembershipDeltaMsg:
    members: list[MemberRecord]


IncomingMessage = Union[
    PingMsg, AckMsg, PingReqMsg, PingReqAckMsg, JoinMsg, MembershipDeltaMsg
]
"""A union of every membership message that `MembershipState.recv` accepts."""


@dataclass(frozen=True)
class OutgoingMessage:
    """A message produced by the state machine, ready to be UDP-sent.

    `target` is the wire address the runner should `sendto`; `payload` is one
    of the message dataclasses above (the runner serialises via messages.py).
    """

    target_shard_id: str
    payload: IncomingMessage  # same set of types; outgoing == incoming


__all__ = [
    "AckMsg",
    "IncomingMessage",
    "JoinMsg",
    "MemberRecord",
    "MemberState",
    "MembershipDeltaMsg",
    "OutgoingMessage",
    "PingMsg",
    "PingReqAckMsg",
    "PingReqMsg",
    "StateTransition",
]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/membership/test_records.py -v`
Expected: 4 passed.

- [ ] **Step 5: Lint + types**

Run: `uv run ruff check src/model_shard/membership tests/membership && uv run mypy src/model_shard/membership tests/membership`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/model_shard/membership/records.py tests/membership/test_records.py
git commit -m "Phase 2: MemberRecord + StateTransition + message tagged unions"
```

---

### Task 4: `MembershipState` skeleton — constructor + `view()`

**Files:**
- Create: `src/model_shard/membership/state.py`
- Create: `tests/membership/test_state.py`

- [ ] **Step 1: Write the failing test**

Create `tests/membership/test_state.py`:

```python
"""Pure state machine tests. Virtual clock; no sockets, no threads."""

import random

from model_shard.membership.config import SwimConfig
from model_shard.membership.records import MemberRecord, MemberState
from model_shard.membership.state import MembershipState, PeerSpec


def make_state(
    self_id: str = "n0",
    peers: tuple[str, ...] = ("n1", "n2"),
    seed: int = 0,
    cfg: SwimConfig | None = None,
) -> MembershipState:
    """Test helper: build a MembershipState with the named peers."""
    self_spec = PeerSpec(shard_id=self_id, host="127.0.0.1", udp_port=10000)
    peer_specs = [
        PeerSpec(shard_id=p, host="127.0.0.1", udp_port=10000 + i + 1)
        for i, p in enumerate(peers)
    ]
    return MembershipState(
        self_spec=self_spec,
        peer_specs=peer_specs,
        rng=random.Random(seed),
        config=cfg or SwimConfig(),
    )


def test_initial_view_contains_self_alive_at_incarnation_zero() -> None:
    s = make_state()
    view = s.view()
    assert "n0" in view
    rec = view["n0"]
    assert rec.state == MemberState.ALIVE
    assert rec.incarnation == 0


def test_initial_view_contains_each_peer_alive_at_incarnation_zero() -> None:
    s = make_state(peers=("n1", "n2", "n3"))
    view = s.view()
    for name in ("n1", "n2", "n3"):
        assert name in view, f"missing peer {name}"
        assert view[name].state == MemberState.ALIVE
        assert view[name].incarnation == 0


def test_view_returns_a_copy_not_internal_reference() -> None:
    s = make_state()
    view = s.view()
    view.clear()
    # Mutating the returned dict must not affect internal state.
    assert "n0" in s.view()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/membership/test_state.py -v`
Expected: `ModuleNotFoundError: No module named 'model_shard.membership.state'`.

- [ ] **Step 3: Implement skeleton**

Create `src/model_shard/membership/state.py`:

```python
"""Pure SWIM state machine — no I/O, no time, no threads.

The runner is responsible for invoking `tick(now)` on a clock and for
delivering received messages to `recv(msg, now)`. Both methods return a list
of `OutgoingMessage` for the runner to send. State transitions are reported
via `changes_since(watermark)` so the runner can fire observer callbacks.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

from model_shard.membership.config import SwimConfig
from model_shard.membership.records import (
    IncomingMessage,
    MemberRecord,
    MemberState,
    OutgoingMessage,
    StateTransition,
)


@dataclass(frozen=True)
class PeerSpec:
    """A peer's static identity. Derived from `shards.yaml` at startup."""

    shard_id: str
    host: str
    udp_port: int


class MembershipState:
    def __init__(
        self,
        self_spec: PeerSpec,
        peer_specs: list[PeerSpec],
        rng: random.Random,
        config: SwimConfig,
    ) -> None:
        self._self_id = self_spec.shard_id
        self._self_incarnation = 0
        self._cfg = config
        self._rng = rng
        self._addrs: dict[str, PeerSpec] = {self_spec.shard_id: self_spec}
        for p in peer_specs:
            self._addrs[p.shard_id] = p
        self._members: dict[str, MemberRecord] = {}
        for p in [self_spec, *peer_specs]:
            self._members[p.shard_id] = MemberRecord(
                shard_id=p.shard_id,
                host=p.host,
                udp_port=p.udp_port,
                state=MemberState.ALIVE,
                incarnation=0,
                last_state_change=0.0,
                suspect_deadline=None,
            )
        self._transitions: list[StateTransition] = []

    # ---------------------------------------------------------- read-only API

    def view(self) -> dict[str, MemberRecord]:
        """Return a snapshot copy of every known member's record."""
        return dict(self._members)

    def changes_since(self, watermark: int) -> list[StateTransition]:
        """Return transitions appended after the given watermark index."""
        return list(self._transitions[watermark:])

    @property
    def transition_watermark(self) -> int:
        """Current length of the transitions log; pass to `changes_since`."""
        return len(self._transitions)

    # --------------------------------------------------------- mutation hooks
    # Filled in by later tasks. tick/recv/local_event are listed here as stubs
    # so type-checkers and importers see the expected surface from Task 4 on.

    def tick(self, now: float) -> list[OutgoingMessage]:
        return []

    def recv(self, msg: IncomingMessage, now: float) -> list[OutgoingMessage]:
        return []


__all__ = ["MembershipState", "PeerSpec"]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/membership/test_state.py -v`
Expected: 3 passed.

- [ ] **Step 5: Lint + types**

Run: `uv run ruff check src/model_shard/membership tests/membership && uv run mypy src/model_shard/membership tests/membership`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/model_shard/membership/state.py tests/membership/test_state.py
git commit -m "Phase 2: MembershipState skeleton with view() and PeerSpec"
```

---

### Task 5: `tick()` — pick a ping target, emit `PingMsg`, track pending probe

**Files:**
- Modify: `src/model_shard/membership/state.py`
- Modify: `tests/membership/test_state.py`

- [ ] **Step 1: Write the failing test** — append to `tests/membership/test_state.py`:

```python
from model_shard.membership.records import PingMsg


def test_tick_emits_no_message_before_first_protocol_period() -> None:
    s = make_state()
    # The first protocol period fires at t = T_PING; before that, no ping.
    out = s.tick(now=0.5)
    assert out == []


def test_tick_emits_ping_at_first_protocol_period() -> None:
    s = make_state(peers=("n1", "n2"), seed=0)
    out = s.tick(now=1.0)  # exactly T_PING = 1000ms
    assert len(out) == 1
    msg = out[0]
    assert isinstance(msg.payload, PingMsg)
    assert msg.payload.from_shard_id == "n0"
    assert msg.payload.from_incarnation == 0
    # Target must be one of the peers, never self.
    assert msg.target_shard_id in {"n1", "n2"}
    assert msg.target_shard_id != "n0"


def test_tick_does_not_re_emit_within_one_period() -> None:
    s = make_state(seed=0)
    s.tick(now=1.0)
    out = s.tick(now=1.5)
    assert out == []


def test_tick_emits_again_after_full_period() -> None:
    s = make_state(seed=0)
    s.tick(now=1.0)
    out = s.tick(now=2.0)
    assert len(out) == 1


def test_tick_emits_no_ping_when_no_alive_peers() -> None:
    s = make_state(peers=())
    out = s.tick(now=10.0)
    assert out == []
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/membership/test_state.py -v`
Expected: 5 new tests fail (`assert [] == ... PingMsg` etc.).

- [ ] **Step 3: Implement protocol-period tracking and ping emission**

In `src/model_shard/membership/state.py`, add to imports and to `MembershipState.__init__`:

```python
from model_shard.membership.records import (
    AckMsg,                     # add
    IncomingMessage,
    MemberRecord,
    MemberState,
    OutgoingMessage,
    PingMsg,                    # add
    PingReqAckMsg,              # add (used in later tasks)
    PingReqMsg,                 # add (used in later tasks)
    StateTransition,
)
```

Inside `__init__`, after the existing fields, add:

```python
        # Protocol-period state. Each period: pick a peer, ping, await ack,
        # escalate to indirect probe if no ack, finally suspect on no positive
        # PingReqAck.
        self._next_period_at: float = float(self._cfg.t_ping_ms) / 1000.0
        self._pending_probe: _PendingProbe | None = None
        self._probe_counter: int = 0
```

Add the `_PendingProbe` dataclass at module top, near `PeerSpec`:

```python
@dataclass(frozen=True)
class _PendingProbe:
    probe_id: str
    target_id: str
    sent_at: float
    indirect_sent_at: float | None  # set when escalated to ping-req
    indirect_targets: tuple[str, ...]  # peers contacted via ping-req
    indirect_acks: int  # count of PingReqAck (success or failure) received
    indirect_success_seen: bool  # any positive PingReqAck received?
```

Replace the stub `tick` method with:

```python
    def tick(self, now: float) -> list[OutgoingMessage]:
        out: list[OutgoingMessage] = []
        if now < self._next_period_at:
            return out

        # Start of new protocol period. Pick a random ALIVE peer (excluding self).
        candidates = [
            r.shard_id
            for r in self._members.values()
            if r.shard_id != self._self_id and r.state == MemberState.ALIVE
        ]
        if not candidates:
            self._next_period_at = now + self._cfg.t_ping_ms / 1000.0
            return out

        target = self._rng.choice(candidates)
        self._probe_counter += 1
        probe_id = f"{self._self_id}:{self._probe_counter}"
        self._pending_probe = _PendingProbe(
            probe_id=probe_id,
            target_id=target,
            sent_at=now,
            indirect_sent_at=None,
            indirect_targets=(),
            indirect_acks=0,
            indirect_success_seen=False,
        )
        out.append(
            OutgoingMessage(
                target_shard_id=target,
                payload=PingMsg(
                    from_shard_id=self._self_id,
                    from_incarnation=self._self_incarnation,
                    deltas=[],
                ),
            )
        )
        self._next_period_at = now + self._cfg.t_ping_ms / 1000.0
        return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/membership/test_state.py -v`
Expected: all tests pass (8 total).

- [ ] **Step 5: Lint + types**

Run: `uv run ruff check src/model_shard/membership tests/membership && uv run mypy src/model_shard/membership tests/membership`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/model_shard/membership/state.py tests/membership/test_state.py
git commit -m "Phase 2: tick() picks ping target and emits Ping with pending-probe tracking"
```

---

### Task 6: `recv(PingMsg)` → emit `AckMsg`

**Files:**
- Modify: `src/model_shard/membership/state.py`
- Modify: `tests/membership/test_state.py`

- [ ] **Step 1: Write the failing test** — append:

```python
from model_shard.membership.records import AckMsg


def test_recv_ping_emits_ack_to_sender() -> None:
    s = make_state(self_id="n0", peers=("n1",))
    msg = PingMsg(from_shard_id="n1", from_incarnation=0, deltas=[])
    out = s.recv(msg, now=0.0)
    assert len(out) == 1
    assert out[0].target_shard_id == "n1"
    payload = out[0].payload
    assert isinstance(payload, AckMsg)
    assert payload.from_shard_id == "n0"
    assert payload.from_incarnation == 0


def test_recv_ping_from_unknown_peer_is_dropped() -> None:
    s = make_state(self_id="n0", peers=("n1",))
    msg = PingMsg(from_shard_id="ghost", from_incarnation=0, deltas=[])
    out = s.recv(msg, now=0.0)
    assert out == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/membership/test_state.py -v`
Expected: 2 new tests fail.

- [ ] **Step 3: Implement** — replace `recv` stub:

```python
    def recv(self, msg: IncomingMessage, now: float) -> list[OutgoingMessage]:
        if isinstance(msg, PingMsg):
            return self._handle_ping(msg, now)
        return []

    def _handle_ping(self, msg: PingMsg, now: float) -> list[OutgoingMessage]:
        if msg.from_shard_id not in self._members:
            return []
        return [
            OutgoingMessage(
                target_shard_id=msg.from_shard_id,
                payload=AckMsg(
                    from_shard_id=self._self_id,
                    from_incarnation=self._self_incarnation,
                    deltas=[],
                ),
            )
        ]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/membership/test_state.py -v`
Expected: 10 passed.

- [ ] **Step 5: Lint + types and commit**

```bash
uv run ruff check src tests scripts && uv run mypy src tests scripts
git add src/model_shard/membership/state.py tests/membership/test_state.py
git commit -m "Phase 2: recv(Ping) emits Ack to sender"
```

---

### Task 7: `recv(AckMsg)` clears the pending probe

**Files:**
- Modify: `src/model_shard/membership/state.py`
- Modify: `tests/membership/test_state.py`

- [ ] **Step 1: Write the failing test** — append:

```python
def test_recv_ack_clears_pending_probe() -> None:
    s = make_state(seed=0)
    s.tick(now=1.0)  # produces a Ping
    pending_target = s._pending_probe.target_id  # type: ignore[union-attr]
    ack = AckMsg(from_shard_id=pending_target, from_incarnation=0, deltas=[])
    s.recv(ack, now=1.2)
    assert s._pending_probe is None  # type: ignore[attr-defined]


def test_recv_ack_from_unrelated_peer_is_ignored() -> None:
    s = make_state(seed=0)
    s.tick(now=1.0)
    pending = s._pending_probe  # type: ignore[attr-defined]
    other = "n2" if pending.target_id == "n1" else "n1"
    ack = AckMsg(from_shard_id=other, from_incarnation=0, deltas=[])
    s.recv(ack, now=1.2)
    # pending probe still in place
    assert s._pending_probe is pending  # type: ignore[attr-defined]
```

- [ ] **Step 2: Run** — Expected: both fail.

Run: `uv run pytest tests/membership/test_state.py -v`

- [ ] **Step 3: Implement** — extend `recv`:

```python
    def recv(self, msg: IncomingMessage, now: float) -> list[OutgoingMessage]:
        if isinstance(msg, PingMsg):
            return self._handle_ping(msg, now)
        if isinstance(msg, AckMsg):
            return self._handle_ack(msg, now)
        return []

    def _handle_ack(self, msg: AckMsg, now: float) -> list[OutgoingMessage]:
        probe = self._pending_probe
        if probe is not None and probe.target_id == msg.from_shard_id:
            self._pending_probe = None
        return []
```

- [ ] **Step 4: Run** — Expected: 12 passed.

Run: `uv run pytest tests/membership/test_state.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/model_shard/membership/state.py tests/membership/test_state.py
git commit -m "Phase 2: recv(Ack) clears matching pending probe"
```

---

### Task 8: `tick()` escalates to `PingReq` after ack timeout

**Files:**
- Modify: `src/model_shard/membership/state.py`
- Modify: `tests/membership/test_state.py`

- [ ] **Step 1: Write the failing test** — append:

```python
from model_shard.membership.records import PingReqMsg


def test_tick_escalates_to_pingreq_after_timeout() -> None:
    # 4 peers so K_INDIRECT=2 random peers can be picked, plus the target.
    s = make_state(self_id="n0", peers=("n1", "n2", "n3", "n4"), seed=0)
    s.tick(now=1.0)
    target = s._pending_probe.target_id  # type: ignore[union-attr]

    # T_TIMEOUT = 500ms; escalate at t = 1.5s.
    out = s.tick(now=1.5)

    pingreqs = [m for m in out if isinstance(m.payload, PingReqMsg)]
    assert len(pingreqs) == 2  # K_INDIRECT
    for m in pingreqs:
        assert m.target_shard_id != target
        assert m.target_shard_id != "n0"
        payload = m.payload
        assert isinstance(payload, PingReqMsg)
        assert payload.target_shard_id == target


def test_tick_does_not_escalate_twice() -> None:
    s = make_state(peers=("n1", "n2", "n3", "n4"), seed=0)
    s.tick(now=1.0)
    s.tick(now=1.5)  # first escalation
    out = s.tick(now=1.6)
    assert all(not isinstance(m.payload, PingReqMsg) for m in out)


def test_tick_does_not_escalate_if_ack_arrived_first() -> None:
    s = make_state(peers=("n1", "n2", "n3", "n4"), seed=0)
    s.tick(now=1.0)
    target = s._pending_probe.target_id  # type: ignore[union-attr]
    s.recv(AckMsg(from_shard_id=target, from_incarnation=0, deltas=[]), now=1.2)
    out = s.tick(now=1.5)
    assert all(not isinstance(m.payload, PingReqMsg) for m in out)
```

- [ ] **Step 2: Run** — Expected: 3 new failures.

Run: `uv run pytest tests/membership/test_state.py -v`

- [ ] **Step 3: Implement** — modify `tick` to also handle escalation. Replace the existing `tick` body's "early return / start new period" structure with:

```python
    def tick(self, now: float) -> list[OutgoingMessage]:
        out: list[OutgoingMessage] = []

        # 1. Escalate pending probe to indirect ping-req if ack overdue.
        out.extend(self._maybe_escalate_probe(now))

        # 2. Start a new protocol period if it's time.
        if now >= self._next_period_at:
            out.extend(self._start_protocol_period(now))

        return out

    def _maybe_escalate_probe(self, now: float) -> list[OutgoingMessage]:
        probe = self._pending_probe
        if probe is None or probe.indirect_sent_at is not None:
            return []
        if now < probe.sent_at + self._cfg.t_timeout_ms / 1000.0:
            return []

        # Pick K_INDIRECT random alive peers, excluding self and the target.
        candidates = [
            r.shard_id
            for r in self._members.values()
            if r.shard_id not in (self._self_id, probe.target_id)
            and r.state == MemberState.ALIVE
        ]
        self._rng.shuffle(candidates)
        chosen = tuple(candidates[: self._cfg.k_indirect])
        out: list[OutgoingMessage] = []
        for helper in chosen:
            out.append(
                OutgoingMessage(
                    target_shard_id=helper,
                    payload=PingReqMsg(
                        from_shard_id=self._self_id,
                        target_shard_id=probe.target_id,
                        probe_id=probe.probe_id,
                        deltas=[],
                    ),
                )
            )
        self._pending_probe = replace(
            probe, indirect_sent_at=now, indirect_targets=chosen
        )
        return out

    def _start_protocol_period(self, now: float) -> list[OutgoingMessage]:
        candidates = [
            r.shard_id
            for r in self._members.values()
            if r.shard_id != self._self_id and r.state == MemberState.ALIVE
        ]
        if not candidates:
            self._next_period_at = now + self._cfg.t_ping_ms / 1000.0
            return []

        target = self._rng.choice(candidates)
        self._probe_counter += 1
        probe_id = f"{self._self_id}:{self._probe_counter}"
        self._pending_probe = _PendingProbe(
            probe_id=probe_id,
            target_id=target,
            sent_at=now,
            indirect_sent_at=None,
            indirect_targets=(),
            indirect_acks=0,
            indirect_success_seen=False,
        )
        self._next_period_at = now + self._cfg.t_ping_ms / 1000.0
        return [
            OutgoingMessage(
                target_shard_id=target,
                payload=PingMsg(
                    from_shard_id=self._self_id,
                    from_incarnation=self._self_incarnation,
                    deltas=[],
                ),
            )
        ]
```

- [ ] **Step 4: Run** — Expected: 15 passed.

Run: `uv run pytest tests/membership/test_state.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/model_shard/membership/state.py tests/membership/test_state.py
git commit -m "Phase 2: tick() escalates pending probe to PingReq on timeout"
```

---

### Task 9: `recv(PingReqMsg)` → ping target, emit `PingReqAck` on result

For Phase 2 simplicity we make the helper *immediately* attempt a synthetic ping by emitting a `PingMsg` to the target and pre-registering a callback via a `pending_helps` map. When the target's `Ack` arrives at the helper, the helper responds to the original requester with `PingReqAck{success=true}`. If the helper's own ack-timeout fires for that synthetic ping, it responds `PingReqAck{success=false}`.

**Files:**
- Modify: `src/model_shard/membership/state.py`
- Modify: `tests/membership/test_state.py`

- [ ] **Step 1: Write the failing test** — append:

```python
def test_recv_pingreq_emits_ping_to_target_and_tracks_help() -> None:
    s = make_state(self_id="helper", peers=("requester", "target"), seed=0)
    msg = PingReqMsg(
        from_shard_id="requester",
        target_shard_id="target",
        probe_id="r:1",
        deltas=[],
    )
    out = s.recv(msg, now=2.0)
    pings = [m for m in out if isinstance(m.payload, PingMsg)]
    assert len(pings) == 1
    assert pings[0].target_shard_id == "target"
    # No PingReqAck yet — we await the target's Ack.
    assert all(not isinstance(m.payload, PingReqAckMsg) for m in out)


def test_recv_target_ack_during_help_emits_pingreqack_success() -> None:
    s = make_state(self_id="helper", peers=("requester", "target"), seed=0)
    s.recv(
        PingReqMsg(
            from_shard_id="requester",
            target_shard_id="target",
            probe_id="r:1",
            deltas=[],
        ),
        now=2.0,
    )
    out = s.recv(
        AckMsg(from_shard_id="target", from_incarnation=0, deltas=[]), now=2.1
    )
    pra = [m for m in out if isinstance(m.payload, PingReqAckMsg)]
    assert len(pra) == 1
    assert pra[0].target_shard_id == "requester"
    payload = pra[0].payload
    assert isinstance(payload, PingReqAckMsg)
    assert payload.success is True
    assert payload.probe_id == "r:1"


def test_help_times_out_emits_pingreqack_failure() -> None:
    s = make_state(self_id="helper", peers=("requester", "target"), seed=0)
    s.recv(
        PingReqMsg(
            from_shard_id="requester",
            target_shard_id="target",
            probe_id="r:1",
            deltas=[],
        ),
        now=2.0,
    )
    # T_TIMEOUT = 500ms — helper gives up at t=2.5
    out = s.tick(now=2.5)
    pra = [m for m in out if isinstance(m.payload, PingReqAckMsg)]
    assert len(pra) == 1
    payload = pra[0].payload
    assert isinstance(payload, PingReqAckMsg)
    assert payload.success is False
    assert pra[0].target_shard_id == "requester"
```

- [ ] **Step 2: Run** — Expected: 3 new failures.

Run: `uv run pytest tests/membership/test_state.py -v`

- [ ] **Step 3: Implement** — add `_PendingHelp` dataclass at top of `state.py`:

```python
@dataclass(frozen=True)
class _PendingHelp:
    probe_id: str
    target_id: str
    requester_id: str
    sent_at: float
```

In `__init__`, add: `self._pending_helps: list[_PendingHelp] = []`.

Extend `recv` to dispatch on `PingReqMsg`:

```python
    def recv(self, msg: IncomingMessage, now: float) -> list[OutgoingMessage]:
        if isinstance(msg, PingMsg):
            return self._handle_ping(msg, now)
        if isinstance(msg, AckMsg):
            return self._handle_ack(msg, now)
        if isinstance(msg, PingReqMsg):
            return self._handle_pingreq(msg, now)
        return []

    def _handle_pingreq(
        self, msg: PingReqMsg, now: float
    ) -> list[OutgoingMessage]:
        if msg.target_shard_id not in self._members:
            return []
        self._pending_helps.append(
            _PendingHelp(
                probe_id=msg.probe_id,
                target_id=msg.target_shard_id,
                requester_id=msg.from_shard_id,
                sent_at=now,
            )
        )
        return [
            OutgoingMessage(
                target_shard_id=msg.target_shard_id,
                payload=PingMsg(
                    from_shard_id=self._self_id,
                    from_incarnation=self._self_incarnation,
                    deltas=[],
                ),
            )
        ]
```

Extend `_handle_ack` to also fulfil any pending helps for that target:

```python
    def _handle_ack(self, msg: AckMsg, now: float) -> list[OutgoingMessage]:
        probe = self._pending_probe
        if probe is not None and probe.target_id == msg.from_shard_id:
            self._pending_probe = None

        out: list[OutgoingMessage] = []
        remaining: list[_PendingHelp] = []
        for h in self._pending_helps:
            if h.target_id == msg.from_shard_id:
                out.append(
                    OutgoingMessage(
                        target_shard_id=h.requester_id,
                        payload=PingReqAckMsg(
                            from_shard_id=self._self_id,
                            target_shard_id=h.target_id,
                            probe_id=h.probe_id,
                            success=True,
                            deltas=[],
                        ),
                    )
                )
            else:
                remaining.append(h)
        self._pending_helps = remaining
        return out
```

Extend `tick` to time out pending helps. Add a new helper called from `tick`:

```python
    def _maybe_timeout_helps(self, now: float) -> list[OutgoingMessage]:
        out: list[OutgoingMessage] = []
        remaining: list[_PendingHelp] = []
        for h in self._pending_helps:
            if now >= h.sent_at + self._cfg.t_timeout_ms / 1000.0:
                out.append(
                    OutgoingMessage(
                        target_shard_id=h.requester_id,
                        payload=PingReqAckMsg(
                            from_shard_id=self._self_id,
                            target_shard_id=h.target_id,
                            probe_id=h.probe_id,
                            success=False,
                            deltas=[],
                        ),
                    )
                )
            else:
                remaining.append(h)
        self._pending_helps = remaining
        return out
```

Call it at the top of `tick`:

```python
    def tick(self, now: float) -> list[OutgoingMessage]:
        out: list[OutgoingMessage] = []
        out.extend(self._maybe_timeout_helps(now))
        out.extend(self._maybe_escalate_probe(now))
        if now >= self._next_period_at:
            out.extend(self._start_protocol_period(now))
        return out
```

- [ ] **Step 4: Run** — Expected: 18 passed.

Run: `uv run pytest tests/membership/test_state.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/model_shard/membership/state.py tests/membership/test_state.py
git commit -m "Phase 2: PingReq fulfilment via pending-help tracking"
```

---

### Task 10: `recv(PingReqAck)` resolves probe; suspect on all-failures

**Files:**
- Modify: `src/model_shard/membership/state.py`
- Modify: `tests/membership/test_state.py`

- [ ] **Step 1: Write the failing test** — append:

```python
def _drive_to_indirect_phase(s: MembershipState) -> _PendingProbe:
    """Helper: advance s to the post-escalation phase and return the probe."""
    s.tick(now=1.0)
    s.tick(now=1.5)  # escalates to PingReq
    probe = s._pending_probe  # type: ignore[attr-defined]
    assert probe is not None
    assert probe.indirect_sent_at is not None
    return probe


def test_positive_pingreqack_clears_probe() -> None:
    s = make_state(peers=("n1", "n2", "n3", "n4"), seed=0)
    probe = _drive_to_indirect_phase(s)
    helper = probe.indirect_targets[0]
    s.recv(
        PingReqAckMsg(
            from_shard_id=helper,
            target_shard_id=probe.target_id,
            probe_id=probe.probe_id,
            success=True,
            deltas=[],
        ),
        now=1.6,
    )
    assert s._pending_probe is None  # type: ignore[attr-defined]


def test_all_negative_pingreqacks_mark_target_suspect() -> None:
    s = make_state(peers=("n1", "n2", "n3", "n4"), seed=0)
    probe = _drive_to_indirect_phase(s)
    for helper in probe.indirect_targets:
        s.recv(
            PingReqAckMsg(
                from_shard_id=helper,
                target_shard_id=probe.target_id,
                probe_id=probe.probe_id,
                success=False,
                deltas=[],
            ),
            now=1.7,
        )
    rec = s.view()[probe.target_id]
    assert rec.state == MemberState.SUSPECT
    assert rec.suspect_deadline is not None
    # deadline = now + T_SUSPECT (4000ms)
    assert abs(rec.suspect_deadline - (1.7 + 4.0)) < 1e-9


def test_partial_negative_pingreqacks_does_not_mark_suspect() -> None:
    s = make_state(peers=("n1", "n2", "n3", "n4"), seed=0)
    probe = _drive_to_indirect_phase(s)
    helper = probe.indirect_targets[0]
    s.recv(
        PingReqAckMsg(
            from_shard_id=helper,
            target_shard_id=probe.target_id,
            probe_id=probe.probe_id,
            success=False,
            deltas=[],
        ),
        now=1.7,
    )
    rec = s.view()[probe.target_id]
    assert rec.state == MemberState.ALIVE
```

- [ ] **Step 2: Run** — Expected: 3 new failures.

Run: `uv run pytest tests/membership/test_state.py -v`

- [ ] **Step 3: Implement** — extend `recv` and add `_handle_pingreqack`:

```python
    def recv(self, msg: IncomingMessage, now: float) -> list[OutgoingMessage]:
        if isinstance(msg, PingMsg):
            return self._handle_ping(msg, now)
        if isinstance(msg, AckMsg):
            return self._handle_ack(msg, now)
        if isinstance(msg, PingReqMsg):
            return self._handle_pingreq(msg, now)
        if isinstance(msg, PingReqAckMsg):
            return self._handle_pingreqack(msg, now)
        return []

    def _handle_pingreqack(
        self, msg: PingReqAckMsg, now: float
    ) -> list[OutgoingMessage]:
        probe = self._pending_probe
        if probe is None or probe.probe_id != msg.probe_id:
            return []

        if msg.success:
            self._pending_probe = None
            return []

        new_acks = probe.indirect_acks + 1
        new_success = probe.indirect_success_seen or msg.success
        if (
            new_acks >= len(probe.indirect_targets)
            and not new_success
        ):
            self._mark_suspect(probe.target_id, now)
            self._pending_probe = None
        else:
            self._pending_probe = replace(
                probe,
                indirect_acks=new_acks,
                indirect_success_seen=new_success,
            )
        return []

    def _mark_suspect(self, shard_id: str, now: float) -> None:
        prev = self._members[shard_id]
        if prev.state in (MemberState.SUSPECT, MemberState.DEAD):
            return
        new = MemberRecord(
            shard_id=shard_id,
            host=prev.host,
            udp_port=prev.udp_port,
            state=MemberState.SUSPECT,
            incarnation=prev.incarnation,
            last_state_change=now,
            suspect_deadline=now + self._cfg.t_suspect_ms / 1000.0,
        )
        self._members[shard_id] = new
        self._transitions.append(
            StateTransition(
                shard_id=shard_id,
                old_state=prev.state,
                new_record=new,
            )
        )
```

- [ ] **Step 4: Run** — Expected: 21 passed.

Run: `uv run pytest tests/membership/test_state.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/model_shard/membership/state.py tests/membership/test_state.py
git commit -m "Phase 2: PingReqAck resolves probe; mark suspect on all-failure"
```

---

### Task 11: `tick()` promotes `suspect → dead` on deadline expiry

**Files:**
- Modify: `src/model_shard/membership/state.py`
- Modify: `tests/membership/test_state.py`

- [ ] **Step 1: Write the failing test** — append:

```python
def test_tick_promotes_suspect_to_dead_at_deadline() -> None:
    s = make_state(peers=("n1", "n2", "n3", "n4"), seed=0)
    probe = _drive_to_indirect_phase(s)
    target = probe.target_id
    for helper in probe.indirect_targets:
        s.recv(
            PingReqAckMsg(
                from_shard_id=helper,
                target_shard_id=target,
                probe_id=probe.probe_id,
                success=False,
                deltas=[],
            ),
            now=1.7,
        )
    # Suspect deadline = 1.7 + 4.0 = 5.7
    s.tick(now=5.7)
    assert s.view()[target].state == MemberState.DEAD


def test_tick_does_not_promote_before_deadline() -> None:
    s = make_state(peers=("n1", "n2", "n3", "n4"), seed=0)
    probe = _drive_to_indirect_phase(s)
    for helper in probe.indirect_targets:
        s.recv(
            PingReqAckMsg(
                from_shard_id=helper,
                target_shard_id=probe.target_id,
                probe_id=probe.probe_id,
                success=False,
                deltas=[],
            ),
            now=1.7,
        )
    s.tick(now=5.0)  # before 1.7 + 4.0 = 5.7
    assert s.view()[probe.target_id].state == MemberState.SUSPECT
```

- [ ] **Step 2: Run** — Expected: 2 failures.

Run: `uv run pytest tests/membership/test_state.py -v`

- [ ] **Step 3: Implement** — add to `tick()` and a helper:

Modify `tick()`:

```python
    def tick(self, now: float) -> list[OutgoingMessage]:
        out: list[OutgoingMessage] = []
        out.extend(self._maybe_timeout_helps(now))
        out.extend(self._maybe_promote_dead(now))
        out.extend(self._maybe_escalate_probe(now))
        if now >= self._next_period_at:
            out.extend(self._start_protocol_period(now))
        return out

    def _maybe_promote_dead(self, now: float) -> list[OutgoingMessage]:
        for shard_id, rec in list(self._members.items()):
            if (
                rec.state == MemberState.SUSPECT
                and rec.suspect_deadline is not None
                and now >= rec.suspect_deadline
            ):
                new = MemberRecord(
                    shard_id=shard_id,
                    host=rec.host,
                    udp_port=rec.udp_port,
                    state=MemberState.DEAD,
                    incarnation=rec.incarnation,
                    last_state_change=now,
                    suspect_deadline=None,
                )
                self._members[shard_id] = new
                self._transitions.append(
                    StateTransition(
                        shard_id=shard_id,
                        old_state=MemberState.SUSPECT,
                        new_record=new,
                    )
                )
        return []
```

- [ ] **Step 4: Run** — Expected: 23 passed.

Run: `uv run pytest tests/membership/test_state.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/model_shard/membership/state.py tests/membership/test_state.py
git commit -m "Phase 2: tick() promotes suspect to dead on deadline expiry"
```

---

### Task 12: Refutation — receiving suspect/dead about self lifts incarnation and broadcasts alive

**Files:**
- Modify: `src/model_shard/membership/state.py`
- Modify: `tests/membership/test_state.py`

The "incoming gossip about a member" surface is via the `deltas` field on Ping/Ack/PingReq/PingReqAck. We've ignored deltas so far; this task starts processing them.

- [ ] **Step 1: Write the failing test** — append:

```python
def _suspect_self_record(self_id: str, incarnation: int) -> MemberRecord:
    return MemberRecord(
        shard_id=self_id,
        host="127.0.0.1",
        udp_port=10000,
        state=MemberState.SUSPECT,
        incarnation=incarnation,
        last_state_change=10.0,
        suspect_deadline=14.0,
    )


def test_recv_ping_with_suspect_self_delta_bumps_own_incarnation() -> None:
    s = make_state(self_id="n0", peers=("n1",))
    delta = _suspect_self_record("n0", incarnation=0)
    msg = PingMsg(from_shard_id="n1", from_incarnation=0, deltas=[delta])
    s.recv(msg, now=10.0)
    assert s._self_incarnation == 1  # type: ignore[attr-defined]
    assert s.view()["n0"].state == MemberState.ALIVE
    assert s.view()["n0"].incarnation == 1


def test_refutation_emits_alive_self_in_ack_deltas() -> None:
    s = make_state(self_id="n0", peers=("n1",))
    delta = _suspect_self_record("n0", incarnation=0)
    msg = PingMsg(from_shard_id="n1", from_incarnation=0, deltas=[delta])
    out = s.recv(msg, now=10.0)
    assert len(out) == 1
    payload = out[0].payload
    assert isinstance(payload, AckMsg)
    refutation = next((d for d in payload.deltas if d.shard_id == "n0"), None)
    assert refutation is not None
    assert refutation.state == MemberState.ALIVE
    assert refutation.incarnation == 1
```

- [ ] **Step 2: Run** — Expected: 2 failures.

Run: `uv run pytest tests/membership/test_state.py -v`

- [ ] **Step 3: Implement** — add a `_apply_deltas` helper and an outgoing-deltas selector. First, refactor `_handle_ping` to apply incoming deltas and include outgoing deltas in the Ack:

```python
    def _handle_ping(self, msg: PingMsg, now: float) -> list[OutgoingMessage]:
        if msg.from_shard_id not in self._members:
            return []
        self._apply_deltas(msg.deltas, now)
        return [
            OutgoingMessage(
                target_shard_id=msg.from_shard_id,
                payload=AckMsg(
                    from_shard_id=self._self_id,
                    from_incarnation=self._self_incarnation,
                    deltas=self._select_outgoing_deltas(),
                ),
            )
        ]
```

Add `_apply_deltas` and `_select_outgoing_deltas`:

```python
    def _apply_deltas(self, deltas: list[MemberRecord], now: float) -> None:
        for d in deltas:
            if d.shard_id == self._self_id:
                self._maybe_refute(d, now)
                continue
            if d.shard_id not in self._members:
                # §5.1 in design: gossip about unknown shard_ids is dropped.
                continue
            self._maybe_apply_peer_delta(d, now)

    def _maybe_refute(self, delta_about_self: MemberRecord, now: float) -> None:
        # If gossip claims we are suspect/dead, refute by lifting our
        # incarnation above whatever the gossip asserts.
        if delta_about_self.state == MemberState.ALIVE:
            return
        if delta_about_self.incarnation < self._self_incarnation:
            return  # stale gossip
        self._self_incarnation = delta_about_self.incarnation + 1
        prev = self._members[self._self_id]
        new = MemberRecord(
            shard_id=self._self_id,
            host=prev.host,
            udp_port=prev.udp_port,
            state=MemberState.ALIVE,
            incarnation=self._self_incarnation,
            last_state_change=now,
            suspect_deadline=None,
        )
        self._members[self._self_id] = new
        self._transitions.append(
            StateTransition(
                shard_id=self._self_id,
                old_state=prev.state,
                new_record=new,
            )
        )

    def _maybe_apply_peer_delta(self, d: MemberRecord, now: float) -> None:
        # Filled in by Task 14 (tiebreaker). Stub for now: accept higher
        # incarnations only.
        prev = self._members[d.shard_id]
        if d.incarnation <= prev.incarnation:
            return
        new = MemberRecord(
            shard_id=d.shard_id,
            host=prev.host,
            udp_port=prev.udp_port,
            state=d.state,
            incarnation=d.incarnation,
            last_state_change=now,
            suspect_deadline=(
                now + self._cfg.t_suspect_ms / 1000.0
                if d.state == MemberState.SUSPECT
                else None
            ),
        )
        self._members[d.shard_id] = new
        self._transitions.append(
            StateTransition(
                shard_id=d.shard_id,
                old_state=prev.state,
                new_record=new,
            )
        )

    def _select_outgoing_deltas(self) -> list[MemberRecord]:
        # Filled in by Task 15 (gossip backlog). For now, always include the
        # current self record so refutations propagate.
        return [self._members[self._self_id]]
```

- [ ] **Step 4: Run** — Expected: 25 passed.

Run: `uv run pytest tests/membership/test_state.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/model_shard/membership/state.py tests/membership/test_state.py
git commit -m "Phase 2: self-refutation via incarnation bump on suspect-self gossip"
```

---

### Task 13: Self-suspicion floor — lift own incarnation above any gossip about self

This refines Task 12's refutation: even when gossip arrives at higher incarnation than our own (after a restart), we lift to `gossip + 1` rather than ignoring. This is what makes restart idempotent without disk persistence.

The implementation in Task 12 already handles this case (`d.incarnation < self._self_incarnation` is the only skip). Add explicit tests to lock the behavior in.

**Files:**
- Modify: `tests/membership/test_state.py`

- [ ] **Step 1: Write the failing test** — append:

```python
def test_refutation_floors_to_higher_incarnation_after_restart() -> None:
    """Simulates: this node was at incarnation 5, was marked dead, restarted
    at incarnation 0. Gossip arrives saying 'n0 dead at inc=5'. Node must
    refute at inc=6, not inc=1."""
    s = make_state(self_id="n0", peers=("n1",))
    assert s._self_incarnation == 0  # type: ignore[attr-defined]
    delta = MemberRecord(
        shard_id="n0",
        host="127.0.0.1",
        udp_port=10000,
        state=MemberState.DEAD,
        incarnation=5,
        last_state_change=10.0,
        suspect_deadline=None,
    )
    s.recv(PingMsg(from_shard_id="n1", from_incarnation=0, deltas=[delta]), now=10.0)
    assert s._self_incarnation == 6  # type: ignore[attr-defined]
    assert s.view()["n0"].incarnation == 6
    assert s.view()["n0"].state == MemberState.ALIVE


def test_stale_gossip_about_self_at_lower_incarnation_is_ignored() -> None:
    s = make_state(self_id="n0", peers=("n1",))
    # First lift our incarnation to 5.
    s.recv(
        PingMsg(
            from_shard_id="n1",
            from_incarnation=0,
            deltas=[
                MemberRecord(
                    shard_id="n0",
                    host="127.0.0.1",
                    udp_port=10000,
                    state=MemberState.SUSPECT,
                    incarnation=4,
                    last_state_change=1.0,
                    suspect_deadline=5.0,
                )
            ],
        ),
        now=1.0,
    )
    assert s._self_incarnation == 5  # type: ignore[attr-defined]
    # Stale dead-at-inc=2 must not affect us.
    s.recv(
        PingMsg(
            from_shard_id="n1",
            from_incarnation=0,
            deltas=[
                MemberRecord(
                    shard_id="n0",
                    host="127.0.0.1",
                    udp_port=10000,
                    state=MemberState.DEAD,
                    incarnation=2,
                    last_state_change=2.0,
                    suspect_deadline=None,
                )
            ],
        ),
        now=2.0,
    )
    assert s._self_incarnation == 5  # type: ignore[attr-defined]
    assert s.view()["n0"].state == MemberState.ALIVE
```

- [ ] **Step 2: Run** — Expected: both pass already (Task 12 implementation covers this). If not, fix `_maybe_refute`.

Run: `uv run pytest tests/membership/test_state.py -v`

- [ ] **Step 3: Commit** (test-only commit; no implementation change expected)

```bash
git add tests/membership/test_state.py
git commit -m "Phase 2: lock self-suspicion floor behavior with explicit tests"
```

---

### Task 14: Same-incarnation tiebreaker — `dead > suspect > alive`

**Files:**
- Modify: `src/model_shard/membership/state.py`
- Modify: `tests/membership/test_state.py`

- [ ] **Step 1: Write the failing test** — append:

```python
def test_same_incarnation_dead_overrides_alive() -> None:
    s = make_state(peers=("n1",))
    # n1 currently alive at inc=0. Gossip says n1 dead at inc=0.
    dead_n1 = MemberRecord("n1", "127.0.0.1", 10001, MemberState.DEAD, 0, 5.0, None)
    s.recv(PingMsg(from_shard_id="n0", from_incarnation=0, deltas=[dead_n1]), now=5.0)
    # Note: n0 sending to itself is a misuse; use a different sender
    s2 = make_state(self_id="me", peers=("n1", "src"))
    s2.recv(
        PingMsg(from_shard_id="src", from_incarnation=0, deltas=[dead_n1]),
        now=5.0,
    )
    assert s2.view()["n1"].state == MemberState.DEAD


def test_same_incarnation_suspect_overrides_alive() -> None:
    s = make_state(self_id="me", peers=("n1", "src"))
    suspect = MemberRecord(
        "n1", "127.0.0.1", 10001, MemberState.SUSPECT, 0, 5.0, 9.0
    )
    s.recv(
        PingMsg(from_shard_id="src", from_incarnation=0, deltas=[suspect]),
        now=5.0,
    )
    assert s.view()["n1"].state == MemberState.SUSPECT


def test_same_incarnation_alive_does_not_override_dead() -> None:
    s = make_state(self_id="me", peers=("n1", "src"))
    # First mark n1 dead via gossip at inc=2.
    dead = MemberRecord("n1", "127.0.0.1", 10001, MemberState.DEAD, 2, 1.0, None)
    s.recv(PingMsg(from_shard_id="src", from_incarnation=0, deltas=[dead]), now=1.0)
    assert s.view()["n1"].state == MemberState.DEAD
    # Now alive gossip at the same inc must not resurrect.
    alive = MemberRecord("n1", "127.0.0.1", 10001, MemberState.ALIVE, 2, 2.0, None)
    s.recv(PingMsg(from_shard_id="src", from_incarnation=0, deltas=[alive]), now=2.0)
    assert s.view()["n1"].state == MemberState.DEAD


def test_higher_incarnation_alive_does_resurrect_dead() -> None:
    s = make_state(self_id="me", peers=("n1", "src"))
    dead = MemberRecord("n1", "127.0.0.1", 10001, MemberState.DEAD, 2, 1.0, None)
    s.recv(PingMsg(from_shard_id="src", from_incarnation=0, deltas=[dead]), now=1.0)
    alive = MemberRecord("n1", "127.0.0.1", 10001, MemberState.ALIVE, 3, 2.0, None)
    s.recv(PingMsg(from_shard_id="src", from_incarnation=0, deltas=[alive]), now=2.0)
    assert s.view()["n1"].state == MemberState.ALIVE
    assert s.view()["n1"].incarnation == 3
```

- [ ] **Step 2: Run** — Expected: failures (current `_maybe_apply_peer_delta` only accepts strictly higher incarnation).

Run: `uv run pytest tests/membership/test_state.py -v`

- [ ] **Step 3: Implement** — replace `_maybe_apply_peer_delta`:

```python
    def _maybe_apply_peer_delta(self, d: MemberRecord, now: float) -> None:
        prev = self._members[d.shard_id]
        if d.incarnation < prev.incarnation:
            return
        if d.incarnation == prev.incarnation and d.state <= prev.state:
            # Same incarnation: only apply if the new state is *more severe*
            # (DEAD > SUSPECT > ALIVE; the IntEnum ordering encodes this).
            return
        new = MemberRecord(
            shard_id=d.shard_id,
            host=prev.host,
            udp_port=prev.udp_port,
            state=d.state,
            incarnation=d.incarnation,
            last_state_change=now,
            suspect_deadline=(
                now + self._cfg.t_suspect_ms / 1000.0
                if d.state == MemberState.SUSPECT
                else None
            ),
        )
        self._members[d.shard_id] = new
        self._transitions.append(
            StateTransition(
                shard_id=d.shard_id,
                old_state=prev.state,
                new_record=new,
            )
        )
```

- [ ] **Step 4: Run** — Expected: all 4 tests pass; total 31 passed.

Run: `uv run pytest tests/membership/test_state.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/model_shard/membership/state.py tests/membership/test_state.py
git commit -m "Phase 2: same-incarnation tiebreaker (dead > suspect > alive)"
```

---

### Task 15: Gossip backlog — track recent transitions, drain into outgoing messages

Replace the placeholder `_select_outgoing_deltas` with a priority queue: every state transition enters the backlog with `priority=0`; each gossip dispatch increments priority and selects up to `K_GOSSIP` lowest-priority entries; entries older than `3 × T_SUSPECT` are GC'd.

**Files:**
- Modify: `src/model_shard/membership/state.py`
- Modify: `tests/membership/test_state.py`

- [ ] **Step 1: Write the failing test** — append:

```python
def test_outgoing_pings_carry_recent_transitions_in_deltas() -> None:
    s = make_state(self_id="me", peers=("n1", "src", "n2"), seed=0)
    # Mark n2 dead via incoming gossip → that adds a transition to the backlog.
    dead = MemberRecord("n2", "127.0.0.1", 10003, MemberState.DEAD, 5, 1.0, None)
    s.recv(PingMsg(from_shard_id="src", from_incarnation=0, deltas=[dead]), now=1.0)
    # Drive a protocol period; expect outgoing Ping to carry the n2-dead delta.
    out = s.tick(now=1.0)
    pings = [m for m in out if isinstance(m.payload, PingMsg)]
    assert len(pings) == 1
    payload = pings[0].payload
    assert isinstance(payload, PingMsg)
    n2_delta = next((d for d in payload.deltas if d.shard_id == "n2"), None)
    assert n2_delta is not None
    assert n2_delta.state == MemberState.DEAD


def test_backlog_caps_at_k_gossip_per_message() -> None:
    cfg = SwimConfig(k_gossip=2)
    s = make_state(
        self_id="me", peers=("a", "b", "c", "d", "src"), seed=0, cfg=cfg
    )
    # Inject 4 transitions (more than K_GOSSIP).
    for i, name in enumerate(("a", "b", "c", "d")):
        d = MemberRecord(name, "127.0.0.1", 10001 + i, MemberState.DEAD, 1, 1.0, None)
        s.recv(PingMsg(from_shard_id="src", from_incarnation=0, deltas=[d]), now=1.0)
    out = s.tick(now=1.0)
    pings = [m for m in out if isinstance(m.payload, PingMsg)]
    payload = pings[0].payload
    assert isinstance(payload, PingMsg)
    # K_GOSSIP=2 entries from backlog, plus self always — 3 total at most.
    assert len(payload.deltas) <= 3


def test_backlog_drains_oldest_first_across_calls() -> None:
    cfg = SwimConfig(k_gossip=1)
    s = make_state(self_id="me", peers=("a", "b", "src"), seed=0, cfg=cfg)
    da = MemberRecord("a", "127.0.0.1", 10001, MemberState.DEAD, 1, 1.0, None)
    db = MemberRecord("b", "127.0.0.1", 10002, MemberState.DEAD, 1, 1.0, None)
    s.recv(PingMsg(from_shard_id="src", from_incarnation=0, deltas=[da]), now=1.0)
    s.recv(PingMsg(from_shard_id="src", from_incarnation=0, deltas=[db]), now=1.0)
    # Two ticks should drain a, then b.
    out1 = s.tick(now=1.0)
    out2 = s.tick(now=2.0)

    def first_non_self(p: PingMsg) -> str | None:
        return next((d.shard_id for d in p.deltas if d.shard_id != "me"), None)

    p1 = out1[0].payload
    p2 = out2[0].payload
    assert isinstance(p1, PingMsg)
    assert isinstance(p2, PingMsg)
    assert first_non_self(p1) == "a"
    assert first_non_self(p2) == "b"
```

- [ ] **Step 2: Run** — Expected: failures.

Run: `uv run pytest tests/membership/test_state.py -v`

- [ ] **Step 3: Implement** — add backlog state to `__init__`:

```python
        self._backlog: list[_BacklogEntry] = []
```

Add `_BacklogEntry` near `_PendingProbe`:

```python
@dataclass
class _BacklogEntry:
    record: MemberRecord
    priority: int  # number of times this entry has been gossiped
    enqueued_at: float
```

Modify the transition-recording call sites (`_mark_suspect`, `_maybe_promote_dead`, `_maybe_refute`, `_maybe_apply_peer_delta`) to also call `self._enqueue_backlog(new, now)` immediately after appending to `self._transitions`. Add the helper:

```python
    def _enqueue_backlog(self, rec: MemberRecord, now: float) -> None:
        # Replace any existing entry for this shard_id with the latest record;
        # priority resets to 0 so the new state propagates.
        self._backlog = [b for b in self._backlog if b.record.shard_id != rec.shard_id]
        self._backlog.append(_BacklogEntry(record=rec, priority=0, enqueued_at=now))
```

Replace `_select_outgoing_deltas`:

```python
    def _select_outgoing_deltas(self) -> list[MemberRecord]:
        # Always include our own current record so refutations and incarnation
        # bumps propagate immediately. Then up to K_GOSSIP backlog entries
        # ordered by ascending priority (oldest gossiped first).
        deltas: list[MemberRecord] = [self._members[self._self_id]]
        self._backlog.sort(key=lambda b: (b.priority, b.enqueued_at))
        for entry in self._backlog[: self._cfg.k_gossip]:
            deltas.append(entry.record)
            entry.priority += 1
        return deltas

    def _gc_backlog(self, now: float) -> None:
        cutoff = 3 * self._cfg.t_suspect_ms / 1000.0
        self._backlog = [b for b in self._backlog if (now - b.enqueued_at) <= cutoff]
```

Call `_gc_backlog(now)` once per `tick()`:

```python
    def tick(self, now: float) -> list[OutgoingMessage]:
        out: list[OutgoingMessage] = []
        self._gc_backlog(now)
        out.extend(self._maybe_timeout_helps(now))
        out.extend(self._maybe_promote_dead(now))
        out.extend(self._maybe_escalate_probe(now))
        if now >= self._next_period_at:
            out.extend(self._start_protocol_period(now))
        return out
```

Also, every Ack/PingReq/PingReqAck must include `_select_outgoing_deltas()` in its `deltas` field — update those return sites in `_handle_ping`, `_handle_ack`, `_handle_pingreq`, `_handle_pingreqack`, `_maybe_timeout_helps`, `_maybe_escalate_probe`, `_start_protocol_period` to pass `deltas=self._select_outgoing_deltas()` instead of `[]`.

- [ ] **Step 4: Run** — Expected: 34 passed.

Run: `uv run pytest tests/membership/test_state.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/model_shard/membership/state.py tests/membership/test_state.py
git commit -m "Phase 2: gossip backlog with priority-queue dissemination"
```

---

### Task 16: `recv(JoinMsg)` → emit `MembershipDeltaMsg`

**Files:**
- Modify: `src/model_shard/membership/state.py`
- Modify: `tests/membership/test_state.py`

- [ ] **Step 1: Write the failing test** — append:

```python
from model_shard.membership.records import JoinMsg, MembershipDeltaMsg


def test_recv_join_emits_membership_delta_with_full_view() -> None:
    s = make_state(self_id="seed", peers=("n1",))
    new_node = MemberRecord(
        shard_id="newcomer",
        host="127.0.0.1",
        udp_port=10099,
        state=MemberState.ALIVE,
        incarnation=0,
        last_state_change=0.0,
        suspect_deadline=None,
    )
    out = s.recv(JoinMsg(self_record=new_node), now=3.0)
    deltas = [m for m in out if isinstance(m.payload, MembershipDeltaMsg)]
    assert len(deltas) == 1
    payload = deltas[0].payload
    assert isinstance(payload, MembershipDeltaMsg)
    ids = {m.shard_id for m in payload.members}
    assert {"seed", "n1", "newcomer"} <= ids
    assert deltas[0].target_shard_id == "newcomer"


def test_recv_join_installs_unknown_newcomer_in_view() -> None:
    s = make_state(self_id="seed", peers=("n1",))
    new_node = MemberRecord(
        shard_id="newcomer",
        host="127.0.0.1",
        udp_port=10099,
        state=MemberState.ALIVE,
        incarnation=0,
        last_state_change=0.0,
        suspect_deadline=None,
    )
    s.recv(JoinMsg(self_record=new_node), now=3.0)
    assert "newcomer" in s.view()
    assert s.view()["newcomer"].state == MemberState.ALIVE
```

- [ ] **Step 2: Run** — Expected: failures.

Run: `uv run pytest tests/membership/test_state.py -v`

- [ ] **Step 3: Implement** — extend `recv` and add `_handle_join`:

```python
    def recv(self, msg: IncomingMessage, now: float) -> list[OutgoingMessage]:
        if isinstance(msg, PingMsg):
            return self._handle_ping(msg, now)
        if isinstance(msg, AckMsg):
            return self._handle_ack(msg, now)
        if isinstance(msg, PingReqMsg):
            return self._handle_pingreq(msg, now)
        if isinstance(msg, PingReqAckMsg):
            return self._handle_pingreqack(msg, now)
        if isinstance(msg, JoinMsg):
            return self._handle_join(msg, now)
        if isinstance(msg, MembershipDeltaMsg):
            return self._handle_delta(msg, now)
        return []

    def _handle_join(self, msg: JoinMsg, now: float) -> list[OutgoingMessage]:
        rec = msg.self_record
        # Install or refresh the newcomer's record.
        prev = self._members.get(rec.shard_id)
        installed = MemberRecord(
            shard_id=rec.shard_id,
            host=rec.host,
            udp_port=rec.udp_port,
            state=MemberState.ALIVE,
            incarnation=rec.incarnation,
            last_state_change=now,
            suspect_deadline=None,
        )
        # If we previously recorded the newcomer as DEAD at higher incarnation,
        # echo *that* record back so the newcomer applies the floor rule.
        if prev is not None and prev.state == MemberState.DEAD and prev.incarnation > rec.incarnation:
            members = list(self._members.values())
        else:
            self._members[rec.shard_id] = installed
            self._transitions.append(
                StateTransition(
                    shard_id=rec.shard_id,
                    old_state=(prev.state if prev else None),
                    new_record=installed,
                )
            )
            self._enqueue_backlog(installed, now)
            members = list(self._members.values())
        return [
            OutgoingMessage(
                target_shard_id=rec.shard_id,
                payload=MembershipDeltaMsg(members=members),
            )
        ]

    def _handle_delta(
        self, msg: MembershipDeltaMsg, now: float
    ) -> list[OutgoingMessage]:
        # Bulk install of records (used by joining nodes after Join).
        for rec in msg.members:
            if rec.shard_id == self._self_id:
                self._maybe_refute(rec, now)
                continue
            prev = self._members.get(rec.shard_id)
            if prev is None:
                self._members[rec.shard_id] = rec
                self._transitions.append(
                    StateTransition(
                        shard_id=rec.shard_id,
                        old_state=None,
                        new_record=rec,
                    )
                )
                self._enqueue_backlog(rec, now)
                continue
            self._maybe_apply_peer_delta(rec, now)
        return []
```

- [ ] **Step 4: Run** — Expected: 36 passed.

Run: `uv run pytest tests/membership/test_state.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/model_shard/membership/state.py tests/membership/test_state.py
git commit -m "Phase 2: Join handler emits MembershipDelta; delta installs on receive"
```

---

### Task 17: Drop gossip about unknown shard_ids

This was already partially handled in `_apply_deltas` (skips unknown). Add explicit tests and a tiny WARNING log.

**Files:**
- Modify: `src/model_shard/membership/state.py`
- Modify: `tests/membership/test_state.py`

- [ ] **Step 1: Write the failing test** — append:

```python
def test_gossip_about_unknown_shard_id_is_dropped(caplog) -> None:
    import logging
    s = make_state(self_id="me", peers=("n1", "src"))
    ghost = MemberRecord(
        "ghost-shard", "10.0.0.99", 10099, MemberState.ALIVE, 0, 1.0, None
    )
    with caplog.at_level(logging.WARNING, logger="model_shard.membership.state"):
        s.recv(
            PingMsg(from_shard_id="src", from_incarnation=0, deltas=[ghost]),
            now=1.0,
        )
    assert "ghost-shard" not in s.view()
    assert any("ghost-shard" in r.message for r in caplog.records)
```

- [ ] **Step 2: Run** — Expected: log assertion fails.

Run: `uv run pytest tests/membership/test_state.py -v`

- [ ] **Step 3: Implement** — at the top of `state.py` add `import logging` and `_LOG = logging.getLogger(__name__)`. Modify `_apply_deltas`:

```python
    def _apply_deltas(self, deltas: list[MemberRecord], now: float) -> None:
        for d in deltas:
            if d.shard_id == self._self_id:
                self._maybe_refute(d, now)
                continue
            if d.shard_id not in self._members:
                _LOG.warning(
                    "dropping gossip about unknown shard_id %r (not in shards.yaml)",
                    d.shard_id,
                )
                continue
            self._maybe_apply_peer_delta(d, now)
```

- [ ] **Step 4: Run** — Expected: 37 passed.

Run: `uv run pytest tests/membership/test_state.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/model_shard/membership/state.py tests/membership/test_state.py
git commit -m "Phase 2: log + drop gossip referencing unknown shard_ids"
```

---

### Task 18: `messages.py` — protobuf ↔ dataclass adapters

**Files:**
- Create: `src/model_shard/membership/messages.py`
- Create: `tests/membership/test_messages.py`

- [ ] **Step 1: Write the failing test**

Create `tests/membership/test_messages.py`:

```python
import pytest

from model_shard._pb import wire_pb2
from model_shard.membership.messages import (
    decode_membership_envelope,
    encode_membership_envelope,
)
from model_shard.membership.records import (
    AckMsg,
    JoinMsg,
    MemberRecord,
    MemberState,
    MembershipDeltaMsg,
    PingMsg,
    PingReqAckMsg,
    PingReqMsg,
)


def _rec(shard_id: str, state: MemberState = MemberState.ALIVE) -> MemberRecord:
    return MemberRecord(
        shard_id=shard_id,
        host="127.0.0.1",
        udp_port=10001,
        state=state,
        incarnation=3,
        last_state_change=1.0,
        suspect_deadline=None,
    )


@pytest.mark.parametrize(
    "msg",
    [
        PingMsg(from_shard_id="a", from_incarnation=2, deltas=[_rec("b")]),
        AckMsg(from_shard_id="a", from_incarnation=2, deltas=[]),
        PingReqMsg(
            from_shard_id="a", target_shard_id="b", probe_id="p1", deltas=[]
        ),
        PingReqAckMsg(
            from_shard_id="a",
            target_shard_id="b",
            probe_id="p1",
            success=True,
            deltas=[_rec("c", MemberState.SUSPECT)],
        ),
        JoinMsg(self_record=_rec("a")),
        MembershipDeltaMsg(members=[_rec("a"), _rec("b", MemberState.DEAD)]),
    ],
)
def test_round_trip_through_protobuf(msg) -> None:
    raw = encode_membership_envelope(msg)
    decoded = decode_membership_envelope(raw)
    assert decoded == msg


def test_decode_unknown_envelope_oneof_returns_none() -> None:
    env = wire_pb2.Envelope()
    env.begin.protocol_version = 1  # not a membership message
    raw = env.SerializeToString()
    assert decode_membership_envelope(raw) is None
```

- [ ] **Step 2: Run** — Expected: ImportError.

Run: `uv run pytest tests/membership/test_messages.py -v`

- [ ] **Step 3: Implement** — create `src/model_shard/membership/messages.py`:

```python
"""Protobuf <-> dataclass adapters for membership messages.

The state machine works in dataclasses (records.py); the wire is protobuf.
Keep these layers separate so the state machine never imports `_pb`.
"""

from __future__ import annotations

from model_shard._pb import wire_pb2
from model_shard.membership.records import (
    AckMsg,
    IncomingMessage,
    JoinMsg,
    MemberRecord,
    MemberState,
    MembershipDeltaMsg,
    PingMsg,
    PingReqAckMsg,
    PingReqMsg,
)

_PROTOCOL_VERSION = 1


def _record_to_pb(r: MemberRecord) -> wire_pb2.MemberRecordPb:
    return wire_pb2.MemberRecordPb(
        shard_id=r.shard_id,
        host=r.host,
        udp_port=r.udp_port,
        state=int(r.state),
        incarnation=r.incarnation,
    )


def _record_from_pb(pb: wire_pb2.MemberRecordPb) -> MemberRecord:
    return MemberRecord(
        shard_id=pb.shard_id,
        host=pb.host,
        udp_port=int(pb.udp_port),
        state=MemberState(int(pb.state)),
        incarnation=int(pb.incarnation),
        last_state_change=0.0,  # wire does not transport this; receiver re-stamps
        suspect_deadline=None,  # similarly, deadlines are recomputed locally
    )


def encode_membership_envelope(msg: IncomingMessage) -> bytes:
    env = wire_pb2.Envelope()
    if isinstance(msg, PingMsg):
        env.ping.protocol_version = _PROTOCOL_VERSION
        env.ping.from_shard_id = msg.from_shard_id
        env.ping.from_incarnation = msg.from_incarnation
        env.ping.deltas.extend(_record_to_pb(d) for d in msg.deltas)
    elif isinstance(msg, AckMsg):
        env.ack.protocol_version = _PROTOCOL_VERSION
        env.ack.from_shard_id = msg.from_shard_id
        env.ack.from_incarnation = msg.from_incarnation
        env.ack.deltas.extend(_record_to_pb(d) for d in msg.deltas)
    elif isinstance(msg, PingReqMsg):
        env.ping_req.protocol_version = _PROTOCOL_VERSION
        env.ping_req.from_shard_id = msg.from_shard_id
        env.ping_req.target_shard_id = msg.target_shard_id
        env.ping_req.probe_id = msg.probe_id
        env.ping_req.deltas.extend(_record_to_pb(d) for d in msg.deltas)
    elif isinstance(msg, PingReqAckMsg):
        env.ping_req_ack.protocol_version = _PROTOCOL_VERSION
        env.ping_req_ack.from_shard_id = msg.from_shard_id
        env.ping_req_ack.target_shard_id = msg.target_shard_id
        env.ping_req_ack.probe_id = msg.probe_id
        env.ping_req_ack.success = msg.success
        env.ping_req_ack.deltas.extend(_record_to_pb(d) for d in msg.deltas)
    elif isinstance(msg, JoinMsg):
        env.join.protocol_version = _PROTOCOL_VERSION
        env.join.self_record.CopyFrom(_record_to_pb(msg.self_record))
    elif isinstance(msg, MembershipDeltaMsg):
        env.membership_delta.protocol_version = _PROTOCOL_VERSION
        env.membership_delta.members.extend(_record_to_pb(d) for d in msg.members)
    else:  # pragma: no cover - exhaustive above
        raise ValueError(f"unsupported membership message type: {type(msg).__name__}")
    return env.SerializeToString()


def decode_membership_envelope(raw: bytes) -> IncomingMessage | None:
    env = wire_pb2.Envelope()
    env.ParseFromString(raw)
    which = env.WhichOneof("payload")
    if which == "ping":
        return PingMsg(
            from_shard_id=env.ping.from_shard_id,
            from_incarnation=int(env.ping.from_incarnation),
            deltas=[_record_from_pb(d) for d in env.ping.deltas],
        )
    if which == "ack":
        return AckMsg(
            from_shard_id=env.ack.from_shard_id,
            from_incarnation=int(env.ack.from_incarnation),
            deltas=[_record_from_pb(d) for d in env.ack.deltas],
        )
    if which == "ping_req":
        return PingReqMsg(
            from_shard_id=env.ping_req.from_shard_id,
            target_shard_id=env.ping_req.target_shard_id,
            probe_id=env.ping_req.probe_id,
            deltas=[_record_from_pb(d) for d in env.ping_req.deltas],
        )
    if which == "ping_req_ack":
        return PingReqAckMsg(
            from_shard_id=env.ping_req_ack.from_shard_id,
            target_shard_id=env.ping_req_ack.target_shard_id,
            probe_id=env.ping_req_ack.probe_id,
            success=bool(env.ping_req_ack.success),
            deltas=[_record_from_pb(d) for d in env.ping_req_ack.deltas],
        )
    if which == "join":
        return JoinMsg(self_record=_record_from_pb(env.join.self_record))
    if which == "membership_delta":
        return MembershipDeltaMsg(
            members=[_record_from_pb(d) for d in env.membership_delta.members]
        )
    return None


__all__ = ["decode_membership_envelope", "encode_membership_envelope"]
```

- [ ] **Step 4: Run**

Run: `uv run pytest tests/membership/test_messages.py -v`
Expected: 7 passed.

- [ ] **Step 5: Lint + types**

Run: `uv run ruff check src tests scripts && uv run mypy src tests scripts`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/model_shard/membership/messages.py tests/membership/test_messages.py
git commit -m "Phase 2: protobuf <-> dataclass adapters for membership messages"
```

---

### Task 19: `transport.py` — UDP socket wrapper with MTU guard

**Files:**
- Create: `src/model_shard/membership/transport.py`
- Create: `tests/membership/test_transport.py`

- [ ] **Step 1: Write the failing test**

Create `tests/membership/test_transport.py`:

```python
import socket
import threading
import time

import pytest

from model_shard.membership.transport import UDPTransport


def _free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_send_and_receive_roundtrip() -> None:
    port_a = _free_udp_port()
    port_b = _free_udp_port()
    received: list[tuple[bytes, tuple[str, int]]] = []
    done = threading.Event()

    def on_recv(data: bytes, addr: tuple[str, int]) -> None:
        received.append((data, addr))
        done.set()

    a = UDPTransport(host="127.0.0.1", port=port_a, on_recv=on_recv)
    b = UDPTransport(host="127.0.0.1", port=port_b, on_recv=lambda *_: None)
    a.start()
    b.start()
    try:
        b.send_to(("127.0.0.1", port_a), b"hello")
        assert done.wait(timeout=1.0)
        assert received[0][0] == b"hello"
        assert received[0][1][1] == port_b
    finally:
        a.stop()
        b.stop()


def test_send_oversize_message_is_dropped(caplog) -> None:
    import logging
    port = _free_udp_port()
    t = UDPTransport(host="127.0.0.1", port=port, on_recv=lambda *_: None)
    t.start()
    try:
        with caplog.at_level(logging.ERROR, logger="model_shard.membership.transport"):
            t.send_to(("127.0.0.1", port + 1), b"x" * 2000)  # > 1400 MTU
        assert any("MTU" in r.message for r in caplog.records)
    finally:
        t.stop()


def test_stop_unblocks_recv_loop() -> None:
    port = _free_udp_port()
    t = UDPTransport(host="127.0.0.1", port=port, on_recv=lambda *_: None)
    t.start()
    t.stop()
    # If stop() doesn't unblock, the thread is still alive after a moment.
    time.sleep(0.5)
    assert not t.is_alive()
```

- [ ] **Step 2: Run** — Expected: ImportError.

Run: `uv run pytest tests/membership/test_transport.py -v`

- [ ] **Step 3: Implement**

Create `src/model_shard/membership/transport.py`:

```python
"""UDP sidecar for SWIM messages. Independent of the TCP envelope used for
activations to avoid head-of-line blocking. One bound socket per node."""

from __future__ import annotations

import logging
import socket
import threading
from collections.abc import Callable
from typing import Final

_LOG = logging.getLogger(__name__)
_MTU_GUARD: Final[int] = 1400  # safe single-datagram size on most networks
_RECV_TIMEOUT_S: Final[float] = 0.25  # short timeout so stop() responds quickly
_RECV_BUFSIZE: Final[int] = 65535  # max UDP datagram


class UDPTransport:
    def __init__(
        self,
        host: str,
        port: int,
        on_recv: Callable[[bytes, tuple[str, int]], None],
    ) -> None:
        self._host = host
        self._port = port
        self._on_recv = on_recv
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.settimeout(_RECV_TIMEOUT_S)
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("UDPTransport already started")
        self._thread = threading.Thread(
            target=self._recv_loop, name="udp-recv", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stopping.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            self._sock.close()
        except OSError:
            pass

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def send_to(self, address: tuple[str, int], payload: bytes) -> None:
        if len(payload) > _MTU_GUARD:
            _LOG.error(
                "dropping oversize UDP message (%d bytes > MTU=%d) to %s:%d",
                len(payload),
                _MTU_GUARD,
                address[0],
                address[1],
            )
            return
        try:
            self._sock.sendto(payload, address)
        except OSError as exc:
            _LOG.warning("UDP sendto %s:%d failed: %s", address[0], address[1], exc)

    def _recv_loop(self) -> None:
        while not self._stopping.is_set():
            try:
                data, addr = self._sock.recvfrom(_RECV_BUFSIZE)
            except TimeoutError:
                continue
            except OSError:
                # socket closed during shutdown
                return
            try:
                self._on_recv(data, addr)
            except Exception:
                _LOG.exception("UDP on_recv callback raised")


__all__ = ["UDPTransport"]
```

- [ ] **Step 4: Run**

Run: `uv run pytest tests/membership/test_transport.py -v`
Expected: 3 passed.

- [ ] **Step 5: Lint + types**

Run: `uv run ruff check src tests scripts && uv run mypy src tests scripts`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/model_shard/membership/transport.py tests/membership/test_transport.py
git commit -m "Phase 2: UDP transport with MTU guard and clean shutdown"
```

---

### Task 20: `bootstrap.py` — sequential seed contact

**Files:**
- Create: `src/model_shard/membership/bootstrap.py`
- Create: `tests/membership/test_bootstrap.py`

- [ ] **Step 1: Write the failing test**

Create `tests/membership/test_bootstrap.py`:

```python
import pytest

from model_shard.membership.bootstrap import (
    BootstrapResult,
    bootstrap_sequential,
)
from model_shard.membership.records import (
    JoinMsg,
    MemberRecord,
    MemberState,
    MembershipDeltaMsg,
)
from model_shard.membership.state import PeerSpec


def _rec(shard_id: str, port: int = 10001) -> MemberRecord:
    return MemberRecord(
        shard_id=shard_id,
        host="127.0.0.1",
        udp_port=port,
        state=MemberState.ALIVE,
        incarnation=0,
        last_state_change=0.0,
        suspect_deadline=None,
    )


def test_bootstrap_skips_self_in_seed_list() -> None:
    self_spec = PeerSpec("me", "127.0.0.1", 10000)
    peers = [self_spec, PeerSpec("a", "127.0.0.1", 10001)]
    sent: list[tuple[str, int]] = []

    def fake_request(addr: tuple[str, int], _payload: bytes, _timeout: float) -> bytes | None:
        sent.append(addr)
        return None  # all seeds time out

    bootstrap_sequential(self_spec, peers, fake_request, timeout_s=0.5)
    # 'me' must not be in sent list.
    assert ("127.0.0.1", 10000) not in sent
    assert ("127.0.0.1", 10001) in sent


def test_bootstrap_returns_first_responding_seed_view() -> None:
    self_spec = PeerSpec("me", "127.0.0.1", 10000)
    peers = [
        self_spec,
        PeerSpec("a", "127.0.0.1", 10001),
        PeerSpec("b", "127.0.0.1", 10002),
    ]
    from model_shard.membership.messages import encode_membership_envelope
    delta = MembershipDeltaMsg(members=[_rec("me", 10000), _rec("a"), _rec("b", 10002)])
    delta_bytes = encode_membership_envelope(delta)

    def fake_request(addr: tuple[str, int], payload: bytes, timeout: float) -> bytes | None:
        # 'a' (port 10001) succeeds; 'b' would too but should never be tried.
        if addr == ("127.0.0.1", 10001):
            return delta_bytes
        raise AssertionError(f"unexpected request to {addr}")

    result = bootstrap_sequential(self_spec, peers, fake_request, timeout_s=0.5)
    assert result.success is True
    assert {m.shard_id for m in result.members} >= {"me", "a", "b"}


def test_bootstrap_returns_failure_when_all_seeds_silent() -> None:
    self_spec = PeerSpec("me", "127.0.0.1", 10000)
    peers = [self_spec, PeerSpec("a", "127.0.0.1", 10001)]

    def fake_request(*_args, **_kwargs):
        return None

    result = bootstrap_sequential(self_spec, peers, fake_request, timeout_s=0.5)
    assert result.success is False
    assert result.members == []
```

- [ ] **Step 2: Run** — Expected: ImportError.

Run: `uv run pytest tests/membership/test_bootstrap.py -v`

- [ ] **Step 3: Implement**

Create `src/model_shard/membership/bootstrap.py`:

```python
"""Sequential seed contact for membership bootstrap.

Reads peer addresses from `shards.yaml` (via the `peers` argument), filters
self out, and tries each in YAML order. The first seed that returns a
`MembershipDelta` reply within the timeout wins; the joining node installs
that view as its initial state and starts the tick loop.

If every seed times out, returns `BootstrapResult(success=False, members=[])`
so the caller can install a single-node view and start anyway.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from model_shard.membership.messages import (
    decode_membership_envelope,
    encode_membership_envelope,
)
from model_shard.membership.records import (
    JoinMsg,
    MemberRecord,
    MemberState,
    MembershipDeltaMsg,
)
from model_shard.membership.state import PeerSpec

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class BootstrapResult:
    success: bool
    members: list[MemberRecord]


# A request function takes (addr, payload_bytes, timeout) and returns the
# response bytes, or None on timeout. Decoupled so tests can stub it.
RequestFn = Callable[[tuple[str, int], bytes, float], bytes | None]


def bootstrap_sequential(
    self_spec: PeerSpec,
    peers: list[PeerSpec],
    request_fn: RequestFn,
    timeout_s: float = 0.5,
) -> BootstrapResult:
    self_record = MemberRecord(
        shard_id=self_spec.shard_id,
        host=self_spec.host,
        udp_port=self_spec.udp_port,
        state=MemberState.ALIVE,
        incarnation=0,
        last_state_change=0.0,
        suspect_deadline=None,
    )
    join_payload = encode_membership_envelope(JoinMsg(self_record=self_record))

    for peer in peers:
        if peer.shard_id == self_spec.shard_id:
            continue
        addr = (peer.host, peer.udp_port)
        _LOG.info("bootstrap: contacting seed %s at %s:%d", peer.shard_id, *addr)
        reply = request_fn(addr, join_payload, timeout_s)
        if reply is None:
            _LOG.info("bootstrap: seed %s timed out", peer.shard_id)
            continue
        decoded = decode_membership_envelope(reply)
        if isinstance(decoded, MembershipDeltaMsg):
            _LOG.info(
                "bootstrap: seed %s replied with %d members",
                peer.shard_id,
                len(decoded.members),
            )
            return BootstrapResult(success=True, members=list(decoded.members))
        _LOG.warning(
            "bootstrap: seed %s replied with unexpected message %r",
            peer.shard_id,
            type(decoded).__name__ if decoded else "None",
        )

    _LOG.warning("bootstrap: all seeds failed; starting in single-node view")
    return BootstrapResult(success=False, members=[])


__all__ = ["BootstrapResult", "RequestFn", "bootstrap_sequential"]
```

- [ ] **Step 4: Run**

Run: `uv run pytest tests/membership/test_bootstrap.py -v`
Expected: 3 passed.

- [ ] **Step 5: Lint + types and commit**

```bash
uv run ruff check src tests scripts && uv run mypy src tests scripts
git add src/model_shard/membership/bootstrap.py tests/membership/test_bootstrap.py
git commit -m "Phase 2: sequential seed bootstrap with single-node fallback"
```

---

### Task 21: `runner.py` — thread loop, observer pattern, exception isolation

**Files:**
- Create: `src/model_shard/membership/runner.py`
- Create: `tests/membership/test_runner.py`

- [ ] **Step 1: Write the failing test**

Create `tests/membership/test_runner.py`:

```python
import socket
import threading
import time

from model_shard.membership.config import SwimConfig
from model_shard.membership.records import MemberState, StateTransition
from model_shard.membership.runner import MembershipRunner
from model_shard.membership.state import PeerSpec


def _free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_runner_starts_and_stops_cleanly() -> None:
    self_spec = PeerSpec("n0", "127.0.0.1", _free_udp_port())
    runner = MembershipRunner(self_spec=self_spec, peers=[], config=SwimConfig())
    runner.start()
    assert runner.is_alive()
    runner.stop()
    assert not runner.is_alive()


def test_runner_observer_fires_on_state_transitions() -> None:
    self_spec = PeerSpec("n0", "127.0.0.1", _free_udp_port())
    peer = PeerSpec("ghost", "127.0.0.1", _free_udp_port())  # never started
    cfg = SwimConfig(t_ping_ms=200, t_timeout_ms=100, t_suspect_ms=400)
    seen: list[StateTransition] = []
    cb_done = threading.Event()

    def cb(t: StateTransition) -> None:
        seen.append(t)
        if t.new_record.state == MemberState.DEAD:
            cb_done.set()

    runner = MembershipRunner(self_spec=self_spec, peers=[peer], config=cfg)
    runner.subscribe(cb)
    runner.start()
    try:
        # Within ~1s the runner should ping ghost, fail, suspect, and dead it.
        assert cb_done.wait(timeout=3.0)
        states = [t.new_record.state for t in seen if t.shard_id == "ghost"]
        assert MemberState.SUSPECT in states
        assert MemberState.DEAD in states
    finally:
        runner.stop()


def test_observer_exception_does_not_wedge_runner() -> None:
    self_spec = PeerSpec("n0", "127.0.0.1", _free_udp_port())
    peer = PeerSpec("ghost", "127.0.0.1", _free_udp_port())
    cfg = SwimConfig(t_ping_ms=200, t_timeout_ms=100, t_suspect_ms=400)
    other_seen = threading.Event()

    runner = MembershipRunner(self_spec=self_spec, peers=[peer], config=cfg)
    runner.subscribe(lambda _t: (_ for _ in ()).throw(RuntimeError("boom")))
    runner.subscribe(lambda _t: other_seen.set())
    runner.start()
    try:
        assert other_seen.wait(timeout=3.0)
    finally:
        runner.stop()
```

- [ ] **Step 2: Run** — Expected: ImportError.

Run: `uv run pytest tests/membership/test_runner.py -v`

- [ ] **Step 3: Implement**

Create `src/model_shard/membership/runner.py`:

```python
"""Threaded runner that drives `MembershipState` against a real UDP transport.

One thread owns both the tick loop (every T_TICK ms) and the receive callback
plumbing (the transport invokes `_on_recv` from its own thread, which posts
work onto an internal queue the runner thread drains).

Observer callbacks are invoked from the runner thread, after `state.tick` /
`state.recv` returns — never reentrantly. Exceptions are caught and logged.
"""

from __future__ import annotations

import logging
import queue
import random
import threading
import time
from collections.abc import Callable
from typing import Final

from model_shard.membership.config import SwimConfig
from model_shard.membership.messages import (
    decode_membership_envelope,
    encode_membership_envelope,
)
from model_shard.membership.records import IncomingMessage, StateTransition
from model_shard.membership.state import MembershipState, PeerSpec
from model_shard.membership.transport import UDPTransport

_LOG = logging.getLogger(__name__)


_INTERNAL_QUEUE_MAX: Final[int] = 4096

ObserverCallback = Callable[[StateTransition], None]


class MembershipRunner:
    def __init__(
        self,
        self_spec: PeerSpec,
        peers: list[PeerSpec],
        config: SwimConfig,
        rng_seed: int | None = None,
    ) -> None:
        self._cfg = config
        self._self_spec = self_spec
        self._peers = peers
        self._rng = random.Random(rng_seed)
        self._state = MembershipState(
            self_spec=self_spec,
            peer_specs=peers,
            rng=self._rng,
            config=config,
        )
        self._addr_by_id: dict[str, tuple[str, int]] = {
            self_spec.shard_id: (self_spec.host, self_spec.udp_port)
        }
        for p in peers:
            self._addr_by_id[p.shard_id] = (p.host, p.udp_port)

        self._inbox: queue.Queue[IncomingMessage] = queue.Queue(_INTERNAL_QUEUE_MAX)
        self._observers: list[ObserverCallback] = []
        self._observers_lock = threading.Lock()
        self._watermark = 0
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None

        self._transport = UDPTransport(
            host=self_spec.host,
            port=self_spec.udp_port,
            on_recv=self._on_recv,
        )

    # --------------------------------------------------------------- lifecycle

    def start(self) -> None:
        self._transport.start()
        self._thread = threading.Thread(
            target=self._run, name="membership-runner", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stopping.set()
        self._transport.stop()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # --------------------------------------------------------------- public API

    def subscribe(self, callback: ObserverCallback) -> None:
        with self._observers_lock:
            self._observers.append(callback)

    @property
    def state(self) -> MembershipState:
        return self._state

    # ---------------------------------------------------------- transport hook

    def _on_recv(self, data: bytes, _addr: tuple[str, int]) -> None:
        decoded = decode_membership_envelope(data)
        if decoded is None:
            return
        try:
            self._inbox.put_nowait(decoded)
        except queue.Full:
            _LOG.warning("membership inbox full; dropping message %s", type(decoded).__name__)

    # ---------------------------------------------------------------- run loop

    def _run(self) -> None:
        tick_period_s = self._cfg.t_tick_ms / 1000.0
        while not self._stopping.is_set():
            now = time.monotonic()
            outgoing = self._state.tick(now)

            # Drain any received messages.
            while True:
                try:
                    msg = self._inbox.get_nowait()
                except queue.Empty:
                    break
                outgoing.extend(self._state.recv(msg, time.monotonic()))

            for o in outgoing:
                addr = self._addr_by_id.get(o.target_shard_id)
                if addr is None:
                    _LOG.warning(
                        "no address for shard_id %r; dropping message",
                        o.target_shard_id,
                    )
                    continue
                self._transport.send_to(addr, encode_membership_envelope(o.payload))

            # Fire observer callbacks for any transitions since last loop.
            new_transitions = self._state.changes_since(self._watermark)
            self._watermark = self._state.transition_watermark
            if new_transitions:
                self._fire_observers(new_transitions)

            self._stopping.wait(tick_period_s)

    def _fire_observers(self, transitions: list[StateTransition]) -> None:
        with self._observers_lock:
            observers = list(self._observers)
        for t in transitions:
            for cb in observers:
                try:
                    cb(t)
                except Exception:
                    _LOG.exception("observer callback raised; suppressing")


__all__ = ["MembershipRunner", "ObserverCallback"]
```

Update `src/model_shard/membership/__init__.py` to re-export `MembershipRunner`:

```python
from model_shard.membership.config import SwimConfig
from model_shard.membership.runner import MembershipRunner
from model_shard.membership.state import MembershipState, PeerSpec

__all__ = ["MembershipRunner", "MembershipState", "PeerSpec", "SwimConfig"]
```

- [ ] **Step 4: Run**

Run: `uv run pytest tests/membership/test_runner.py -v`
Expected: 3 passed (may take ~5s total due to real-time waits).

- [ ] **Step 5: Lint + types**

Run: `uv run ruff check src tests scripts && uv run mypy src tests scripts`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/model_shard/membership/runner.py src/model_shard/membership/__init__.py tests/membership/test_runner.py
git commit -m "Phase 2: MembershipRunner thread with observer pattern and exception isolation"
```

---

### Task 22: Add `udp_port` derivation in `shard_map.py`

The `ShardMap` currently exposes only TCP `port`. The runner needs each peer's UDP port too. Per the spec, derive `udp_port = tcp_port + 1000`. Expose it as a property on `ShardSpec`.

**Files:**
- Modify: `src/model_shard/shard_map.py`
- Modify: `tests/test_shard_map.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_shard_map.py`:

```python
def test_shard_spec_udp_port_is_tcp_port_plus_1000() -> None:
    from model_shard.shard_map import NodeAddress, ShardSpec
    spec = ShardSpec(
        shard_id="x",
        address=NodeAddress(host="127.0.0.1", port=9001),
        start_layer=0,
        end_layer=10,
    )
    assert spec.udp_port == 10001
```

- [ ] **Step 2: Run** — Expected: AttributeError.

Run: `uv run pytest tests/test_shard_map.py -v`

- [ ] **Step 3: Implement** — add a `udp_port` `@property` to `ShardSpec`. Edit `src/model_shard/shard_map.py`:

```python
@dataclass(frozen=True)
class ShardSpec:
    shard_id: str
    address: NodeAddress
    start_layer: int
    end_layer: int

    @property
    def udp_port(self) -> int:
        """SWIM UDP port; derived as tcp_port + 1000.

        See `docs/superpowers/specs/2026-04-16-phase2-gossip-discovery-design.md`
        §7.1. If a future deployment needs an explicit field, add `swim_port`
        to the YAML schema and override this derivation.
        """
        return self.address.port + 1000
```

- [ ] **Step 4: Run**

Run: `uv run pytest tests/test_shard_map.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/model_shard/shard_map.py tests/test_shard_map.py
git commit -m "Phase 2: derive ShardSpec.udp_port = tcp_port + 1000"
```

---

### Task 23: `node.py` — `ENABLE_GOSSIP` toggle and `MembershipRunner` construction

**Files:**
- Modify: `src/model_shard/node.py`
- Create: `tests/test_node_membership.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_node_membership.py`:

```python
"""Unit tests for the node.py / membership integration. Do NOT load the model
— these tests use a stub LoadedModel to keep the suite fast."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock

import pytest

from model_shard.shard_map import NodeAddress, ShardMap, ShardSpec


def _make_shardmap() -> ShardMap:
    return ShardMap(
        {
            "head": ShardSpec("head", NodeAddress("127.0.0.1", 19001), 0, 10),
            "mid": ShardSpec("mid", NodeAddress("127.0.0.1", 19002), 10, 20),
            "tail": ShardSpec("tail", NodeAddress("127.0.0.1", 19003), 20, 30),
        }
    )


def test_node_constructs_membership_runner_when_gossip_enabled(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_GOSSIP", "true")
    from model_shard.node import Node

    sm = _make_shardmap()
    n = Node(
        shard=sm.lookup("head"),
        shard_map=sm,
        loaded_model=MagicMock(),
        total_layers=30,
    )
    assert n.membership is not None
    n.shutdown()


def test_node_does_not_construct_runner_when_gossip_disabled(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_GOSSIP", "false")
    from model_shard.node import Node

    sm = _make_shardmap()
    n = Node(
        shard=sm.lookup("head"),
        shard_map=sm,
        loaded_model=MagicMock(),
        total_layers=30,
    )
    assert n.membership is None
    n.shutdown()
```

- [ ] **Step 2: Run** — Expected: AttributeError on `n.membership`.

Run: `uv run pytest tests/test_node_membership.py -v`

- [ ] **Step 3: Implement** — modify `src/model_shard/node.py`:

Add to imports near the top:

```python
import os

from model_shard.membership import MembershipRunner, PeerSpec, SwimConfig
from model_shard.membership.records import StateTransition
```

In `Node.__init__`, after `self._stopping = threading.Event()` and `self._server_sock = None`, add:

```python
        self._membership: MembershipRunner | None = None
        if _gossip_enabled():
            self._membership = self._build_membership_runner()
```

Add the public property and helpers near the end of the class:

```python
    @property
    def membership(self) -> MembershipRunner | None:
        return self._membership

    def _build_membership_runner(self) -> MembershipRunner:
        self_spec = PeerSpec(
            shard_id=self._shard.shard_id,
            host=self._shard.address.host,
            udp_port=self._shard.udp_port,
        )
        peer_specs = [
            PeerSpec(
                shard_id=sid,
                host=self._shard_map.lookup(sid).address.host,
                udp_port=self._shard_map.lookup(sid).udp_port,
            )
            for sid in self._shard_map.all_shards()
            if sid != self._shard.shard_id
        ]
        return MembershipRunner(
            self_spec=self_spec,
            peers=peer_specs,
            config=SwimConfig(),
        )
```

Modify `serve_forever` to start the runner when present:

```python
    def serve_forever(self) -> None:
        if self._membership is not None:
            self._membership.subscribe(self._on_membership_change)
            self._membership.start()
        # ... existing body unchanged ...
```

Modify `shutdown` to also stop the runner:

```python
    def shutdown(self) -> None:
        self._stopping.set()
        if self._membership is not None:
            self._membership.stop()
```

Add a stub observer (filled in by Task 25):

```python
    def _on_membership_change(self, transition: StateTransition) -> None:
        # Wired in Task 25 to drop/redial TCP peer connections.
        pass
```

Add at module bottom:

```python
def _gossip_enabled() -> bool:
    return os.environ.get("ENABLE_GOSSIP", "true").lower() not in ("0", "false", "no")
```

- [ ] **Step 4: Run**

Run: `uv run pytest tests/test_node_membership.py -v`
Expected: 2 passed.

- [ ] **Step 5: Lint + types**

Run: `uv run ruff check src tests scripts && uv run mypy src tests scripts`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/model_shard/node.py tests/test_node_membership.py
git commit -m "Phase 2: Node constructs MembershipRunner under ENABLE_GOSSIP toggle"
```

---

### Task 24: Admission control — `_handle_begin` rejects when any peer is not `alive`

**Files:**
- Modify: `src/model_shard/node.py`
- Modify: `tests/test_node_membership.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_node_membership.py`:

```python
import io

from model_shard._pb import wire_pb2
from model_shard.envelope import recv_envelope


def test_admission_rejects_when_a_peer_is_dead(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_GOSSIP", "true")
    from model_shard.membership.records import MemberRecord, MemberState
    from model_shard.node import Node

    sm = _make_shardmap()
    n = Node(
        shard=sm.lookup("head"),
        shard_map=sm,
        loaded_model=MagicMock(),
        total_layers=30,
    )
    # Force mid into DEAD state in the runner's view.
    assert n.membership is not None
    members = n.membership.state._members  # type: ignore[attr-defined]
    members["mid"] = MemberRecord(
        "mid", "127.0.0.1", 20002, MemberState.DEAD, 1, 0.0, None
    )

    # Build an in-memory client stream and a BeginRequest.
    buf = io.BytesIO()
    req = wire_pb2.BeginRequest(
        protocol_version=1,
        request_id="req-1",
        sequence_id="seq-1",
        prompt_token_ids=[1, 2, 3],
        sampling=wire_pb2.SamplingParams(greedy=True),
        start_layer=0,
        max_new_tokens=4,
    )
    n._handle_begin(req, buf)  # type: ignore[arg-type]
    buf.seek(0)
    env, _ = recv_envelope(buf)
    assert env.WhichOneof("payload") == "error"
    assert env.error.code == wire_pb2.ERR_SHARD_UNAVAILABLE
    assert "mid" in env.error.detail
    n.shutdown()


def test_admission_passes_when_all_peers_alive(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_GOSSIP", "true")
    from model_shard.node import Node

    sm = _make_shardmap()
    n = Node(
        shard=sm.lookup("head"),
        shard_map=sm,
        loaded_model=MagicMock(),
        total_layers=30,
    )
    # All peers are alive in the initial view, so admission passes.
    # We bail out before MLX work runs by raising in the mock.
    n._lm.embed = MagicMock(side_effect=RuntimeError("mlx not real"))
    buf = io.BytesIO()
    req = wire_pb2.BeginRequest(
        protocol_version=1,
        request_id="req-2",
        sequence_id="seq-2",
        prompt_token_ids=[1, 2, 3],
        sampling=wire_pb2.SamplingParams(greedy=True),
        start_layer=0,
        max_new_tokens=4,
    )
    with pytest.raises(RuntimeError, match="mlx not real"):
        n._handle_begin(req, buf)  # type: ignore[arg-type]
    n.shutdown()
```

- [ ] **Step 2: Run** — Expected: failures (admission check not implemented).

Run: `uv run pytest tests/test_node_membership.py -v`

- [ ] **Step 3: Implement** — modify `_handle_begin` in `node.py`. Right after the `if not self.is_head` block:

```python
        unavailable = self._unavailable_peer()
        if unavailable is not None:
            _send_error(
                client_stream,
                req.request_id,
                wire_pb2.ERR_SHARD_UNAVAILABLE,
                f"shard {unavailable!r} not alive",
            )
            return
```

Add the helper at the bottom of `Node` (before the closing `__all__`):

```python
    def _unavailable_peer(self) -> str | None:
        if self._membership is None:
            return None
        view = self._membership.state.view()
        for sid in self._shard_map.all_shards():
            rec = view.get(sid)
            if rec is None or rec.state.name != "ALIVE":
                return sid
        return None
```

- [ ] **Step 4: Run**

Run: `uv run pytest tests/test_node_membership.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/model_shard/node.py tests/test_node_membership.py
git commit -m "Phase 2: admission control rejects BeginRequest when any peer not alive"
```

---

### Task 25: Observer hook — close TCP on alive→non-alive, redial on dead→alive

**Files:**
- Modify: `src/model_shard/node.py`
- Modify: `tests/test_node_membership.py`

- [ ] **Step 1: Write the failing test** — append:

```python
def test_observer_closes_outbound_on_peer_going_suspect(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_GOSSIP", "true")
    from model_shard.membership.records import (
        MemberRecord,
        MemberState,
        StateTransition,
    )
    from model_shard.node import Node

    sm = _make_shardmap()
    n = Node(
        shard=sm.lookup("head"),
        shard_map=sm,
        loaded_model=MagicMock(),
        total_layers=30,
    )
    # Inject a fake outbound stream so we can assert it gets closed.
    closed = MagicMock()
    n._out_stream = MagicMock(close=closed)  # type: ignore[assignment]
    n._out_sock = MagicMock(close=MagicMock())  # type: ignore[assignment]

    new_rec = MemberRecord("mid", "127.0.0.1", 20002, MemberState.SUSPECT, 0, 0.0, 4.0)
    n._on_membership_change(
        StateTransition(shard_id="mid", old_state=MemberState.ALIVE, new_record=new_rec)
    )

    assert closed.called
    assert n._out_stream is None  # type: ignore[attr-defined]
    n.shutdown()
```

- [ ] **Step 2: Run** — Expected: failure (observer is a stub).

Run: `uv run pytest tests/test_node_membership.py -v`

- [ ] **Step 3: Implement** — replace the stub `_on_membership_change`:

```python
    def _on_membership_change(self, transition: StateTransition) -> None:
        # Only react to transitions involving our downstream peer (the only
        # peer this node actively dials). The membership runner observes ALL
        # transitions; we filter to just the relevant one.
        if transition.shard_id != self._downstream.shard_id:
            return
        new_state = transition.new_record.state
        if new_state.name in ("SUSPECT", "DEAD"):
            _LOG.info(
                "downstream peer %s -> %s; closing outbound TCP",
                transition.shard_id,
                new_state.name,
            )
            self._close_outbound()
        elif new_state.name == "ALIVE" and transition.old_state is not None:
            _LOG.info(
                "downstream peer %s -> ALIVE; outbound TCP will redial on next send",
                transition.shard_id,
            )
            # The lazy `_ensure_out_stream` already redials on next write.
```

- [ ] **Step 4: Run**

Run: `uv run pytest tests/test_node_membership.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/model_shard/node.py tests/test_node_membership.py
git commit -m "Phase 2: observer closes outbound TCP on peer leaving alive"
```

---

### Task 26: Surface `Error{SHARD_UNAVAILABLE, is_final=true}` to client on broken pipe mid-decode

The decode loop in `_drive_decode_loop` calls `_forward_activation`, which calls `_write_out`, which raises `OSError`/`BrokenPipeError` when the outbound stream is closed by the observer. Currently the exception propagates and the request hangs from the client's perspective. We need to catch and emit a structured Error to the client.

**Files:**
- Modify: `src/model_shard/node.py`
- Modify: `tests/test_node_membership.py`

- [ ] **Step 1: Write the failing test** — append:

```python
def test_decode_loop_emits_error_to_client_on_broken_pipe(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_GOSSIP", "true")
    from model_shard.node import Node, _HeadRequestState

    sm = _make_shardmap()
    n = Node(
        shard=sm.lookup("head"),
        shard_map=sm,
        loaded_model=MagicMock(),
        total_layers=30,
    )

    # Set up a fake _drive_decode_loop scenario: a head state pointing at an
    # in-memory client stream. Simulate a broken pipe on _forward_activation.
    buf = io.BytesIO()
    state = _HeadRequestState(client_stream=buf, max_new_tokens=4)
    state.token_queue.put(123)  # one token to process

    monkeypatch.setattr(
        n,
        "_forward_activation",
        MagicMock(side_effect=BrokenPipeError("peer closed")),
    )
    monkeypatch.setattr(n, "_run_my_layers", MagicMock(return_value=MagicMock()))
    n._lm.embed = MagicMock(return_value=MagicMock())  # type: ignore[attr-defined]
    monkeypatch.setattr(
        "model_shard.node.embed_tokens", lambda *_a, **_k: MagicMock()
    )

    with n._state_lock:
        n._kv_caches["req-1"] = []
        n._head_states["req-1"] = state

    n._drive_decode_loop("req-1", state)

    buf.seek(0)
    # Skip the SampledToken envelope (token 123); read the next envelope (the error).
    env1, _ = recv_envelope(buf)
    env2, _ = recv_envelope(buf)
    assert env2.WhichOneof("payload") == "error"
    assert env2.error.code == wire_pb2.ERR_SHARD_UNAVAILABLE
    assert env2.error.is_final if hasattr(env2.error, "is_final") else True
    n.shutdown()
```

(Note: the existing `Error` proto does not have `is_final`. We'll communicate finality by simply not sending more tokens after the error.)

- [ ] **Step 2: Run** — Expected: failure (BrokenPipeError propagates).

Run: `uv run pytest tests/test_node_membership.py -v`

- [ ] **Step 3: Implement** — wrap the inside of `_drive_decode_loop` in a try/except for `OSError`:

```python
    def _drive_decode_loop(
        self, request_id: str, state: _HeadRequestState
    ) -> None:
        try:
            while state.generated < state.max_new_tokens:
                token_id = state.token_queue.get()
                state.generated += 1
                is_final = state.generated >= state.max_new_tokens

                _send_sampled_token_to(
                    state.client_stream,
                    request_id,
                    token_id,
                    position=state.generated - 1,
                    is_final=is_final,
                )

                if is_final:
                    break

                with self._state_lock:
                    cache = self._kv_caches[request_id]
                h = embed_tokens(self._lm, mx.array([[token_id]]))
                h = self._run_my_layers(h, cache)
                self._forward_activation(request_id, h)

            self._broadcast_end(request_id)
        except OSError as exc:
            _LOG.warning("decode loop aborted by broken pipe: %s", exc)
            with contextlib.suppress(OSError):
                _send_error(
                    state.client_stream,
                    request_id,
                    wire_pb2.ERR_SHARD_UNAVAILABLE,
                    f"downstream peer unavailable: {exc}",
                )
            with self._state_lock:
                self._kv_caches.pop(request_id, None)
                self._head_states.pop(request_id, None)
```

- [ ] **Step 4: Run**

Run: `uv run pytest tests/test_node_membership.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/model_shard/node.py tests/test_node_membership.py
git commit -m "Phase 2: emit Error{SHARD_UNAVAILABLE} to client on broken pipe mid-decode"
```

---

### Task 27: Behavioral E2E — three-node convergence + dead detection

This is the first `slow` behavioral test. It spawns three real `Node` processes via the existing `scripts/run_node.py` and asserts cluster behavior over real UDP.

**Files:**
- Create: `tests/membership/test_e2e.py`
- (Optional) Modify: `scripts/run_node.py` if it doesn't already accept the existing CLI shape.

- [ ] **Step 1: Write the failing test**

Create `tests/membership/test_e2e.py`:

```python
"""Behavioral end-to-end tests for the membership layer.

Marked `slow` because each test starts/stops 3 real Python subprocesses
(via `scripts/run_node.py`) and the model is mocked out via the
SHARD_DRY_RUN env var (set inside scripts/run_node.py — see Task 27).
"""

from __future__ import annotations

import contextlib
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
RUN_NODE = REPO / "scripts" / "run_node.py"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _write_shards_yaml(tmp_path: Path) -> tuple[Path, dict[str, int]]:
    """Write a temporary shards.yaml with random ports; return path and tcp ports."""
    head, mid, tail = _free_port(), _free_port(), _free_port()
    cfg = {
        "shards": {
            "head": {"host": "127.0.0.1", "port": head, "start_layer": 0, "end_layer": 10},
            "mid": {"host": "127.0.0.1", "port": mid, "start_layer": 10, "end_layer": 20},
            "tail": {"host": "127.0.0.1", "port": tail, "start_layer": 20, "end_layer": 30},
        }
    }
    p = tmp_path / "shards.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p, {"head": head, "mid": mid, "tail": tail}


def _spawn_node(shard_id: str, shards_yaml: Path) -> subprocess.Popen:
    env = {**os.environ, "ENABLE_GOSSIP": "true", "SHARD_DRY_RUN": "true"}
    return subprocess.Popen(
        [sys.executable, str(RUN_NODE), "--shard", shard_id, "--config", str(shards_yaml)],
        env=env,
    )


@contextlib.contextmanager
def _cluster(tmp_path: Path) -> Iterator[tuple[Path, dict[str, int], dict[str, subprocess.Popen]]]:
    shards_yaml, ports = _write_shards_yaml(tmp_path)
    procs = {sid: _spawn_node(sid, shards_yaml) for sid in ("head", "mid", "tail")}
    try:
        yield shards_yaml, ports, procs
    finally:
        for p in procs.values():
            with contextlib.suppress(ProcessLookupError):
                p.terminate()
        for p in procs.values():
            with contextlib.suppress(subprocess.TimeoutExpired):
                p.wait(timeout=5)


def _query_view(host: str, port: int) -> dict[str, str] | None:
    """Reach into the head's debug HTTP endpoint (added in Task 27 step 3)."""
    import json
    import urllib.request
    try:
        with urllib.request.urlopen(
            f"http://{host}:{port}/membership", timeout=1.0
        ) as resp:
            return {k: v["state"] for k, v in json.loads(resp.read()).items()}
    except Exception:
        return None


@pytest.mark.slow
def test_three_nodes_converge_on_alive(tmp_path: Path) -> None:
    with _cluster(tmp_path) as (_, ports, _):
        debug_port = ports["head"] + 2000  # convention: head exposes /membership at tcp+2000
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            view = _query_view("127.0.0.1", debug_port)
            if view and all(v == "ALIVE" for v in view.values()) and len(view) == 3:
                return
            time.sleep(0.2)
        pytest.fail(f"cluster did not converge within 5s; final view={view}")
```

- [ ] **Step 2: Run** — Expected: failure (no `SHARD_DRY_RUN` support, no debug endpoint).

Run: `uv run pytest -m slow tests/membership/test_e2e.py -v`

- [ ] **Step 3: Implement** — modify `scripts/run_node.py` to support `SHARD_DRY_RUN=true` (skip MLX model load, instantiate `Node` with a `MagicMock` LoadedModel) and to expose a tiny HTTP debug endpoint at `tcp_port + 2000` that serves `/membership`.

Patch outline (add at top of `scripts/run_node.py` after existing imports):

```python
import http.server
import json
import os
import socketserver
import threading


def _start_membership_debug_endpoint(node, debug_port: int) -> None:
    handler_node = node

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/membership":
                self.send_response(404)
                self.end_headers()
                return
            if handler_node.membership is None:
                payload = {}
            else:
                view = handler_node.membership.state.view()
                payload = {
                    sid: {"state": rec.state.name, "incarnation": rec.incarnation}
                    for sid, rec in view.items()
                }
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args: object) -> None:
            pass  # silence

    srv = socketserver.TCPServer(("127.0.0.1", debug_port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
```

In the existing `main` (or equivalent CLI entrypoint), after `Node(...)` construction:

```python
    if os.environ.get("SHARD_DRY_RUN") == "true":
        from unittest.mock import MagicMock
        loaded_model = MagicMock()
    else:
        loaded_model = ...  # existing model load path

    node = Node(shard=spec, shard_map=sm, loaded_model=loaded_model, total_layers=30)
    _start_membership_debug_endpoint(node, debug_port=spec.address.port + 2000)
    node.serve_forever()
```

(If `scripts/run_node.py` doesn't already structure this way, refactor minimally — keep all existing behavior.)

- [ ] **Step 4: Run**

Run: `uv run pytest -m slow tests/membership/test_e2e.py::test_three_nodes_converge_on_alive -v`
Expected: pass within ~5s.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_node.py tests/membership/test_e2e.py
git commit -m "Phase 2: E2E test — three-node convergence on alive"
```

---

### Task 28: E2E — kill detection and rejoin

**Files:**
- Modify: `tests/membership/test_e2e.py`

- [ ] **Step 1: Add tests** — append:

```python
@pytest.mark.slow
def test_kill_one_node_others_detect_dead(tmp_path: Path) -> None:
    with _cluster(tmp_path) as (_, ports, procs):
        debug_port = ports["head"] + 2000
        # Wait for convergence first.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            view = _query_view("127.0.0.1", debug_port)
            if view and all(v == "ALIVE" for v in view.values()) and len(view) == 3:
                break
            time.sleep(0.2)
        else:
            pytest.fail("did not converge before kill")

        procs["mid"].terminate()
        procs["mid"].wait(timeout=3)

        # Within ~7s the head should mark mid dead.
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            view = _query_view("127.0.0.1", debug_port)
            if view and view.get("mid") == "DEAD":
                return
            time.sleep(0.2)
        pytest.fail(f"head did not detect mid dead; final view={view}")


@pytest.mark.slow
def test_killed_node_rejoins_returns_to_alive(tmp_path: Path) -> None:
    with _cluster(tmp_path) as (shards_yaml, ports, procs):
        debug_port = ports["head"] + 2000
        # Converge.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            view = _query_view("127.0.0.1", debug_port)
            if view and all(v == "ALIVE" for v in view.values()):
                break
            time.sleep(0.2)
        # Kill mid.
        procs["mid"].terminate()
        procs["mid"].wait(timeout=3)
        # Wait for dead.
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            view = _query_view("127.0.0.1", debug_port)
            if view and view.get("mid") == "DEAD":
                break
            time.sleep(0.2)
        # Restart mid.
        procs["mid"] = _spawn_node("mid", shards_yaml)
        # Within ~5s mid should be alive again.
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            view = _query_view("127.0.0.1", debug_port)
            if view and view.get("mid") == "ALIVE":
                return
            time.sleep(0.2)
        pytest.fail(f"mid did not rejoin to alive; final view={view}")
```

- [ ] **Step 2: Run**

Run: `uv run pytest -m slow tests/membership/test_e2e.py -v`
Expected: 3 passed (~25s total).

- [ ] **Step 3: Commit**

```bash
git add tests/membership/test_e2e.py
git commit -m "Phase 2: E2E tests — kill detection and rejoin"
```

---

### Task 29: E2E — admission control and bootstrap fallback

**Files:**
- Modify: `tests/membership/test_e2e.py`

- [ ] **Step 1: Add tests** — append:

```python
@pytest.mark.slow
def test_bootstrap_with_unreachable_seeds_starts_in_single_node_view(tmp_path: Path) -> None:
    """Spawn ONE node whose shards.yaml lists two non-running peers.
    The node must boot and report itself alive; the missing peers should be
    detected as suspect/dead within T_SUSPECT."""
    shards_yaml, ports = _write_shards_yaml(tmp_path)
    debug_port = ports["head"] + 2000
    head_proc = _spawn_node("head", shards_yaml)
    try:
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            view = _query_view("127.0.0.1", debug_port)
            if (
                view
                and view.get("head") == "ALIVE"
                and view.get("mid") in ("SUSPECT", "DEAD")
            ):
                return
            time.sleep(0.2)
        pytest.fail(f"single-node bootstrap fallback failed; final view={view}")
    finally:
        head_proc.terminate()
        head_proc.wait(timeout=3)
```

- [ ] **Step 2: Run**

Run: `uv run pytest -m slow tests/membership/test_e2e.py::test_bootstrap_with_unreachable_seeds_starts_in_single_node_view -v`
Expected: pass.

- [ ] **Step 3: Commit**

```bash
git add tests/membership/test_e2e.py
git commit -m "Phase 2: E2E test — bootstrap fallback to single-node view"
```

---

### Task 30: Phase 1 regression net — Tier 1 and Tier 2 still pass with membership running

**Files:**
- (No new code; this task verifies existing slow tests pass with `ENABLE_GOSSIP=true`.)

- [ ] **Step 1: Run the full slow suite**

Run: `ENABLE_GOSSIP=true uv run pytest -m slow -v`
Expected: every Phase 1 slow test (`test_distributed_pipeline`, `test_tier1_tokens`, `test_tier2_hidden`, etc.) passes. Total ~3 minutes.

- [ ] **Step 2: If any Phase 1 test fails**

Investigate. Likely causes:
- Admission control firing during cluster warm-up — resolve by giving the test a 3s grace period before sending `BeginRequest`, or by ensuring membership has converged.
- Observer closing TCP unexpectedly — verify the observer only acts on the *downstream* peer's transitions.

Fix only the failing path; do not modify Phase 1 test assertions.

- [ ] **Step 3: Add a regression marker test** — append to `tests/membership/test_e2e.py`:

```python
@pytest.mark.slow
def test_phase1_tier1_passes_with_membership_running(monkeypatch) -> None:
    """Smoke regression: import the Phase 1 Tier 1 module and run one case
    with ENABLE_GOSSIP=true to ensure the membership layer doesn't break it."""
    monkeypatch.setenv("ENABLE_GOSSIP", "true")
    # Delegate to the Phase 1 test module's parametrized first case.
    from tests import test_tier1_tokens  # type: ignore[import-not-found]
    # Phase 1 test_tier1_tokens.test_token_exact uses 5 prompts; run the first.
    if hasattr(test_tier1_tokens, "PROMPTS"):
        prompt = test_tier1_tokens.PROMPTS[0]
        test_tier1_tokens.test_token_exact(prompt)  # type: ignore[attr-defined]
```

(If the Phase 1 module's API differs, adjust the call to match.)

- [ ] **Step 4: Commit**

```bash
git add tests/membership/test_e2e.py
git commit -m "Phase 2: Phase 1 Tier 1 regression test under ENABLE_GOSSIP"
```

---

### Task 31: Final acceptance — full lint, mypy, fast + slow, single 3-node smoke

- [ ] **Step 1: Lint and types — clean**

Run: `uv run ruff check src tests scripts`
Expected: `All checks passed!`
Run: `uv run mypy src tests scripts`
Expected: `Success: no issues found in N source files.`

- [ ] **Step 2: Fast suite — green**

Run: `uv run pytest -v`
Expected: all fast tests pass in <5s.

- [ ] **Step 3: Slow suite — green**

Run: `uv run pytest -m slow -v`
Expected: all slow tests pass.

- [ ] **Step 4: Manual smoke — three real nodes for 30s**

Open 4 terminals.

Terminal 1: `ENABLE_GOSSIP=true uv run python scripts/run_node.py --shard layer_0-10 --config config/shards.yaml`
Terminal 2: `ENABLE_GOSSIP=true uv run python scripts/run_node.py --shard layer_10-20 --config config/shards.yaml`
Terminal 3: `ENABLE_GOSSIP=true uv run python scripts/run_node.py --shard layer_20-30 --config config/shards.yaml`
Terminal 4: `curl -s http://127.0.0.1:11001/membership | python -m json.tool`

Expected output: all three shards `ALIVE`. Then `kill -9` the layer_10-20 process; within ~7s, terminal 4 query reports it `DEAD`. Restart it; within ~3s, returns to `ALIVE`.

- [ ] **Step 5: Update README with Phase 2 status** — append a one-paragraph note to `README.md`:

```markdown
## Phase 2 status: Gossip Discovery — complete

Each node now runs a SWIM-style membership protocol over UDP (port `tcp_port + 1000`).
The head admits `BeginRequest`s only when every required shard is `ALIVE`; in-flight
requests fail with `Error{SHARD_UNAVAILABLE, is_final=true}` if a peer transitions
out of `ALIVE` mid-decode. Set `ENABLE_GOSSIP=false` to bypass and reproduce Phase 1
behavior. See `docs/superpowers/specs/2026-04-16-phase2-gossip-discovery-design.md`.
```

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "Phase 2 complete: Gossip Discovery (SWIM membership)"
```

- [ ] **Step 7: Update memory** — Phase 2 is now complete. Mention to the operator that they may want to update `~/.claude/projects/-Users-lukechang-Github-model-shard/memory/project_gossip_moe.md` to mark Phase 2 done and Phase 3 (Expert-Level Sharding) next.

---

## Self-Review (run after writing the full plan)

### 1. Spec coverage

| Spec section | Implemented in tasks |
|---|---|
| §1.3 D1 — Scope: membership only | Tasks 17, 22 (shards.yaml unchanged role); whole plan never gossips shard layout |
| §1.3 D2 — SWIM core, no Lifeguard | Tasks 5–11, 15 |
| §1.3 D3 — UDP sidecar transport | Tasks 19, 22 |
| §1.3 D4 — Admission + fail-fast on in-flight | Tasks 24, 26 |
| §1.3 D5 — Sequential bootstrap from shards.yaml | Task 20 |
| §1.3 D6 — Pure state machine + thin behavioral | Tasks 4–17 (state), 27–30 (behavioral) |
| §3.1 MembershipState public API | Task 4 (skeleton), 5–17 (mutators) |
| §3.2 6 wire messages + Error code | Task 1 |
| §3.3 MembershipRunner with observers | Task 21 |
| §3.4 Default constants | Task 2 |
| §3.5 node.py integration | Tasks 23–26 |
| §4.1 steady state | Tasks 5–7 |
| §4.2 cold start | Task 20 |
| §4.3 failure detection | Tasks 8–11 |
| §4.4 mid-decode failure | Task 26, E2E in Task 28 |
| §4.5 rejoin | Task 28 (rejoin E2E) |
| §5.1 edge cases handled | Tasks 12–17, 19 (MTU) |
| §5.3 logging policy | Tasks 17, 19 (warnings); INFO via runner observer side |
| §6.1 fast suite | Tasks 4–17 |
| §6.2 behavioral suite (10 cases) | Tasks 27–30 cover cases 1–3, 4–6, 8–9 |
| §6.4 CI shape | Task 31 |
| §7.1 migration / port derivation | Task 22 |
| §7.2 ENABLE_GOSSIP rollback | Task 23 |
| §9 acceptance | Task 31 |

**Gaps to flag:** behavioral cases 4 (admission), 5 (in-flight failure), 7 (simultaneous restart), and 10 (observer exception) from spec §6.2 are covered by *unit* tests in Tasks 24, 26, 21, 21 respectively, not by the slow E2E suite. This is a deliberate trade — the unit tests are fully deterministic (mocking subprocess restarts is brittle), and the spec's intent (catch the regression) is met. If the operator prefers a strict literal mapping, adding three more E2E cases to Task 28/29 is mechanical.

### 2. Placeholder scan

Searched for "TBD", "TODO", "implement later", "fill in", "similar to". No occurrences in task bodies. The placeholder note in Task 12's `_select_outgoing_deltas` is intentional — it gets replaced in Task 15.

### 3. Type / name consistency

- `MemberRecord` field names match across `records.py`, `state.py`, `messages.py`.
- `MembershipState` API: `view()`, `tick()`, `recv()`, `changes_since()`, `transition_watermark` — referenced consistently.
- `_PendingProbe` and `_PendingHelp` introduced in Tasks 5/9 and never renamed.
- `MembershipRunner` public API: `start`, `stop`, `is_alive`, `subscribe`, `state` — consistent in Tasks 21, 23, 25.
- `OutgoingMessage.target_shard_id` vs. `payload` — used uniformly.
- Wire field naming: snake_case in protobuf, mapped to camelCase nowhere (Python protobuf bindings preserve snake_case).

### 4. Scope check

The plan covers a single subsystem (SWIM membership for the existing 3-node cluster). It does **not** attempt shard-map propagation, hot-plane gossip, or retry — all explicitly deferred per the spec. Plan size is appropriate for a single implementation cycle (~31 tasks, mostly small).

---
