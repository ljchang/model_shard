# Phase 2 — Gossip Discovery: Design

**Status:** Proposed
**Date:** 2026-04-16
**Phase of:** [Gossip MoE Inference roadmap](../../../README.md) (6 phases)
**Predecessor:** Phase 1 (Static Pipeline) — complete at commit `0caf928`.
**Successor (next phase):** Phase 3 — Expert-Level Sharding.
**Authoritative project spec:** `/Users/lukechang/Downloads/gossip-moe-inference-spec.md` §5 (Gossip Protocol).

---

## 1. Goals and non-goals

### 1.1 In scope

- Each node runs a SWIM-style failure detector and maintains a live view of which other nodes are reachable.
- The head node refuses new `BeginRequest` admissions when any required shard's owning node is not `alive`.
- In-flight requests fail cleanly with a structured `Error{SHARD_UNAVAILABLE}` envelope when a peer transitions out of `alive` mid-decode (no hangs, no retry).
- Gossip rides on a new UDP transport, separate from the existing TCP activation channel, to avoid head-of-line blocking.
- Phase 1's Tier 1 (token-exact) and Tier 2 (per-layer hidden state) acceptance suites continue to pass with the membership layer running.

### 1.2 Explicit non-goals (deferred to later phases)

| Feature | Deferred to | Reason |
|---|---|---|
| Shard-map propagation (gossip the layer→node mapping) | Phase 3 | Keeps Phase 1 `ShardMap` interface untouched; `shards.yaml` stays authoritative. |
| Hot-plane gossip (load, queue depth, expert heat) | Phase 4 | Different convergence requirements; design space wide open. |
| Retry on rejoin | Phase 6 | Requires the shard-map propagation that Phase 3 introduces. |
| Lifeguard SWIM extensions (refutation acceleration, dogpile suppression, awareness) | Phase 6 | Disproportionate complexity vs. localhost-dev value. |
| Asymmetric-partition handling, accusation suppression | Phase 6 | Localhost can't reproduce the conditions. |
| TLS / message authentication | Phase 6 | Trust model is "all nodes are mine" for the prototype. |
| mDNS / multicast discovery | Later | `shards.yaml` covers our scale; mDNS is a future *alternate* bootstrap, not a replacement. |
| Discrete-event simulator for N-node convergence proofs | Phase 6 | Pure state-machine tests verify protocol invariants; trust SWIM literature for asymptotics. |

### 1.3 Locked design decisions

| # | Decision | Rationale |
|---|---|---|
| 1 | Scope = membership only (no shard-map gossip) | Smallest viable slice; preserves `ShardMap` interface; Phase 1 acceptance tests act as a regression net. |
| 2 | Protocol = SWIM core (no Lifeguard) | Spec §5 says "SWIM-style"; standard literature; correct failure detector for heterogeneous hardware. |
| 3 | Transport = UDP sidecar (separate from TCP activations) | Avoids head-of-line blocking behind large activation payloads; canonical SWIM. |
| 4 | Integration = admission control + fail-fast on in-flight, no retry | Visible behavior change without re-opening Phase 3 scope. |
| 5 | Bootstrap = sequential seed contact reusing `shards.yaml` | Zero new config; deterministic for tests; mechanical migration path to a true seed list later. |
| 6 | Testing = pure state machine + thin behavioral layer | Fast, deterministic, forward-compatible with Phase 6 simulator. |

---

## 2. Architecture and file layout

A new `membership/` Python package, parallel to existing modules. The SWIM core is **pure** (no I/O); the runner and transport adapt it to threads and UDP sockets. Existing files receive only small surgical hooks.

```
src/model_shard/
├── (Phase 1 files — unchanged in structure)
├── node.py                ← gains MembershipObserver hook (~30 new lines)
└── membership/
    ├── __init__.py
    ├── state.py           Pure SWIM state machine. No imports from socket,
    │                      threading, or time. Takes `now: float` as input;
    │                      returns (new_state, [outgoing_msg]).
    ├── messages.py        Message dataclasses + protobuf ser/de. Lives
    │                      outside state.py so the state machine doesn't
    │                      depend on protobuf.
    ├── transport.py       UDPTransport: bind, send_to(addr, bytes),
    │                      recv_loop(callback). MTU-aware: drop+log if a
    │                      message would exceed 1400 bytes.
    ├── runner.py          MembershipRunner: owns one thread that calls
    │                      state.tick(now) on a 100ms timer and routes
    │                      transport callbacks into state.recv(...).
    │                      Notifies observers on state transitions.
    └── bootstrap.py       Sequential seed contact logic. Reads peer
                           addresses from ShardMap (skipping self).

proto/wire.proto            ← gains 6 new message types in Envelope oneof.
config/shards.yaml          ← unchanged. Implicitly becomes the seed list.
tests/membership/
├── test_state.py           Fast pure state machine tests (~40 cases).
└── test_e2e.py             Slow behavioral tests (~10 cases, marked slow).
```

### 2.1 Dependency direction

```
node.py
   ↓ (subscribes / read-only view)
runner.py ──→ state.py ──→ messages.py
   ↓                              ↑
transport.py  ────────────────────┘
   ↓
(OS UDP sockets)
```

`state.py` and `runner.py` know nothing about MLX, the inference engine, or activations. The membership layer is a self-contained subsystem that `node.py` *consumes*.

### 2.2 Threading and runtime model

Phase 1 is thread-per-connection (no event loop). Phase 2 adds **one** new thread per node — the membership runner thread — driving the SWIM tick loop and a tight UDP `recvfrom` loop. No `asyncio` introduction. Refactoring to `asyncio` is an explicit non-goal of Phase 2.

### 2.3 Symmetry

Membership runs on every node, including the head. There is no "membership coordinator." The head just happens to also be the node clients connect to.

---

## 3. Components

### 3.1 `MembershipState` (pure core)

```python
@dataclass
class MemberRecord:
    shard_id: str          # stable identity (matches shards.yaml key)
    address: NodeAddress   # host + UDP port for SWIM
    state: Literal["alive", "suspect", "dead"]
    incarnation: int       # bumped by the member itself on refutation
    last_state_change: float        # monotonic seconds
    suspect_deadline: float | None  # when suspect → dead promotion fires

class MembershipState:
    def __init__(
        self,
        self_id: str,
        self_address: NodeAddress,
        peers: list[ShardSpec],
        rng: random.Random,
        config: SwimConfig,
    ) -> None: ...
    def tick(self, now: float) -> list[OutgoingMessage]: ...
    def recv(self, msg: IncomingMessage, now: float) -> list[OutgoingMessage]: ...
    def local_event(self, evt: LocalEvent, now: float) -> list[OutgoingMessage]: ...
    def view(self) -> dict[str, MemberRecord]: ...
    def changes_since(self, watermark: int) -> list[StateTransition]: ...
```

Key properties:
- All randomness flows through the injected `rng`. Tests pin the seed for reproducibility.
- All time advancement flows through `now` parameters. No `time.monotonic()` calls inside `state.py`.
- All I/O flows through the returned `OutgoingMessage` lists. No socket calls inside `state.py`.

### 3.2 Wire messages (added to `wire.proto` Envelope oneof)

| Message | Direction | Purpose |
|---|---|---|
| `Ping` | A → B | "Are you alive? My incarnation is N. Here are K membership updates." |
| `Ack` | B → A | "Yes, my incarnation is M. Here are K membership updates back." |
| `PingReq` | A → C | "Please ping B for me; I couldn't reach it directly." |
| `PingReqAck` | C → A | Forwards B's ack, or signals "I also couldn't reach B." |
| `Join` | new node → seed | "I am joining; here is my full record." |
| `MembershipDelta` | seed → new node | One-shot push of seed's full membership view. |

Every steady-state message piggybacks up to `K_GOSSIP=3` recent membership changes, selected by a small priority queue keyed on "how many times has this update been gossiped." This is what gives O(log N) convergence as a side effect of the failure detector — no separate gossip cycle needed.

The existing `Error` envelope also gains one new `code` enum value, `SHARD_UNAVAILABLE`, used by `node.py` (§3.5) to signal admission rejection or in-flight cancellation when a peer is not `alive`.

### 3.3 `MembershipRunner`

Owns one thread with two responsibilities:

1. **Tick loop.** Every `T_TICK=100ms`, call `state.tick(time.monotonic())`, hand the returned messages to `UDPTransport.send_to`. The 100ms cadence is the *clock granularity*; actual ping interval (`T_PING=1000ms`) and other timers are gated inside `state.tick`.
2. **Receive loop.** A second short loop driven by the transport's callback — when a UDP datagram arrives, decode to a message, call `state.recv(...)`, send the responses.

Plus an **observer pattern**: `runner.subscribe(callback)`. Callbacks fire on state transitions (`alive→suspect`, `suspect→dead`, `dead→alive`). `node.py` subscribes to drop and redial peer TCP connections in response.

Observer invariants:
- Callbacks are invoked **after** `state.recv` / `state.tick` returns, never reentrantly from inside the state machine.
- Each callback is wrapped in `try/except`; an exception is logged and does not wedge the runner or other observers.

### 3.4 Default timing constants

| Constant | Default | Source |
|---|---|---|
| `T_PING` (interval between pings) | 1000ms | Memberlist default |
| `T_TICK` (state machine clock) | 100ms | Sub-second precision |
| `T_TIMEOUT` (ack deadline) | 500ms | Half of T_PING |
| `K_INDIRECT` (ping-req fanout) | 2 | SWIM paper |
| `T_SUSPECT` (suspect → dead deadline) | 4 × T_PING = 4000ms | SWIM paper |
| `K_GOSSIP` (deltas per message) | 3 | Empirical |

All values configurable via `SwimConfig`; defaults set for the localhost 3-node case but reasonable up to ~30 nodes.

### 3.5 `node.py` integration surface

The only Phase 1 file that changes. Three additions:

1. **Construction:** `Node.__init__` now also constructs a `MembershipRunner`, passes itself as observer.
2. **Admission:** `_handle_begin_request` checks `runner.state.view()` — if any shard in `shards.yaml` is not `alive`, reject with `Error{code: SHARD_UNAVAILABLE, detail: "shard X is dead"}`.
3. **Failure cascade:** `on_membership_change(transition)` — when a peer goes `alive → suspect | dead`, close the persistent TCP connection to that peer; in-flight reads/writes raise `BrokenPipeError`, which is caught and surfaced to the client as `Error{SHARD_UNAVAILABLE, is_final=true}`. When a peer goes `dead → alive`, redial the TCP connection.

No changes to `mlx_engine.py`, `transport.py`, `envelope.py`, `request.py`, `shard.py`, or `shard_map.py`.

---

## 4. Data flow

### 4.1 Steady state

Every 1s, each node picks a random alive peer and pings it with a 500ms ack deadline. A successful ack within the deadline produces no state transition. A missed ack escalates to indirect probing (§4.3) — liveness is determined per-probe, not by tracking last-seen freshness. Network rate at 3 nodes: ~450 B/s per direction per node.

### 4.2 Cold start / join

```
1. Read shards.yaml → list of all shards including self.
2. Filter out self by shard_id → seed candidates in YAML order.
3. For each seed sequentially:
     a. UDP-send Join{shard_id, address, incarnation=0} with 500ms timeout.
     b. On MembershipDelta reply: install seed's view as initial state,
        mark self as alive, start tick loop. Done.
     c. On timeout: try next seed.
4. If all seeds fail: log warning, install single-node view (just self alive),
   start tick loop. Other nodes will discover us when they ping us.
```

Important property: a node always starts. Bootstrap failure is non-fatal. This enables recovery from "all nodes restart simultaneously" — they each fail to find each other initially, then converge on first successful ping.

### 4.3 Failure detection (suspect → dead)

```
T+0ms     A pings B. B's process is wedged or killed.
T+500ms   A's ack deadline expires. A enters indirect-probe mode.
T+500ms   A picks K_INDIRECT=2 peers (C, D), sends each a PingReq{target: B}.
T+1000ms  Neither C nor D gets an ack from B; both reply PingReqAck{success: false}.
T+1000ms  A marks B suspect locally. suspect_deadline = T+5000ms.
          A piggybacks the suspect-B transition in its next outgoing pings.
T+1000ms  Other nodes receive the suspect-B delta and install it,
          each running their own independent suspect_deadline.
T+5000ms  A's suspect_deadline expires. A marks B dead. Observer fires.
T+5000ms  Same fires on each other node within ~one gossip round.
```

**Refutation:** if B is actually fine (just slow), B receives suspect-B gossip from some peer, recognizes the gossip is about itself, bumps its incarnation by 1, and broadcasts `alive(incarnation=N+1)`. The higher incarnation wins last-writer-wins; every node clears the suspicion.

### 4.4 Mid-decode failure (the user-visible scenario)

```
T+0      Client → head: BeginRequest. Head admits (all shards alive).
T+0      Decode loop starts. Hidden states stream head→mid→tail.
T+50ms   Tail produces SampledToken[0]; head streams it to client.
T+100ms  Mid is killed. Head's TCP write to mid succeeds at the OS level
         (kernel buffer) but no data flows. Decode loop blocks.
         Head's MembershipRunner pings mid every 1s.
T+1100ms Ping to mid times out → indirect ping-reqs to tail.
T+1600ms Tail can't reach mid. Head marks mid suspect.
T+5600ms Suspect_deadline → mid marked dead. Observer fires.
         node.py closes head's TCP conn to mid.
         Decode-loop write/read raises BrokenPipeError.
         Head sends Error{SHARD_UNAVAILABLE, is_final=true} to client.
T+5600ms Head's admission control now refuses new BeginRequests.
```

Total wall-clock kill-to-client-error: ~5.5s, dominated by `T_SUSPECT`. Tunable trade-off: lower `T_SUSPECT` reduces this latency at the cost of more false positives. Default kept.

### 4.5 Rejoin

```
1. mid restarts. Reads shards.yaml. Sequential bootstrap to head.
2. Head receives Join{shard_id: mid, incarnation: 0}.
3. Head sees mid is currently dead in its view at incarnation N > 0.
   Head replies with MembershipDelta containing mid's own dead record.
4. mid sees gossip about itself with state=dead, lifts its own incarnation
   to N+1 (self-suspicion floor rule), broadcasts alive(N+1).
5. Within one gossip round (~1s), all nodes mark mid alive.
6. Each node's observer fires for mid: dead → alive.
   node.py redials TCP connection to mid.
7. Head's admission control allows new BeginRequests.
```

Total rejoin time: ~1–2s.

---

## 5. Edge cases

### 5.1 Handled in design

| Scenario | Behavior |
|---|---|
| UDP packet loss | Indirect probing through K=2 peers; requires ≥3 simultaneous losses to misdiagnose. |
| Concurrent same-incarnation conflicting updates | Tiebreaker: `dead > suspect > alive`. Deterministic. |
| Gossip referencing unknown shard_id | Dropped with warning log. `shards.yaml` is authoritative under membership-only scope. |
| Self-suspicion | Bump own incarnation, refute. |
| Self-suspicion floor on rapid restart | Lift own incarnation to gossip-asserted value + 1 before refuting. Idempotent restart without on-disk state. |
| Observer callback exception | Caught and logged at the runner; cannot wedge SWIM. |
| MTU overflow | Asserted-and-dropped at transport. Logged at ERROR (indicates a bug, not a runtime condition). |

### 5.2 Failure modes requiring explicit test coverage

1. Bootstrap with all seeds unreachable → node starts in single-node view.
2. Suspect-self refutation race → exactly one refutation message, not duplicates.
3. Rapid kill/restart of same node → incarnation monotonic across restarts via self-suspicion floor.
4. Observer fires during state mutation → enforced post-return invocation in runner.

### 5.3 Logging policy

- `INFO`: every state transition with shard_id, incarnation, reason. Sparse, high-signal.
- `DEBUG`: every UDP send/recv with message type and target. Off by default.
- `WARNING/ERROR`: bootstrap failures, observer exceptions, MTU overflows, malformed messages.

No metrics emission in Phase 2. A metrics endpoint becomes a contract; defer until Phase 4 needs it.

---

## 6. Testing strategy

Two layers, mirroring Phase 1's `fast` / `slow` split.

### 6.1 Fast suite — pure state machine (`tests/membership/test_state.py`)

Drives `MembershipState` directly with synthesized messages and a virtual clock. No sockets, no threads, no `time.sleep`. Each test runs in <10ms; full suite in <1s. Deterministic, repeatable, parallelizable.

Target ~40 test cases across:
- **State transitions:** alive→suspect (post indirect probe), suspect→dead (deadline), suspect→alive (higher incarnation), dead→alive (rejoin), tiebreaker, self-refutation, self-suspicion floor.
- **Dissemination:** backlog priority semantics, `K_GOSSIP=3` ordering, GC of stale entries.
- **Tick semantics:** at-most-one-Ping per `T_PING`, zero pings when all peers dead, deadlines advance.
- **Bootstrap and join:** empty-view first-Join behavior, MembershipDelta install, dead-record returned to incarnation-0 joiner.
- **Edge cases (§5):** unknown-shard_id drop, concurrent conflict resolution.

Tests use a `FakeRandom` with seeded sequences and an explicit `clock` value advanced per step. Tests read like scripts.

### 6.2 Behavioral suite (`tests/membership/test_e2e.py`, marked `slow`)

Real 3-node localhost cluster, real UDP sockets, real threads, real-time. Each test starts/stops a fresh cluster.

| # | Test | Purpose |
|---|---|---|
| 1 | `test_three_nodes_converge_on_alive` | Convergence: all three report each other alive within <3s. |
| 2 | `test_kill_one_node_others_detect_dead` | Failure detection: dead within <7s of `process.terminate()`. |
| 3 | `test_killed_node_rejoins_returns_to_alive` | Rejoin: convergence to alive within <3s of restart. |
| 4 | `test_admission_control_rejects_when_peer_dead` | Admission: `Error{SHARD_UNAVAILABLE}` reply within <1s of `BeginRequest`. |
| 5 | `test_in_flight_request_fails_cleanly_on_kill` | The user-visible Q4 scenario; clean error within <7s. |
| 6 | `test_bootstrap_with_unreachable_seeds` | Single-node-view fallback within `<T_SUSPECT`. |
| 7 | `test_simultaneous_cluster_restart` | Convergence after triple-restart within <5s. |
| 8 | `test_existing_tier1_passes_with_membership_running` | Regression net: Phase 1 token-exact acceptance under membership. |
| 9 | `test_existing_tier2_passes_with_membership_running` | Regression net: Phase 1 hidden-state acceptance under membership. |
| 10 | `test_observer_exception_does_not_wedge_runner` | Functional: a raising observer does not wedge SWIM. |

Tests 8 and 9 are the most important behavioral cases — they prove the membership layer doesn't break Phase 1 correctness.

### 6.3 Out of scope for Phase 2 testing

- Convergence at scale (50+ nodes) — Phase 6 simulator.
- Cross-machine timing — when the 3090/DGX cluster integration begins.
- Adversarial / fuzzed inputs — Phase 6 with TLS/auth.

### 6.4 CI shape

```bash
uv run pytest tests/membership/test_state.py            # <1s, every PR
uv run pytest -m slow tests/membership/test_e2e.py      # ~30s, every PR
uv run pytest -m slow                                    # full slow suite, ~3 min, on main + pre-tag
uv run ruff check src tests scripts                      # unchanged
uv run mypy src tests scripts                            # unchanged, strict
uv run python -m grpc_tools.protoc -I proto --python_out=src/model_shard/_pb proto/wire.proto
```

---

## 7. Migration and rollback

### 7.1 Migration

Phase 2 is **strictly additive** at the file level: one new package (`membership/`), six new protobuf message types, ~30 new lines in `node.py`. No Phase 1 file is rewritten or restructured.

The `shards.yaml` config gains no new fields. UDP port for SWIM is derived deterministically (`tcp_port + 1000`) so existing configs work unchanged. (If derived ports collide with another service in a future deployment, an explicit `swim_port` field can be added — explicitly out of scope for Phase 2.)

### 7.2 Rollback

A `ENABLE_GOSSIP=false` env var (default `true`) bypasses `MembershipRunner` construction entirely. With it disabled, `node.py` behaves identically to Phase 1 — admission control always passes, no observer hooks fire. This is a safety valve for development, not a production toggle; it should be removed in Phase 3 once the membership layer has stabilized.

---

## 8. Open questions for implementation planning

These are deliberately deferred from design to plan, to be resolved during `writing-plans`:

1. **Protobuf vs. dataclass for in-memory `MemberRecord`.** Design says dataclass; plan should confirm the protobuf↔dataclass boundary in `messages.py` is clean.
2. **Threading: `threading.Thread` vs. `concurrent.futures.ThreadPoolExecutor` for the runner.** Design implies the former; plan picks.
3. **UDP socket lifecycle on test teardown.** macOS holds bound UDP ports briefly after close; tests need `SO_REUSEADDR` and possibly a port range.
4. **Order of operations during shutdown.** Drain the runner before closing the transport, or close the transport to unblock the runner's `recvfrom`?
5. **Whether to land protobuf changes in a separate prep commit before the rest of Phase 2.** Probably yes, to keep the diff readable.

---

## 9. Acceptance for Phase 2

Phase 2 is complete when:

- All 6 design decisions (§1.3) are reflected in shipped code.
- Fast suite (~40 cases) passes in <1s.
- Behavioral suite (~10 cases) passes in <30s, including tests 8 and 9 (Tier 1 + Tier 2 regression net under membership).
- `ruff check` and `mypy --strict` clean across new code.
- `ENABLE_GOSSIP=false` reproduces Phase 1 behavior exactly.
- A 3-node localhost cluster started via `scripts/run_node.py` shows correct `alive` convergence and correct dead-detection on `kill -9` of one node.
