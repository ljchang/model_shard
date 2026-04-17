# Phase 5b — Dynamic Expert Migration, Heat Tracking, and Decode-Loop Hang Fix

**Status:** draft, 2026-04-17
**Scope:** Option 1 from brainstorming (mechanism-first). Three pieces:
- **A. Heat tracking.** Every node counts how often *it routed* to each (layer, expert); sparse top-N gossips via SWIM piggyback.
- **B. Migration mechanism.** Target-pull replication: a node with high local routing demand for expert E (not currently hosted) requests E from a current owner over TCP, concatenates into its compact stack, gossips new ownership.
- **D. Decode-loop hang fix.** Observer-triggered queue poison so the head's `drive_decode_loop` unblocks immediately when any peer leaves ALIVE, instead of waiting indefinitely on `token_queue.get()`.

Policy (the decide-when-to-migrate part, "C") is a simple threshold stub in this phase; sophisticated heat-driven replicate/evict is deferred to Phase 6 or a 5c follow-up.

## 1. Background & Decisions

### 1.1 Why now
Phase 5a made "a node holds only a subset of experts" the default runtime shape (opt-in via `ENABLE_PARTIAL_LOAD`). Until 5a, every node held every expert's weights — migration was meaningless because the target already had them. 5a also locked in the key mutability result (`QuantizedSwitchLinear.num_experts` is a `@property`; `mx.take` onto the stacked tensor preserves quant semantics bit-exactly on the no-sort path). 5b is the symmetric construction: remove an expert's slice from one node's stack, ship it over TCP, and *grow* the target's compact stack by `mx.concatenate`.

The decode-hang (D) is bundled in because the fix touches the same membership-observer callback that 5b needs for safe cross-peer coordination.

### 1.2 Decisions

- **D1. Option 1 scope.** Mechanism-first. A + B + D. Heat-driven sophisticated policy is deferred.
- **D2. Heat signal.** *Local routing count* — each node increments `heat[(layer, expert)]` every time its own routing selects that expert (the `top_k_ids` produced by `run_attention_and_route`). Decays via EMA with the same shape as `LoadTracker` (alpha=0.3). Reported as the sparse top-N entries (N=16 default) so UDP MTU holds.
- **D3. Heat gossip transport.** Piggyback on SWIM Ping/Ack/PingReq/PingReqAck alongside the existing `LoadReport`. New wire field `repeated ExpertHeatReport heat` on each of those four messages. Payload per report: sender `shard_id`, repeated `{layer_idx, expert_id, heat_ema_x100}` tuples. Peer nodes accumulate these into `MembershipRunner.latest_heat()` (parallel to `latest_loads()`).
- **D4. Migration initiation.** *Target-pull*. A node inspects its own heat map and its own `_live_experts` registry; if a (layer, expert) is hot locally and not hosted, it issues `ExpertWeightRequest` to one current owner. No source-initiated push; no central orchestrator.
- **D5. Owner discovery.** Union of `ShardSpec.moe_experts` (bootstrap, frozen) and received `OwnershipDelta` gossip. `Node` exposes a method `owners_of(layer_idx, expert_id) -> set[str]` that callers use to pick a source. Tie-break: least-loaded owner via the same P2C logic Phase 4 already uses (`LoadReport`-derived peer loads).
- **D6. Wire protocol.** Two new `Envelope` payloads:
  - `ExpertWeightRequest { protocol_version, request_id, layer_idx, expert_id }` — target → source.
  - `ExpertWeightTransfer { protocol_version, request_id, layer_idx, expert_id, repeated TensorDescriptor tensors, uint32 tensor_count }` — source → target; out-of-band payload = 9 tensors concatenated in fixed order (see §3.2). `tensor_count=9` for Phase 5b.
  - On source error (expert no longer held, slice fails, etc.): existing `Error { ERR_SHARD_UNAVAILABLE }` on the same connection. Target aborts this migration and reconsiders on next scan interval.
- **D7. Monolithic per-expert transfer.** A single envelope per expert migration. ~3.3 MB payload per (layer, expert). Not chunked. Forward-compatible with chunking later because the protobuf already declares per-tensor descriptors.
- **D8. Attach semantics.** Receiver mutates its compact stack under `_MLX_COMPUTE_LOCK`:
  1. For each of the 9 tensors in (gate_proj, up_proj, down_proj) × (weight, scales, biases):
     `proj.<attr> = mx.concatenate([proj.<attr>, incoming[None, ...]], axis=0)`
  2. Append `expert_id` to `lm.held_ids_per_layer[layer_idx]` as the new tail element.
  3. `mx.eval(model.parameters())` to realize.
  4. Update `Node._live_experts[layer_idx].add(expert_id)`.
  5. Release the lock.
  6. Enqueue an `OwnershipDelta{ADD}` onto the membership runner's outbound gossip buffer.
- **D9. Ownership registry.** `Node._live_experts: dict[int, set[int]]`, initialized from `shard.moe_experts` (converting tuples to sets), mutated on successful attach. `ShardSpec` stays frozen — it's the bootstrap declaration, nothing more. `ExpertOrchestrator` receives `live_owners_provider: Callable[[int, int], set[str]]` instead of a static `owners` map; the callback returns the current live owner set for `(layer_idx, expert_id)` by union-ing bootstrap `moe_experts` across all shards with gossip-observed `OwnershipDelta`s. Called inside `run_split_layer` when computing the by-owner grouping, so any ownership delta received since the last routing call is picked up automatically.
- **D10. Ownership gossip.** New piggybacked field `repeated OwnershipDelta ownership` on Ping/Ack/PingReq/PingReqAck. Payload: `{shard_id, layer_idx, expert_id, action=ADD}` (ADD-only in 5b; idempotent; no version field needed because replication never requires distinguishing stale from fresh in a ADD-only model). `MembershipRunner` tracks a set `self._ownership_adds: set[tuple[shard_id, layer_idx, expert_id]]` that starts empty and monotonically grows. Node reads it to compute live owner union.
- **D11. Replication only; no eviction in 5b.** Source keeps expert E after transfer. Both owners serve E afterward; P2C picks between them. Eviction is a Phase 6 problem — it introduces stale-ownership races and requires the version field we've deliberately skipped.
- **D12. Source concurrency.** Source slices E out of its own compact stack via `mx.take(proj.<attr>, [local_slot_of(E)], axis=0)`. This is a pure read. `_MLX_COMPUTE_LOCK` is held only during `mx.take` + `mx.eval` + `tensor_to_bytes` (~tens of ms on localhost). The subsequent TCP send of the 3.3 MB blob happens *outside* the lock so concurrent `ExpertRequest` handlers don't stall.
- **D13. Policy stub (C).** Periodic jittered scan. Three env-var knobs:
  - `MIGRATION_SCAN_INTERVAL_SECONDS` (default `10.0`).
  - `MIGRATION_HEAT_THRESHOLD` (default `50` — minimum local heat-EMA count to consider pulling).
  - `MIGRATION_MAX_EXPERTS_PER_LAYER` (default `128` — memory guard; M5-sized cluster uses unlimited).
  Single in-flight migration per node (hard cap). Scan interval is jittered ±25% to avoid cross-node synchronized scans.
- **D14. Decode-loop hang fix (D).** `Node._on_membership_change` gains a branch: on any peer-left-ALIVE transition, iterate `self._head_states` under `_state_lock` and enqueue a sentinel `_POISON_TOKEN = -1` into each `token_queue`. `_drive_decode_loop` (node.py:315) checks `if token_id == _POISON_TOKEN:` immediately after `token_queue.get()`; on sentinel it raises `PeerLeftAliveError` (new in `node.py`, a `RuntimeError` subclass) which is caught by the same `ExpertRpcFailure`-pattern branch already present at lines 352-363 and translated to `Error{ERR_SHARD_UNAVAILABLE, is_final=true}` for the client.
- **D15. Correctness bar.** Bit-exact on the no-sort path (same constraint as 5a). After A→B migration of expert E at layer L:
  - `mx.array_equal(run_selected_experts(lm_A, h, L, [E]), run_selected_experts(lm_B, h, L, [E]))` for input `h` with `B*Seq ≤ 7`.
  - Cross-check: both equal `run_selected_experts(lm_full, h, L, [E])`.
- **D16. Migration / rollback gate.** New env var `ENABLE_DYNAMIC_MIGRATION=false` (default). When false, the periodic scan thread is not started, no migration RPCs are sent, no ownership deltas are gossiped. Heat counters still increment in-process (cheap and useful for observability) but no report is piggybacked on SWIM messages. When true, requires `ENABLE_PARTIAL_LOAD=true` (5b only makes sense with 5a active); `Node.__init__` raises on contradictory settings.
- **D17. Non-goals.** Eviction; migration of chassis weights; migration of full layers; runtime *shrinking* of the compact stack; migration across heterogeneous quant schemes; transfer cancellation / resumption; persisted migration history; cluster-wide heat aggregation (each node sees only neighbours it has heard from).

## 2. Components

### 2.1 `src/model_shard/heat.py` (new)

```python
class HeatTracker:
    def __init__(self, alpha: float = 0.3, top_n: int = 16) -> None: ...
    def observe(self, layer_idx: int, expert_ids: list[int]) -> None: ...
    def report(self) -> list[tuple[int, int, int]]:
        """Return sparse [(layer_idx, expert_id, ema_x100), ...] sorted by
        EMA descending, capped at top_n. Suitable for UDP piggyback."""
    def local_heat(self, layer_idx: int, expert_id: int) -> int:
        """Return the current EMA×100 for one (layer, expert). Used by the
        migration policy scan to decide whether to pull."""
```

Thread-safe (lock-guarded, same shape as `LoadTracker`). Mutated from the compute path (`moe.run_attention_and_route` callers) and read from the gossip thread (`report()`) and migration scan thread (`local_heat()`).

### 2.2 `src/model_shard/migration.py` (new)

```python
@dataclass
class MigrationPolicy:
    scan_interval_s: float
    heat_threshold: int
    max_experts_per_layer: int

class MigrationScanner:
    """Periodic scan thread that initiates target-pull migrations.

    Dependencies (injected):
      - heat_tracker: HeatTracker            (read local_heat)
      - live_experts: dict[int, set[int]]    (what we already host)
      - owner_lookup: Callable[[int, int], set[str]]  (who else hosts it)
      - load_provider: Callable[[], dict[str, int]]   (peer loads for P2C tie-break)
      - peer_rpc: ExpertWeightPeerRPC       (TCP client, §2.3)
      - attacher: Callable[[int, int, list[mx.array]], None]
                                             (receiver-side concat, §2.4)
      - ownership_announcer: Callable[[int, int], None]
                                             (enqueue OwnershipDelta gossip, §2.5)
    """

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def _scan_once(self) -> None: ...     # testable synchronous entrypoint
```

One `MigrationScanner` per `Node`. Runs a daemon thread that sleeps `scan_interval_s ± 25% jitter`, then calls `_scan_once`. Single in-flight cap enforced by a `threading.Lock` around the RPC call. Tests exercise `_scan_once` directly.

### 2.3 `src/model_shard/migration.py` — `ExpertWeightPeerRPC`

```python
class ExpertWeightPeerRPC:
    """TCP client for ExpertWeightRequest → ExpertWeightTransfer."""
    def pull(self, source_shard_id: str, layer_idx: int, expert_id: int
             ) -> list[mx.array]:
        """Open a TCP connection to source, send ExpertWeightRequest,
        block on ExpertWeightTransfer, return the 9 tensors in the fixed
        order (gate.w, gate.s, gate.b, up.w, up.s, up.b, down.w, down.s,
        down.b). Raises on error envelope or timeout."""
```

Mirrors the shape of `TcpPeerRPC` (Phase 3) but carries the 9-tensor payload.

### 2.4 `src/model_shard/partial_load.py` — `attach_expert`

```python
def attach_expert(
    lm: LoadedModel,
    layer_idx: int,
    expert_id: int,
    tensors: list[mx.array],   # 9 tensors in fixed order
    mlx_lock: threading.Lock,
) -> None:
    """Grow the compact stack at layer_idx by one expert.

    Invariant: expert_id must NOT already be in lm.held_ids_per_layer[layer_idx].
    Caller enforces; attach_expert raises ValueError on violation.

    Steps under mlx_lock:
      1. For each (proj_name, attr) in the fixed 9-tensor order, set
         proj.<attr> = mx.concatenate([proj.<attr>, tensor[None, ...]], axis=0)
      2. lm.held_ids_per_layer[layer_idx] = (*old, expert_id)
      3. mx.eval(model.parameters())
    """
```

Note: `lm.held_ids_per_layer` becomes `dict[int, tuple[int, ...]]` but the value is *replaced* (new tuple) rather than mutated, so thread-safe dict reads are safe.

### 2.5 `src/model_shard/partial_load.py` — `slice_expert`

```python
def slice_expert(
    lm: LoadedModel,
    layer_idx: int,
    expert_id: int,
    mlx_lock: threading.Lock,
) -> list[mx.array]:
    """Return the 9 tensors for one expert, in fixed order. Does not
    modify lm. Holds mlx_lock only during mx.take + mx.eval."""
```

Used by the source-side handler (§2.6) and directly by the bit-exact test (§5.2).

### 2.6 `src/model_shard/node.py` — new handlers and state

New handler `_handle_expert_weight_request`:
```
def _handle_expert_weight_request(
    self, req: wire_pb2.ExpertWeightRequest,
    inbound_stream: BinaryIO,
) -> None:
    """Slice the requested expert out of our compact stack and reply
    with ExpertWeightTransfer. Error{SHARD_UNAVAILABLE} on miss."""
```

New sentinel and error class:
```python
_POISON_TOKEN: int = -1
class PeerLeftAliveError(RuntimeError): ...
```

`_drive_decode_loop` gains sentinel check (D14). `_on_membership_change` gains the poison branch (D14).

`Node.__init__` wires up:
- `self._heat_tracker = HeatTracker(...)`
- `self._live_experts: dict[int, set[int]] = {L: set(ids) for L, ids in shard.moe_experts.items()}`
- `self._scanner = MigrationScanner(...)` if `_dynamic_migration_enabled()`.
- `self._ownership_seen: set[tuple[str, int, int]]` — gossip-observed ownership (seeded at init with every bootstrap `(sid, L, e)` so we have a complete initial view).

`Node.serve_forever` calls `self._scanner.start()` after `self._membership.start()`; `shutdown` calls `self._scanner.stop()`.

Dispatcher extended in `_dispatch` to handle the new wire types: `expert_weight_request` and `expert_weight_transfer`.

The orchestrator's `owners` static mapping is replaced by a live-views method `self._orchestrator.owners_of(layer_idx, expert_id)` that reads `shard.moe_experts ∪ self._ownership_seen`.

### 2.7 `src/model_shard/expert_orchestrator.py` changes

- `ExpertOrchestrator.__init__` gains `live_owners_provider: Callable[[int, int], set[str]]` (replaces the static `owners: Mapping[str, set[int]]` argument, which becomes a bootstrap seed).
- `group_expert_ids_by_owner_loaded` signature grows a `live_owners_provider` parameter so routing picks up new replicas automatically.
- Called once per `run_split_layer` invocation. O(k) lookups per call (small k; hot path tolerance OK).

### 2.8 `src/model_shard/membership/*` changes

- `LoadReportRecord` already exists. Add `HeatReportRecord` and `OwnershipDeltaRecord` dataclasses in `records.py`.
- `MembershipRunner` gains:
  - `start_heat_source(callable)` — mirror of `start_load_source`. Called from `Node.__init__`. Reads `HeatTracker.report()`, includes in outbound Ping/Ack.
  - `announce_ownership_add(shard_id, layer_idx, expert_id)` — enqueues an `OwnershipDelta{ADD}` to piggyback on the next K gossip rounds (TTL-limited; default 5 rounds).
  - `latest_heat() -> dict[str, list[HeatReportRecord]]` — per-peer most recent.
  - `ownership_view() -> set[tuple[str, int, int]]` — monotonic union of all ADD deltas ever received.
- Per-message size budget: with `top_n=16` heat entries per sender and ~4 ownership deltas in flight at a time, stays well under the 1400-byte UDP MTU budget when combined with existing `LoadReport`s.

## 3. Wire Protocol

### 3.1 New `wire.proto` messages

```proto
message ExpertHeatReport {
  string shard_id = 1;
  repeated ExpertHeatEntry entries = 2;
  int64  ts_unix_ms = 3;
}
message ExpertHeatEntry {
  uint32 layer_idx = 1;
  uint32 expert_id = 2;
  uint32 heat_ema_x100 = 3;  // EMA × 100 (same convention as LoadReport)
}

message OwnershipDelta {
  string shard_id  = 1;
  uint32 layer_idx = 2;
  uint32 expert_id = 3;
  uint32 action    = 4;  // 0 = ADD; 1 = REMOVE (reserved for Phase 6)
  int64  ts_unix_ms = 5;
}

message ExpertWeightRequest {
  uint32 protocol_version = 1;
  string request_id = 2;   // correlation id; not the inference request_id
  uint32 layer_idx  = 3;
  uint32 expert_id  = 4;
}

message ExpertWeightTransfer {
  uint32 protocol_version = 1;
  string request_id = 2;
  uint32 layer_idx  = 3;
  uint32 expert_id  = 4;
  // 9 tensors in fixed order: gate.w, gate.s, gate.b,
  //                           up.w,  up.s,  up.b,
  //                           down.w, down.s, down.b.
  // Payload is their concatenation; each descriptor's byte_count lets
  // the receiver split.
  repeated TensorDescriptor tensors = 5;
  uint32 tensor_count = 6;  // must equal 9 for Phase 5b
}
```

### 3.2 Piggyback fields on existing SWIM messages

Concrete tag allocations (next unused in each existing message, read from `proto/wire.proto` at spec-time):

```proto
// Ping and Ack: last used tag = 5 (loads)
message Ping { ...existing fields 1-5...
  repeated ExpertHeatReport heat      = 6;
  repeated OwnershipDelta   ownership = 7;
}
message Ack { ...existing fields 1-5...
  repeated ExpertHeatReport heat      = 6;
  repeated OwnershipDelta   ownership = 7;
}
// PingReq: last used tag = 6 (loads)
message PingReq { ...existing fields 1-6...
  repeated ExpertHeatReport heat      = 7;
  repeated OwnershipDelta   ownership = 8;
}
// PingReqAck: last used tag = 7 (loads)
message PingReqAck { ...existing fields 1-7...
  repeated ExpertHeatReport heat      = 8;
  repeated OwnershipDelta   ownership = 9;
}
```

### 3.3 Additions to `Envelope.oneof payload`

Last used tag in `Envelope.oneof payload` = 15 (`expert_response`). New:

```proto
ExpertWeightRequest  expert_weight_request  = 16;
ExpertWeightTransfer expert_weight_transfer = 17;
```

### 3.4 Out-of-band payload format for `ExpertWeightTransfer`

The TCP frame is `[msg_len:4][msg][tensor_len:4][tensor_bytes]` as already documented in `transport.py`. `tensor_bytes` is the concatenation of all 9 per-tensor byte blobs in the declared order. Each tensor's starting offset is derivable from the running sum of earlier descriptors' `byte_count` fields. No padding, no alignment — tensor dtypes preserve their raw stored representation (u32 for quant weights, bf16 for scales/biases).

## 4. Memory Model

Migration grows the receiver's resident memory. On Phase 5a with the default YAML (~44 experts per shard at layer 15), adding one more expert per layer raises the per-shard expert footprint by ~3.3 MB per (layer, expert). M5 has ample headroom. On 24 GB targets (3090), the `MIGRATION_MAX_EXPERTS_PER_LAYER` guard bounds growth.

Source memory is unchanged (replication, not move).

## 5. Testing Strategy

### 5.1 Fast tests (no model load)

- `test_heat_tracker.py` — `observe()` increments EMA correctly; `report()` returns sorted top-N; thread-safe concurrent observes.
- `test_migration_scanner_policy.py` — `_scan_once` picks the hottest (layer, expert) not-in-`_live_experts` over threshold; respects single-in-flight cap; respects `max_experts_per_layer`.
- `test_wire_expert_weight_roundtrip.py` — protobuf encode/decode of new messages; `ExpertWeightTransfer` with synthetic 9-tensor payload round-trips bytes equally.
- `test_membership_ownership_gossip.py` — `announce_ownership_add` includes the delta in outbound Ping; receiver merges into `ownership_view`; idempotent on repeated delivery.
- `test_decode_hang_fix.py` — construct a `Node` fixture; inject a peer-left-ALIVE transition; assert `token_queue` gets the sentinel; assert `_drive_decode_loop` exits via `PeerLeftAliveError` and emits `Error{ERR_SHARD_UNAVAILABLE, is_final=true}`.

### 5.2 Slow tests (model load required)

- `test_migration_bit_exact_per_expert.py` — the load-bearing correctness proof:
  1. Load `lm_full` (full 128-stack).
  2. Load `lm_A` via `load_model_partial(held={15: [0, 3, 6, 9]})`.
  3. Load `lm_B` via `load_model_partial(held={15: [1, 4, 7, 10]})`.
  4. Execute `slice_expert(lm_A, 15, 3, lock)` → 9 tensors.
  5. Execute `attach_expert(lm_B, 15, 3, tensors, lock)`.
  6. For a synthetic input `h` with `B*Seq=7` (no-sort path), assert:
     `mx.array_equal(run_selected_experts(lm_A, h, 15, [3]), run_selected_experts(lm_B, h, 15, [3]))`
  7. And: both equal `run_selected_experts(lm_full, h, 15, [3])`.
- `test_migration_over_tcp.py` — spin up two `Node` fixtures; trigger a manual pull via `MigrationScanner._scan_once` on B with a forced heat vector; verify expert is transferred and attached; rerun bit-exact comparison on the receiver after attach.
- `test_ownership_gossip_convergence.py` — 3-node in-process cluster; node B attaches expert E; after ≥2 gossip rounds, assert nodes A and C see `(B, L, E)` in `ownership_view()`; assert subsequent P2C routing from A or C can pick B as a candidate.
- `test_decode_hang_fix_e2e.py` — 3-node in-process Tier 1 fixture; kill mid-decode peer; assert head closes decode loop within `SUSPECT_PERIOD + 1s` with `ERR_SHARD_UNAVAILABLE` (previously blocked indefinitely).

### 5.3 Regression

- All Phase 1-4 slow tests must still pass with `ENABLE_DYNAMIC_MIGRATION=false` (default).
- With `ENABLE_DYNAMIC_MIGRATION=true` and `ENABLE_PARTIAL_LOAD=true`, Tier 1 E2E (5 prompts, ≤8 tokens each for no-sort path) still produces tokens bit-exact to the Phase 1 reference — with the migration scanner running in the background and making no pulls (no prompt is long enough to generate migration-worthy heat).
- `ENABLE_DYNAMIC_MIGRATION=true` + `ENABLE_PARTIAL_LOAD=false` → `Node.__init__` raises `ValueError` with a message explaining the dependency.

## 6. Acceptance

1. `ruff check`, `mypy` clean.
2. Fast suite green (including all new Phase 5b fast tests).
3. All Phase 5b slow tests pass.
4. `ENABLE_DYNAMIC_MIGRATION=false` (default) → Phase 3/4/5a slow suite unchanged.
5. With both flags on, 3-node in-process Tier 1 E2E still produces reference tokens.
6. Decode-hang fix (D) verified by killing a peer mid-decode; head exits decode loop cleanly within ~`SUSPECT_PERIOD + 1s`.
7. README updated with a Phase 5b status paragraph; spec cross-linked from `project_gossip_moe.md` memory file.

## 7. Open Risks and Mitigations

- **R1 — Concurrent attach on the same receiver.** Target pulls E₁ from A, pulls E₂ from C concurrently. Both attaches race on `_MLX_COMPUTE_LOCK`. Mitigation: the single-in-flight-cap in `MigrationScanner` limits this to one pull at a time per node; attach is serialized by construction. A future multi-in-flight scanner needs explicit queuing.
- **R2 — Ownership convergence lag.** Node B attaches E at t=0; node C's orchestrator doesn't see B in owner set until gossip round ~t=1-2s. During the window, C keeps routing E exclusively to A — correct, just suboptimal. Acceptable.
- **R3 — Transfer failure mid-stream.** TCP hangup during the 3.3 MB transfer. Mitigation: target catches `socket.error` / `EOFError` in `ExpertWeightPeerRPC.pull`; aborts the migration; does not update `_live_experts`; next scan interval retries (possibly picking a different source via P2C). No partial attach possible — attach is single-envelope-atomic.
- **R4 — Sort-path FP noise (inherited from 5a §7.5).** Bit-exact holds only on no-sort path (`indices.size < 64`). Tests constrain `B*Seq ≤ 7`. Tier 1 E2E uses short prompts (≤8 tokens) so decode steps stay on no-sort. Documented, not fixed in 5b.
- **R5 — Heat signal gameability / stampede.** If multiple targets all have high local heat for the same E at the same scan tick, they may all pull from the same A simultaneously. Stampede is bounded: only nodes *actually routing* to E have non-zero heat; single-in-flight cap limits concurrent pulls per node; P2C tie-break spreads source selection. Scan jitter (±25%) reduces synchronized ticks. Accepted risk in 5b.
- **R6 — Gossip flooding on ownership adds.** With 128 experts × 30 layers and per-node in-flight TTL of 5 gossip rounds, worst-case ~4 deltas per message is bounded. Size budget holds. Revisit only if per-message budget becomes tight in later phases.

## 8. References

- Phase 5a spec: `docs/superpowers/specs/2026-04-17-phase5a-partial-expert-loading-design.md`
- Phase 4 spec: `docs/superpowers/specs/2026-04-16-phase4-load-aware-routing-design.md`
- Phase 3 spec: `docs/superpowers/specs/2026-04-16-phase3-expert-sharding-design.md`
- Phase 2 spec: `docs/superpowers/specs/2026-04-16-phase2-gossip-discovery-design.md`
- Gossip MoE full spec: `/Users/lukechang/Downloads/gossip-moe-inference-spec.md` §10 (Dynamic Migration) §5.4 (Hot-plane)
- Phase 3 known hang: `node.py:316` decode-loop `queue.get()` under peer-death (D14 fix)
