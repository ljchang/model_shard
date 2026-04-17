# Phase 4 — Load-Aware Routing (with static per-expert replication)

**Status:** draft, 2026-04-16
**Scope:** Give nodes an observable signal about each other's load and let the `ExpertOrchestrator` prefer less-loaded owners when an expert has multiple candidates. Uses static YAML replication to create multi-candidate scenarios; Phase 5 will add dynamic migration. Single-layer expert split (layer 15) from Phase 3 remains the testbed.

## 1. Background & Decisions

### 1.1 What's new vs Phase 3
Phase 3 pinned each expert to exactly one owner via `config/shards.yaml` `moe_experts`. The orchestrator's `group_expert_ids_by_owner` built a `dict[id, owner]` where each key mapped to one string. Phase 4 lifts that:
- `moe_experts` entries may overlap across shards (parser already allows this — Task 3's validator didn't enforce cross-shard disjointness).
- An expert with 2+ candidate owners is dispatched to the less-loaded one.
- Nodes gossip a compact load report (EMA of queue depth) piggybacked on existing SWIM Ping/Ack frames.

### 1.2 Decisions
- **D1 Success criterion.** Routing correctness: given known loads `(L_A, L_B)`, the orchestrator sends to whichever candidate has the lower EMA. Verified by deterministic tests. No throughput claim.
- **D2 Replication source.** Static YAML overlap in `moe_experts`. Weights are already resident on every node (Phase 1 shortcut — each node loads the full 14 GB model), so multi-owner requires no migration. Phase 5 later moves to real per-expert weight loading and gossip-driven replication.
- **D3 Routing policy.** Power-of-two-choices with a twist: when only 2 candidates exist (the Phase 4 common case), always compare both (P2C degenerates to pick-min). When ≥3 candidates (possible in future), sample two uniformly, pick the less-loaded.
- **D4 Anti-oscillation.** EMA smoothing with α=0.3 on queue depth. ±10% uniform jitter applied at report time (not at routing time, so the receiver sees a slightly noisy value that's already settled). Routing decisions use the most recent gossiped EMA; no sticky routing in Phase 4 (experts have no KV cache; sticky matters only when attention layers become multi-owner, which is Phase 5+).
- **D5 Hot-plane transport.** Piggyback on the existing SWIM UDP sidecar. Each Ping/Ack/PingReq/PingReqAck gains a `repeated LoadReport loads` field. No new transport. Hot signals inherit cold plane's O(log N) convergence.
- **D6 Signals.** **Per-shard queue depth EMA** (scalar per shard) is the only Phase 4 signal. Expert heat histograms, per-layer latency EMAs, and batch-formation notices are explicitly Phase 5 inputs to migration — not needed here.
- **D7 Non-goals.** Batching, dynamic replication/migration, KV-cache affinity for attention, expert heat maps, latency histograms, any real-weight movement.

## 2. Topology (unchanged from Phase 3 except for YAML overlap)

```
head(layers 0-10; layer_15 experts {0,3,…,126, +0, 2})   ─┐
                                                           ├─ fan-in to mid
mid (layers 10-20; layer_15 experts {1,4,…,127, +0, 1};   ─┤   (orchestrator, loads-aware)
     attention+router+aggregator for L15)                  │
                                                           │
tail(layers 20-30; layer_15 experts {2,5,…,125, +1, 2})   ─┘
```

Three "hot" experts are deliberately replicated:
- expert 0: head + mid
- expert 1: mid + tail
- expert 2: tail + head

Every other expert keeps its single Phase 3 owner. When the router's top-8 picks any of {0, 1, 2}, the orchestrator has a choice; it applies power-of-two-choices against the peers' reported loads.

## 3. Components

### 3.1 `src/model_shard/load.py` (new)

Pure load-tracking helper:

```python
class LoadTracker:
    """EMA of queue depth, with jitter at report time."""
    def __init__(self, alpha: float = 0.3, jitter_pct: float = 0.1,
                 rng: random.Random | None = None) -> None: ...
    def observe(self, depth: int) -> None:
        """Called whenever queue depth changes (enter/exit of handler)."""
    def report(self) -> int:
        """Returns jittered EMA × 100 (integer encoding for wire compactness)."""
```

`rng` is parameterized for test determinism. Thread-safe via an internal lock (observe may come from multiple handler threads; report from the gossip thread).

### 3.2 `src/model_shard/membership/runner.py` — piggyback load

- `MembershipRunner.start_load_source(fn: Callable[[], LoadReport])` — register a callable that returns this node's own current load (called by the ping-emit loop).
- `MembershipRunner.latest_loads() -> Mapping[str, LoadReport]` — returns last-seen peer loads, updated on inbound Ping/Ack/PingReq/PingReqAck.
- Outgoing message builder appends `self_load = load_source()` to the `loads` field.
- Inbound handler extracts `loads[*]` and updates the per-peer load cache, keyed by `shard_id`.

Stale entries (>5s old) are treated as "unknown"; orchestrator treats unknown peers as having maximum load (so it prefers peers with fresh data).

### 3.3 `src/model_shard/moe.py` — multi-owner grouping

Replace `group_expert_ids_by_owner`:

```python
def group_expert_ids_by_owner_loaded(
    top_k_ids: list[int],
    owners: Mapping[str, set[int]],                   # multi-owner aware: id may appear in multiple sets
    peer_loads: Mapping[str, int],                    # gossiped EMA × 100
    self_shard_id: str,
    self_load: int,                                   # local load (not in peer_loads)
    rng: random.Random,
) -> dict[str, list[int]]:
    """Partition top_k_ids by owner, using power-of-two-choices when an
    expert has multiple candidates. Deterministic given rng state."""
```

The old `group_expert_ids_by_owner` (single-owner) remains as a thin wrapper calling the new function with synthetic uniform loads, so existing Phase 3 tests keep passing.

### 3.4 `src/model_shard/expert_orchestrator.py`

- `ExpertOrchestrator` gains two optional fields: `loads_provider: Callable[[], Mapping[str, int]] = lambda: {}` and `rng: random.Random | None = None`.
- `run_split_layer` threads these through to `group_expert_ids_by_owner_loaded`.
- Backward compatible: omitting both reduces to Phase 3 behavior.

### 3.5 `src/model_shard/node.py`

- Construct a `LoadTracker`; wire `observe()` at the entry and exit of `_handle_expert_request`.
- Pass `tracker.report` to `MembershipRunner.start_load_source`.
- Pass `runner.latest_loads` (closure) as the `loads_provider` to `ExpertOrchestrator`.
- Expose `/loads` on the HTTP debug endpoint (`tcp_port + 2000`) returning the node's view of all peer loads — useful for tests and manual smoke.

## 4. Wire Protocol

Add to `proto/wire.proto`:

```proto
message LoadReport {
  string shard_id       = 1;
  uint32 queue_depth_ema = 2;  // EMA × 100, so 250 means 2.5 average
  int64  ts_unix_ms     = 3;
}
```

Extend each existing SWIM message. Tag numbers differ per message (add one above the current max):

```proto
message Ping {         repeated LoadReport loads = 5; }    // deltas is 4
message Ack {          repeated LoadReport loads = 5; }    // deltas is 4
message PingReq {      repeated LoadReport loads = 6; }    // deltas is 5
message PingReqAck {   repeated LoadReport loads = 7; }    // deltas is 6
```

Wire backward-compat: Phase 3 peers ignore the new field; Phase 4 peers see empty loads from Phase 3 peers and treat them as unknown — specifically, `queue_depth_ema == UINT32_MAX` sentinel, which always loses the power-of-two comparison, so the orchestrator deprioritizes peers without fresh data.

Regenerate `wire_pb2.py` via `grpc_tools.protoc`.

## 5. Data Flow

### 5.1 Load observation (per node, continuously)

1. `_handle_expert_request` increments the tracker on entry.
2. `tracker.observe(current_depth)` updates the EMA.
3. On exit (success or error), decrement.
4. Gossip ping loop (every 1s per SwimConfig) calls `tracker.report()` and piggybacks on the outgoing Ping.

### 5.2 Routing decision (per token, per expert-split layer)

1. `run_attention_and_route` → `(post_attn_h, top_k_ids, top_k_weights)`.
2. `group_expert_ids_by_owner_loaded(...)`:
   - For each top-k id: if only one candidate, use it.
   - Else: power-of-two-choices. Sample two from candidates with `rng`; pick the one with lower `peer_loads` (or `self_load` if self is a candidate).
3. Orchestrator fans out as today.

## 6. Testing Strategy

### 6.1 Fast tests (no model)
- `LoadTracker.observe / report` EMA correctness across known sequences.
- Jitter range check (seeded rng → deterministic).
- `group_expert_ids_by_owner_loaded` on fixture loads: 2-candidate case always picks lower; 3-candidate case samples two and picks lower-of-two.
- LoadReport envelope roundtrip (extends the Phase 2 membership message roundtrip tests).

### 6.2 Slow tests
- **Multi-owner orchestrator**: mocked `loads_provider` returns {peer_a: 100, peer_b: 10}; orchestrator sends expert 0's work to peer_b across N iterations (deterministic since rng is seeded).
- **Regression**: Tier 1 bit-exact + Tier 2 tolerance still pass with Phase 4 flags, because the routing CHOICE doesn't affect math — any valid owner computes the same expert output.
- **E2E load-shift**: spawn 3 subprocess nodes with overlap config. Artificially inject sleep into one node's `_handle_expert_request`. Run sustained inference. After convergence, query the `/loads` debug endpoint and assert the router is skewing traffic to the fast nodes (via a counter exposed alongside loads).

### 6.3 Regression
- Phase 1 slow tests pass with `ENABLE_GOSSIP=true ENABLE_EXPERT_SHARD=true` (default already).
- Phase 2 membership tests pass with the LoadReport field on outgoing pings.
- Phase 3 split-equivalence still passes unchanged (no new math).

## 7. Migration / Rollback

No new env var for Phase 4 — the behavior auto-activates when:
- `moe_experts` has multi-owner entries (operator opts in by editing YAML).
- `ENABLE_EXPERT_SHARD=true` (from Phase 3).
- `ENABLE_GOSSIP=true` (default; provides the peer-loads source).

With `ENABLE_GOSSIP=false`, `loads_provider()` returns `{}` and the orchestrator degrades to "pick the first candidate" (deterministic; equivalent to Phase 3 when the YAML has one owner per expert).

## 8. Acceptance Criteria

1. `uv run pytest` — all fast tests pass.
2. `uv run pytest -m slow` — all slow tests pass (in isolation, per Phase 3's known Metal state issue for the full-suite run).
3. `ENABLE_GOSSIP=true ENABLE_EXPERT_SHARD=true uv run pytest -m slow` passes.
4. Deterministic routing-correctness test asserts the router picks the less-loaded candidate in a controlled 2-candidate scenario.
5. `ruff check`, `mypy` clean.
6. README updated with Phase 4 status.

## 9. References

- Phase 3 spec: `docs/superpowers/specs/2026-04-16-phase3-expert-sharding-design.md`
- Gossip MoE full spec: `docs/gossip-moe-inference-spec.md` §5 (gossip), §7 (load-aware routing)
- SWIM runner: `src/model_shard/membership/runner.py`
- `moe_experts` YAML extension: Task 3 of Phase 3 plan (commit `e9c1531`)
