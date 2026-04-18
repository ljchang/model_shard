# Phase 6-C — Expert Eviction + `OwnershipDelta{REMOVE}`

**Status:** draft, 2026-04-18
**Scope:** Under capacity pressure, a node evicts a cold migration-added expert from its compact stack, gossips `OwnershipDelta{action=1}`, and converges cluster-wide via last-writer-wins on `ts_unix_ms`. Third and final sub-project of Phase 6.

## 1. Background & Decisions

### 1.1 Why now
Phase 5b gave every node a mutable `_live_experts` registry and a target-pull migration scanner — but replication-only. `_live_experts` grows forever; capacity is bounded only by the soft `MIGRATION_MAX_EXPERTS_PER_LAYER` knob which, when hit, just stops pulls. Phase 6-A made retries robust; Phase 6-B made chains verifiable. What's still missing is the ability to *shrink* the stack. Without it, deployments on memory-constrained hardware (24 GB 3090, DGX Spark nodes) saturate after any prolonged migration activity.

Phase 6-C closes the Phase 6 trilogy (6-A retry + 6-B provenance + 6-C eviction) and unlocks the 3090/Spark cluster deployment by completing the memory story that 5a opened.

### 1.2 Decisions

- **D1. Scope.** Local eviction with gossip-based ADD/REMOVE convergence. Out: quorum-based last-replica protection, two-phase tentative eviction, memory-pressure probing, eviction of bootstrap-held experts.

- **D2. Eviction trigger & victim selection.** Capacity-driven with coldest-heat victim:
  - **Trigger:** eviction runs on every `MigrationScanner` tick (after the pull pass), and fires only when some layer has `len(_live_experts[L]) >= MIGRATION_MAX_EXPERTS_PER_LAYER`. Below capacity, the eviction pass is a no-op.
  - **Victim:** lowest `heat_tracker.local_heat(L, E)` among `_live_experts[L] - bootstrap_held`, also not within `MIGRATION_EVICT_COOLDOWN_SECONDS` of its attach time.
  - **Never-evict set:** experts in `self._shard.moe_experts[L]` (YAML-declared, bootstrap role).

- **D3. Convergence data structure.** Promote `_ownership_seen: set[tuple[str, int, int]]` (5b, ADD-only) to `_ownership_view: dict[tuple[str, int, int], tuple[int, int]]` where the value is `(action, ts_unix_ms)`. On receive: if incoming `ts > stored_ts`, overwrite; else drop. Backward-compatible API: `ownership_view() -> set[tuple[str, int, int]]` returns the subset whose action is ADD. Preserves every existing caller's contract.

- **D4. Wire protocol unchanged.** `OwnershipDelta` already has `action: uint32` (0=ADD, 1=REMOVE) and `ts_unix_ms: int64` — introduced in 5b Task 1, reserved for 6-C. The `_OutboundOwnership` TTL queue from 5b Task 9 handles REMOVE symmetrically without change.

- **D5. Safety invariants.**
  1. **Last-replica guard.** Before evicting, `self.owners_of(L, E) - {self._shard.shard_id}` must be non-empty; else raise `LastReplicaError`, no state mutation.
  2. **Bootstrap-held guard.** Experts in YAML `moe_experts` are never evicted.
  3. **Cooldown guard.** Newly-attached experts (via 5b migration) are ineligible for eviction for `MIGRATION_EVICT_COOLDOWN_SECONDS` (default 30s) after attach, preventing attach/evict oscillation under tight capacity.
  4. **Compute lock serialization.** `detach_expert` holds `_MLX_COMPUTE_LOCK` across the stack mutation. Queued `ExpertRequest` handlers that grab the lock after eviction see the post-eviction compact stack and correctly return `ERR_WRONG_SHARD` via `_handle_expert_request`'s `_live_experts` check (see D7).

- **D6. Accepted race.** Two nodes holding the last-two replicas can simultaneously evict if each, at local-check time, sees the other as "the other owner." Window is narrow: local lock serializes own-evictions; a single in-flight eviction per node; gossip converges within one round. Phase 6-C documents-and-accepts; quorum is Phase 7 polish if deployment shows the race firing.

- **D7. `_handle_expert_request` authority.** Today the handler validates against `self._shard.moe_experts` (bootstrap). Change it to `self._live_experts` so the post-eviction state is authoritative. This also fixes a latent 5b bug: a node that migrated-in an expert not in bootstrap YAML would currently reject inbound `ExpertRequest` for that expert — verify by reading 5b integration code.

- **D8. Scanner integration (5b).** `MigrationScanner._scan_once` gains an eviction pass **after** the existing pull pass. Same single-in-flight lock — pull and evict don't interleave in one tick, preventing oscillation. New method `_maybe_evict_one()` scans `_live_experts` for capacity-saturated layers, picks a victim per D2, and calls `Node.migration_detach(L, E)`.

- **D9. Provenance interaction (6-B).** During convergence, a chain may carry an OP_EXPERT from a node that has since evicted. Downstream validators reject per 6-B's current-ownership semantics. The existing Task 10 tail→head→client error path handles this; client receives `ERR_INVALID_PROVENANCE` and retries. Documented transient; no 6-C change to 6-B needed.

- **D10. Retry interaction (6-A).** Eviction-during-fan-out: node A sends `ExpertRequest` to B, B has just evicted E. B returns `ERR_WRONG_SHARD`. `_phase_b_with_retry` catches `ExpertRpcFailure`, excludes B from the excluded-peers set, re-routes via `live_owners_provider` to another replica. No 6-C change to 6-A needed.

- **D11. Gate.** `ENABLE_EVICTION=true` default-on. Eviction only fires under capacity pressure, so default-on is safe and matches the "5b's replication now has memory bounds" narrative. Set `ENABLE_EVICTION=false` to reproduce pre-6-C replicate-only behavior. `MIGRATION_EVICT_COOLDOWN_SECONDS=30` is the other new knob.

- **D12. Correctness bar.**
  1. `attach_expert` → `detach_expert` round-trip on a synthetic stack leaves byte-identical tensors to the original (proves complementary-index mutation is the inverse of concatenation).
  2. Interleaved `OwnershipDelta{ADD, ts=t1}` and `OwnershipDelta{REMOVE, ts=t2}` arriving in any order produce the same final `ownership_view` at every node.
  3. Evicting the sole live owner raises `LastReplicaError` and does not mutate state.
  4. Slow E2E: 3-node cluster, force a migration pull followed by capacity-pressure-triggered eviction; chain stays valid, tokens continue, `_live_experts` shrinks back to the target capacity.
  5. Slow race: scanner tries to evict an expert while an inbound `ExpertRequest` for that expert is mid-compute under the lock; the in-flight compute completes correctly; the next inbound request for the same expert gets `ERR_WRONG_SHARD`; 6-A retry routes to an alternate replica; client tokens unaffected.

- **D13. Non-goals (explicit).**
  - No two-phase / quorum eviction.
  - No memory-pressure probing (`mx.metal.get_active_memory()` or similar).
  - No eviction of bootstrap-held experts.
  - No cross-node eviction coordination.
  - No automatic retry of an evicted expert's pull (scanner picks it up on next tick if still hot).
  - No KV-cache eviction (separate concern; Phase 7+).

## 2. Components

### 2.1 `src/model_shard/partial_load.py` — `detach_expert`

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

    Raises KeyError if expert_id is not currently held.
    """
```

Implementation: for each `(proj, attr)` in `_PROJ_ATTR_ORDER`, compute surviving local slots = `[i for i, eid in enumerate(held) if eid != expert_id]`, then `proj.<attr> = mx.take(proj.<attr>, mx.array(surviving), axis=0)`. Update `held_ids_per_layer[layer_idx]`. `mx.eval` the 9 new tensors.

### 2.2 `src/model_shard/node.py` — new state + method

New instance state in `Node.__init__`:
```python
self._live_experts_attach_ts: dict[tuple[int, int], float] = {}
```

Populated by `migration_attach` (store `time.time()` for the new (L, E) key).

New method `migration_detach`:
```python
def migration_detach(self, layer_idx: int, expert_id: int) -> None:
    """Evict expert (layer_idx, expert_id). Raises LastReplicaError if
    evicting would leave no other live owner. Raises ValueError if the
    expert is bootstrap-held or within the attach cooldown."""
```

Flow:
1. Acquire `_live_experts_lock`. Assert expert is in `_live_experts[layer_idx]`.
2. Reject if expert_id in `self._shard.moe_experts.get(layer_idx, ())`.
3. Reject if `time.time() - self._live_experts_attach_ts.get((layer_idx, expert_id), 0) < _migration_evict_cooldown_s()`.
4. `other_owners = self.owners_of(layer_idx, expert_id) - {self._shard.shard_id}`. If empty, raise `LastReplicaError`.
5. Call `detach_expert(self._lm, layer_idx, expert_id, _MLX_COMPUTE_LOCK)`.
6. Remove from `_live_experts[layer_idx]`. Pop `_live_experts_attach_ts[(layer_idx, expert_id)]`.
7. Update `self._ownership_view[(self._shard.shard_id, layer_idx, expert_id)] = (1, now_ms)`.
8. If `self._membership is not None`: `self._membership.announce_ownership_remove(layer_idx, expert_id)`.

### 2.3 `Node` data-structure migration

Rename/promote:
- `Node._ownership_seen: set[tuple[str, int, int]]` → `Node._ownership_view: dict[tuple[str, int, int], tuple[int, int]]` (value = `(action, ts_unix_ms)`).
- `Node.owners_of(L, E)` rewritten to consult `_ownership_view` and return `{sid for (sid, ll, e), (act, _) in self._ownership_view.items() if ll == L and e == E and act == 0}`.
- Initialization at boot seeds bootstrap-held experts as `(ADD, 0)` (timestamp 0 so any later real gossip supersedes).

### 2.4 `Node._handle_expert_request` correctness fix

Change `hosted = set(self._shard.moe_experts.get(layer_idx, ()))` to `hosted = set(self._live_experts.get(layer_idx, set()))`. Add a comment noting the 6-C authority shift.

### 2.5 `src/model_shard/migration.py` — `_maybe_evict_one`

New method on `MigrationScanner`:

```python
def _maybe_evict_one(self) -> None:
    """Under capacity pressure at some layer, evict the coldest
    non-bootstrap, non-cooldown expert. Skips if no layer is at capacity
    or no eligible victim exists (e.g., all cold experts are within
    cooldown or evicting any would leave no other live owner)."""
```

Integration into `_scan_once`:

```python
def _scan_once(self) -> None:
    if not self._in_flight.acquire(blocking=False):
        return
    try:
        self._maybe_pull_one()
        if self._eviction_enabled:
            self._maybe_evict_one()
    finally:
        self._in_flight.release()
```

Victim selection algorithm:
```python
def _maybe_evict_one(self):
    for layer_idx, held in list(self._live_experts.items()):
        if len(held) < self._policy.max_experts_per_layer:
            continue
        eligible = held - self._bootstrap_held.get(layer_idx, set())
        eligible = {
            e for e in eligible
            if time.time() - self._attach_ts_provider(layer_idx, e)
               >= self._policy.evict_cooldown_s
        }
        if not eligible:
            continue
        victim = min(eligible, key=lambda e: self._heat_tracker.local_heat(layer_idx, e))
        try:
            self._evict_fn(layer_idx, victim)
        except LastReplicaError:
            continue  # try next layer
        return  # evicted one; done for this tick
```

`MigrationPolicy` grows two fields:
```python
evict_cooldown_s: float = 30.0
eviction_enabled: bool = True
```

`MigrationScanner` gains constructor parameters `bootstrap_held: dict[int, set[int]]`, `attach_ts_provider: Callable[[int, int], float]`, `evict_fn: Callable[[int, int], None]`.

### 2.6 `src/model_shard/membership/runner.py` — `announce_ownership_remove`

Symmetric to `announce_ownership_add` (5b Task 9). Both go into the same `_outbound_ownership` TTL queue; the receiver-side scrape already keys by `(shard_id, layer_idx, expert_id)` via the new `_ownership_view` semantics (D3).

```python
def announce_ownership_remove(self, layer_idx: int, expert_id: int) -> None:
    """Gossip an OwnershipDelta{action=REMOVE}. Same TTL+piggyback shape
    as announce_ownership_add."""
```

### 2.7 `src/model_shard/membership/records.py` — unchanged

`OwnershipDeltaRecord` already has the `action` field (from 5b Task 2). No schema change.

## 3. Wire Protocol

**No changes.** `OwnershipDelta.action=1` and `ts_unix_ms` are already on the wire from 5b. The `_OutboundOwnership` TTL queue and the fused piggyback walk from the 5b cleanup both already handle REMOVEs without modification.

## 4. Memory Model

Per-node resident memory delta when evicting one expert at layer 15: ~3.3 MB per expert (chassis + 9 tensors). For a 3090 with 24 GB total, eviction of 10 migration-added experts frees ~33 MB — small per eviction but keeps the ceiling from drifting up under sustained migration.

The `MIGRATION_MAX_EXPERTS_PER_LAYER` knob (5b) sets the per-layer ceiling. `ENABLE_EVICTION=true` (D11 default) makes that ceiling enforceable in practice, rather than just a rate-limit on pulls.

## 5. Testing Strategy

### 5.1 Fast unit tests

- `tests/test_partial_load_detach.py`:
  - `test_detach_expert_shrinks_stack_by_one` — synthetic `(k, ...)` stack, detach slot 1 of 4, assert shape[0] == 3 and held_ids matches.
  - `test_detach_expert_preserves_other_rows_bit_exactly` — after detach, the surviving rows are byte-identical to originals via `mx.array_equal`.
  - `test_attach_detach_roundtrip_is_identity` — a sequence `attach_expert(... E) → detach_expert(... E)` returns the stack to its original state byte-for-byte.
  - `test_detach_expert_raises_on_not_held` — detach a missing expert raises KeyError.

- `tests/test_ownership_view_convergence.py`:
  - `test_add_then_remove_last_writer_wins` — ADD{t=1} then REMOVE{t=2}; view shows REMOVE.
  - `test_remove_then_add_older_drops` — REMOVE{t=2} then ADD{t=1}; REMOVE stays.
  - `test_ownership_view_public_api_returns_only_adds` — a view with one ADD and one REMOVE exposes only the ADD key from `ownership_view()`.

- `tests/test_node_eviction.py`:
  - `test_migration_detach_updates_state` — mocked `lm`; `migration_detach` removes expert from `_live_experts`, updates `_ownership_view`, calls `announce_ownership_remove`.
  - `test_migration_detach_rejects_bootstrap_held` — eviction of a YAML-declared expert raises ValueError.
  - `test_migration_detach_rejects_within_cooldown` — attach at t=0, attempt evict at t=5 (cooldown=30); raises ValueError.
  - `test_migration_detach_last_replica_raises` — if `owners_of` returns only self; raises LastReplicaError.
  - `test_migration_detach_succeeds_with_multiple_owners` — `owners_of` returns `{self, peer}`; detach proceeds.

- `tests/test_migration_scanner_eviction.py`:
  - `test_scan_once_evicts_when_at_capacity` — capacity=2, `_live_experts[L]={E1, E2}`, both non-bootstrap; coldest gets evicted.
  - `test_scan_once_skips_eviction_under_capacity` — capacity=10, len=5; no eviction.
  - `test_scan_once_skips_bootstrap_held` — all at-capacity experts are bootstrap; no eviction.
  - `test_scan_once_skips_within_cooldown` — coldest is within cooldown; no eviction this tick.

- `tests/test_handle_expert_request_authority.py`:
  - `test_handle_expert_request_uses_live_experts` — migrate-in expert E not in bootstrap; `_handle_expert_request` serves it.
  - `test_handle_expert_request_rejects_evicted` — evict an expert; subsequent `_handle_expert_request` returns `ERR_WRONG_SHARD`.

### 5.2 Slow tests

- `tests/test_eviction_e2e.py`:
  - `test_full_attach_evict_cycle_over_tcp` — 3-node cluster; force a migration pull of expert E to node B; then force-evict E; verify `ownership_view` on peers A and C converges to "B does NOT own E"; Tier 1 tokens continue correctly (routing falls back to original owner).

- `tests/test_eviction_race_with_expert_request.py`:
  - `test_inflight_expert_request_completes_before_eviction` — monkeypatch: evict during in-flight ExpertRequest for the same expert; verify compute finishes on the pre-eviction stack, subsequent request returns `ERR_WRONG_SHARD`, 6-A retry succeeds on an alternate replica.

### 5.3 Regression

- All Phase 1-6B fast + slow tests pass with `ENABLE_EVICTION=false` (treats Phase 6-C as a no-op).
- `ENABLE_EVICTION=true` Tier 1 E2E passes (eviction never fires because capacity isn't hit on Phase 4's overlapping config — same as 5b's regression story).

## 6. Risks & Mitigations

- **R1 — Last-replica race.** Two nodes holding the last two replicas both pass the local "other owner exists" check and both evict. Mitigation: narrow window (local lock + single in-flight + ~1 gossip round convergence); documented-and-accepted.

- **R2 — Thrashing under sustained hot traffic.** A hot expert is pulled, heat doesn't decay fast enough, then gets evicted under capacity, then re-pulled. Mitigation: 30s cooldown + capacity-only trigger (not heat-only) makes oscillation cost-bounded.

- **R3 — Provenance transient rejection.** Addressed by 6-B's Task-10 error path and 6-A's retry. No new code needed in 6-C.

- **R4 — Clock skew across cluster.** Last-writer-wins on `ts_unix_ms` assumes NTP-synced clocks (~ms accuracy). Documented constraint; acceptable for research prototype.

- **R5 — Scanner starvation.** `_maybe_evict_one` only runs after `_maybe_pull_one` under the same in-flight lock. A node stuck in a "pull succeeds every tick" regime never evicts, even at capacity. Mitigation: the pull policy already refuses to pull when `len(_live_experts[L]) >= max_experts_per_layer` (5b), so pull-stuck-at-capacity isn't possible; once at capacity, pulls skip and eviction gets its turn.

- **R6 — Unknown-shard evictions from gossip.** If a node receives `OwnershipDelta{REMOVE}` for a (shard, L, E) it never had ADD for, or the remote shard_id is unknown to the receiver: harmless — the incoming delta overwrites nothing; `ownership_view()` continues not to include it.

## 7. References

- Phase 5b spec: `docs/superpowers/specs/2026-04-17-phase5b-dynamic-migration-design.md` (§D10 OwnershipDelta, §D11 replication-only carryover)
- Phase 6-A spec: `docs/superpowers/specs/2026-04-17-phase6a-expert-retry-design.md` (§D7 typed ExpertRpcFailure, excluded-peer flow)
- Phase 6-B spec: `docs/superpowers/specs/2026-04-17-phase6b-provenance-verification-design.md` (§D8 Rule 5 authorization via owners_of; §Task-10-gap tail→head error propagation)
- `src/model_shard/partial_load.py` — existing `attach_expert` (5b Task 6); 6-C adds the inverse.
- `src/model_shard/migration.py` — existing `MigrationScanner` (5b Tasks 15-17); 6-C extends.
- `src/model_shard/membership/runner.py` — existing `announce_ownership_add` + TTL queue (5b Task 9); 6-C adds the REMOVE sibling.
