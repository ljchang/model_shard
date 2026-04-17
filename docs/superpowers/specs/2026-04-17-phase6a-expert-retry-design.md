# Phase 6-A — Expert-Peer Retry

**Status:** draft, 2026-04-17
**Scope:** When a peer dies mid-`ExpertRequest` fan-out, the node waiting on it retries the failed expert subset to an alternate replica, preserving partial results from peers that already completed. First sub-project of Phase 6 (Fault Tolerance & Verification).

## 1. Background & Decisions

### 1.1 Why now
Phase 5b proved that experts can be replicated across nodes and that routing can dynamically pick up new replicas. But when a replica-holding peer dies mid-inference, the current behavior aborts the entire request with `Error{ERR_SHARD_UNAVAILABLE}` — wasting 5b's replication. Phase 6-A closes this gap: replicas existing AND replicas being used for recovery. No cold-plane changes required; 5b's `live_owners_provider` is sufficient.

### 1.2 Decomposition of Phase 6
Phase 6 per the roadmap is "Fault Tolerance & Verification." Reading the spec and the existing `ProvenanceEntry` groundwork (`src/model_shard/request.py:14-19`), this naturally decomposes into three independent sub-projects:

- **6-A (this spec):** Expert-peer retry. Fault recovery when a replica owner dies during fan-out.
- **6-B (future):** Provenance-chain verification. Hash-based detection of lying/compromised peers.
- **6-C (future):** Eviction + REMOVE `OwnershipDelta`. Memory-bounded deployments on 24 GB 3090s.

Each has its own brainstorm → spec → plan cycle.

### 1.3 Decisions

- **D1. Scope.** Expert-peer failure only. Out of scope: pipeline-peer failure (needs cold-plane redundancy design, a separate effort); head-peer failure (client-side story); Byzantine/compromised peers (sub-project 6-B).

- **D2. Retry mechanism: bounded retries, not circuit-breaker or hedging.** Circuit-breaker is a Phase 7 polish; hedging costs 2× bandwidth for single-digit latency wins — not worth it on a research prototype.

- **D3. Retry placement.** Inside `ExpertOrchestrator.run_split_layer` Phase B, wrapping `_gather_with_abort`. Phase A (local compute) and Phase C (aggregation) remain unchanged. Retry is local to the node whose fan-out the failure occurred in — preserves the decentralized-no-orchestrator architectural invariant.

- **D4. Partial-success preservation.** The `outputs: dict[int, mx.array]` accumulated by `_gather_with_abort` is preserved across retries. Only the expert ids that did NOT complete (the ones owned by the failed peer) are re-dispatched. Completed peers' results are never re-run.

- **D5. Retry budget.** 3 attempts total (1 initial + 2 retries). Backoff sequence indexed by retry number: `backoff_ms[0]` = sleep before retry 1, `backoff_ms[1]` = sleep before retry 2. Default `(100, 500)` (in ms). If `backoff_ms` has fewer elements than retries, the last element is reused. On exhaustion: raise `ExpertRpcFailure`, which the existing decode loop translates to `Error{ERR_SHARD_UNAVAILABLE, is_final=true}`. Defaults chosen for fast failure signaling to the client without thrashing peers.

- **D6. Excluded-peer memory.** Local to a single `run_split_layer` invocation. Each time a peer RPC fails, its shard_id is added to a local `excluded_peers: set[str]` that filters `live_owners_provider` result for the remainder of this fan-out. No cross-invocation memory (no persistent circuit-breaker). The assumption: SWIM gossip will converge on the peer's DEAD state before the next fan-out starts, so future invocations naturally avoid it via `live_owners_provider`.

- **D7. API surface.** Minimal.
  - `ExpertRpcFailure.failed_peer: str` — promote the already-present info from the message to a typed field. Required for the retry loop.
  - `ExpertOrchestrator` fields: `retry_max_attempts: int = 3` and `retry_backoff_ms: tuple[int, ...] = (0, 100, 500)`, env-driven defaults.
  - `live_owners_provider` signature unchanged.

- **D8. Correctness bar.** Bit-exact under retry. For the same input `h` and expert ids `E`, `run_split_layer` with peer B failing once and retrying to replica C produces `mx.array_equal` output to `run_split_layer` with no failure. This is a direct corollary of 5b's migration bit-exactness proof (which showed any valid replica of expert E produces the same output as any other).

- **D9. Gate.** `ENABLE_EXPERT_RETRY=true` default. Set to `false` or `EXPERT_RETRY_MAX_ATTEMPTS=0` to reproduce pre-6-A "fail-fast" behavior for regression tests that need it.

- **D10. Non-goals (explicit).**
  - No persistent circuit-breaker; excluded-peer set is per-invocation.
  - No hedged/parallel redundant requests.
  - No retry for Phase A local-compute failures (OOM on self).
  - No retry when the last-standing owner was the failed peer.
  - No cold-plane changes.
  - No retry for ExpertWeightRequest (migration pull); that already has a "retry on next scan" fallback via the `MigrationScanner`.

## 2. Components

### 2.1 `src/model_shard/expert_orchestrator.py`

**Changes to `ExpertRpcFailure`:**
```python
class ExpertRpcFailure(RuntimeError):
    def __init__(
        self, message: str, *, failed_peer: str, layer_idx: int
    ) -> None:
        super().__init__(message)
        self.failed_peer = failed_peer
        self.layer_idx = layer_idx
```

All existing raise sites in `expert_orchestrator.py` updated to pass the peer name and layer as typed args. Message string preserved for log-compat.

**New fields on `ExpertOrchestrator`:**
```python
    retry_max_attempts: int = 3
    retry_backoff_ms: tuple[int, ...] = (100, 500)
```

**Phase B modification in `run_split_layer`:** replace the single `self._gather_with_abort(...)` call with a retry loop:

```python
excluded_peers: set[str] = set()
attempts = 0
while True:
    attempts += 1
    try:
        self._gather_with_abort(futures, abort_events, outputs, layer_idx)
        break
    except ExpertRpcFailure as exc:
        if attempts >= self.retry_max_attempts:
            raise
        excluded_peers.add(exc.failed_peer)
        # attempts=1 means first call failed; we're about to do retry 1.
        # backoff_ms[0] sleeps before retry 1; backoff_ms[1] before retry 2.
        backoff_idx = min(attempts - 1, len(self.retry_backoff_ms) - 1)
        time.sleep(self.retry_backoff_ms[backoff_idx] / 1000.0)

        # Determine which experts still need computing.
        missing = [eid for eid in all_ids if eid not in outputs]

        # Re-route, excluding the peer(s) we've seen fail.
        def _filtered_provider(eid: int) -> set[str]:
            raw = (
                self.live_owners_provider(eid)
                if self.live_owners_provider is not None
                else set()
            )
            return raw - excluded_peers

        by_owner = group_expert_ids_by_owner_loaded(
            missing,
            owners={
                sid: ids for sid, ids in self.owners.items()
                if sid not in excluded_peers
            },
            peer_loads=self.loads_provider(),
            self_shard_id=self.self_shard_id,
            self_load=self.loads_provider().get(self.self_shard_id, 0),
            rng=self.rng,
            live_owners_provider=_filtered_provider,
        )

        # Any experts routed to self? Run them inline under the MLX lock.
        local_ids = by_owner.pop(self.self_shard_id, [])
        if local_ids:
            with self._mlx_guard():
                local_retry = run_selected_experts(lm, post_attn, layer_idx, local_ids)
                outputs.update(local_retry)

        # Re-submit futures for surviving peers.
        futures = {
            peer: self._executor.submit(
                self.peer_rpc.call, peer, request_id, layer_idx, ids, post_attn
            )
            for peer, ids in by_owner.items()
        }
        abort_events = {peer: threading.Event() for peer in futures}
        if abort_events:
            with self._in_flight_lock:
                self._in_flight[request_id] = abort_events
```

**Observer-abort compatibility:** if a peer that was excluded earlier in this invocation then leaves ALIVE via SWIM, `notify_peer_left_alive` will still set its abort event — but that peer has no futures targeting it in the current iteration, so the event fires harmlessly. No change to the observer path.

### 2.2 `src/model_shard/node.py`

`Node.__init__` reads the new env vars and passes them to `_build_expert_orchestrator`:

```python
self._retry_max_attempts = _expert_retry_max_attempts()
self._retry_backoff_ms = _expert_retry_backoff_ms()
```

`_build_expert_orchestrator` extends the `ExpertOrchestrator(...)` call:

```python
return ExpertOrchestrator(
    ...,
    retry_max_attempts=self._retry_max_attempts,
    retry_backoff_ms=self._retry_backoff_ms,
)
```

Env-var helpers at module level:

```python
def _expert_retry_enabled() -> bool:
    return os.environ.get("ENABLE_EXPERT_RETRY", "true").lower() in ("1", "true", "yes")


def _expert_retry_max_attempts() -> int:
    if not _expert_retry_enabled():
        return 1  # fail-fast, legacy Phase 3 behavior
    return int(os.environ.get("EXPERT_RETRY_MAX_ATTEMPTS", "3"))


def _expert_retry_backoff_ms() -> tuple[int, ...]:
    raw = os.environ.get("EXPERT_RETRY_BACKOFF_MS", "100,500")
    return tuple(int(x.strip()) for x in raw.split(",") if x.strip())
```

### 2.3 No wire-protocol changes

Retry is entirely local. No proto edits, no gossip changes.

## 3. Memory & Performance

Retry loop allocates no additional tensors beyond what `_gather_with_abort` already accumulates. Backoff is synchronous `time.sleep` — blocks the decode-loop thread for up to ~600ms across 3 attempts in the worst case. Acceptable for a research prototype on a small cluster; revisit with async retries if a future phase shows this latency regress on long decode runs.

## 4. Testing Strategy

### 4.1 Fast unit tests

`tests/test_expert_retry_unit.py`:

- `test_retry_succeeds_on_second_attempt_to_replica` — `peer_rpc` mock fails once on peer B, succeeds to peer C; assert `outputs` contains all expected experts.
- `test_partial_outputs_preserved_across_retry` — 3-peer fan-out (A local, B peer, C peer). B fails, C has already returned. Assert C's outputs survive and only B's subset is retried.
- `test_retry_exhaustion_raises_with_typed_failure` — all peers fail; final `ExpertRpcFailure` has `failed_peer` set and `retry_max_attempts` reached.
- `test_excluded_peer_stays_excluded_within_invocation` — peer B fails, retry routes to C. Subsequent expert on same invocation that was nominally owned by B also goes to C.
- `test_retry_disabled_via_env_matches_phase5b_behavior` — `retry_max_attempts=1` (single attempt), first failure immediately bubbles up.

### 4.2 Fast correctness (bit-exact)

Extend `tests/test_expert_orchestrator.py`:

- `test_retry_produces_bit_exact_output` — inject a `peer_rpc` that fails once on a specific expert subset, then retries successfully. Compare the final `run_split_layer` output to a run without the injected failure. Assert `mx.array_equal`. Relies on 5b's bit-exactness property that any valid replica produces the same output.

### 4.3 Slow E2E

`tests/test_expert_retry_e2e.py`:

- 3-node cluster with Phase 4's overlapping `moe_experts` config (experts 0/1/2 replicated on two shards each). Start a long generation (≥32 tokens to cross multiple decode rounds). Kill one of the two replica owners for layer 15 during decode. Assert the generation completes without error and the tokens match the Phase 1 reference for the prompts that route through overlapping experts. Uses the same kill-mid-decode pattern as Task 22's decode-hang E2E.

### 4.4 Regression

- All Phase 3 / 4 / 5b slow tests must still pass with defaults (`ENABLE_EXPERT_RETRY=true`). Since retries only fire on failure, defaults-on is safe.
- `ENABLE_EXPERT_RETRY=false` reproduces Phase 3 behavior exactly (first failure aborts the fan-out).

## 5. Acceptance

1. `ruff check`, `mypy` clean.
2. Fast suite green (including new retry unit + correctness tests).
3. New slow E2E test passes.
4. Phase 3/4/5b slow regression green with defaults.
5. With `ENABLE_EXPERT_RETRY=false`, the behavior is indistinguishable from Phase 5b (first failure aborts).
6. README Phase 6-A status paragraph added; memory file updated.

## 6. Risks & Mitigations

- **R1 — Retry during cluster-wide degradation.** If many peers are failing simultaneously (network partition), every fan-out retries 3× and sleeps up to 600ms. For a 30-layer pipeline with N MoE layers, this could add seconds per token. Mitigation: SWIM gossip converges on DEAD peers within `SUSPECT_PERIOD + 1s`; once converged, `live_owners_provider` excludes them and retry skips the dead-peer dispatch entirely. Transient amplification is bounded by `SUSPECT_PERIOD`.

- **R2 — Thundering herd on the replica.** When peer B dies, every node's orchestrator simultaneously shifts its B-bound traffic to replica C. C's queue depth spikes. Mitigation: Phase 4's P2C routing (`live_owners_provider` + `loads_provider`) already load-balances across replicas; the P2C sample spreads retry dispatches. For single-replica experts (degenerate P2C = one choice), this amplifies C's load but not for long — SWIM converges and the system re-balances.

- **R3 — Retry loop observes stale `live_owners_provider`.** `live_owners_provider` reads gossip state that may still include the dead peer for up to ~SUSPECT_PERIOD before convergence. Mitigation: the local `excluded_peers` set filters the callback's result inside the retry loop, so even stale gossip can't re-select the known-dead peer within the same invocation.

- **R4 — Retry budget too low / too high.** 3 attempts may fail if a cluster has many replicas degrading together. 10 attempts may waste latency in the common case. Mitigation: env-var tunable. Default of 3 chosen for the expected Phase 5b deployment (≤3 replicas per expert); deployments with more replicas should raise the default.

- **R5 — Peer-A excluded set isn't propagated to upstream / downstream.** A peer we excluded locally may still be routed to by other nodes. Acceptable: each node's orchestrator is independently responsible for its own fan-outs. Cross-node exclusion would require a new gossip surface (Phase 7).

## 7. References

- Phase 5b spec: `docs/superpowers/specs/2026-04-17-phase5b-dynamic-migration-design.md` (§D9 live_owners_provider, §D4 replica semantics)
- Phase 4 spec: `docs/superpowers/specs/2026-04-16-phase4-load-aware-routing-design.md` (§P2C routing)
- Phase 3 spec: `docs/superpowers/specs/2026-04-16-phase3-expert-sharding-design.md` (§orchestrator structure)
- Task 18 (decode-hang fix): `docs/superpowers/plans/2026-04-17-phase5b-dynamic-migration.md` Task 18 (for the mid-decode kill test pattern)
- `src/model_shard/expert_orchestrator.py` — existing `_gather_with_abort` and `ExpertRpcFailure` class.
- `src/model_shard/request.py:14-19` — `ProvenanceEntry.hash` field (reserved for future 6-B, not touched here).
