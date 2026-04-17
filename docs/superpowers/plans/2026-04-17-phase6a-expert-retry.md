# Phase 6-A Expert-Peer Retry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a peer dies mid-`ExpertRequest` fan-out, the node waiting on it retries the failed expert subset to an alternate replica, preserving partial results from peers that already completed — bit-exactly equivalent to a no-failure run.

**Architecture:** `ExpertOrchestrator.run_split_layer` Phase B gains a retry loop around `_gather_with_abort`. Typed `ExpertRpcFailure.failed_peer` field lets the retry loop exclude known-failed peers from the next dispatch. The excluded-peer set is local to a single invocation (no persistent circuit-breaker). Routing filters through `live_owners_provider` as in Phase 5b; gossip eventually converges on the DEAD peer so subsequent invocations avoid it naturally.

**Tech Stack:** Python 3.13, MLX, existing protobuf/TCP transport (no wire changes), pytest with `slow` marker for model-loading tests.

**Spec:** `docs/superpowers/specs/2026-04-17-phase6a-expert-retry-design.md` (decisions D1-D10).

---

## File Structure

**Modify:**
- `src/model_shard/expert_orchestrator.py` — `ExpertRpcFailure` typed fields (`failed_peer`, `layer_idx`); new `ExpertOrchestrator` fields `retry_max_attempts`, `retry_backoff_ms`; retry loop inside `run_split_layer` Phase B.
- `src/model_shard/node.py` — env-var helpers (`_expert_retry_enabled`, `_expert_retry_max_attempts`, `_expert_retry_backoff_ms`); pass retry fields into `_build_expert_orchestrator`.

**Create:**
- `tests/test_expert_retry_unit.py` — 5 fast unit tests using mocked `peer_rpc`.
- `tests/test_expert_retry_bit_exact.py` — fast correctness test: no-failure run == one-shot-failure-plus-retry run.
- `tests/test_expert_retry_e2e.py` — slow E2E: 3-node cluster with overlapping replicas, kill one replica mid-generation, assert tokens match reference.

**Update at the end:**
- `README.md` — Phase 6-A status paragraph.
- Memory file at `/Users/lukechang/.claude/projects/-Users-lukechang-Github-model-shard/memory/project_gossip_moe.md` — Phase 6-A COMPLETE entry.

---

## Task ordering

1. `ExpertRpcFailure` typed fields + regression check (pre-req for the retry loop to extract `failed_peer`).
2. `ExpertOrchestrator` retry fields + env-var plumbing in `Node`.
3. Retry loop in `run_split_layer` Phase B + fast unit tests (TDD).
4. Fast bit-exact correctness test.
5. Slow E2E — kill-replica mid-generation.
6. README + memory update.

---

### Task 1: Typed `ExpertRpcFailure` fields

**Files:**
- Modify: `src/model_shard/expert_orchestrator.py` — `ExpertRpcFailure` class + all raise sites in the file
- Test: `tests/test_expert_retry_unit.py` (create, add this test)

- [ ] **Step 1: Write the failing test**

Create `tests/test_expert_retry_unit.py`:

```python
"""Phase 6-A expert-retry unit tests."""
from __future__ import annotations

import pytest

from model_shard.expert_orchestrator import ExpertRpcFailure


def test_expert_rpc_failure_has_typed_fields():
    exc = ExpertRpcFailure(
        "peer 'B' died", failed_peer="B", layer_idx=15
    )
    assert exc.failed_peer == "B"
    assert exc.layer_idx == 15
    assert "peer 'B' died" in str(exc)


def test_expert_rpc_failure_rejects_missing_typed_fields():
    # Positional message-only construction should fail — these fields are required.
    with pytest.raises(TypeError):
        ExpertRpcFailure("something broke")  # type: ignore[call-arg]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_expert_retry_unit.py::test_expert_rpc_failure_has_typed_fields -v`
Expected: FAIL — `ExpertRpcFailure` doesn't yet accept `failed_peer`/`layer_idx` kwargs.

- [ ] **Step 3: Modify `ExpertRpcFailure` to accept typed fields**

In `src/model_shard/expert_orchestrator.py`, find the existing `ExpertRpcFailure` class (around line 41):

```python
class ExpertRpcFailure(RuntimeError):  # noqa: N818 — explicit name per plan
    """Raised by ExpertOrchestrator when a peer RPC fails..."""
```

Replace with:

```python
class ExpertRpcFailure(RuntimeError):  # noqa: N818 — explicit name per plan
    """Raised by ExpertOrchestrator when a peer RPC fails (timeout, broken
    pipe, observer-triggered close). The node's request handler translates
    this into Error{SHARD_UNAVAILABLE, is_final=true} for the client.

    Phase 6-A: gains typed ``failed_peer`` and ``layer_idx`` fields so the
    retry loop in ``run_split_layer`` can exclude the known-failed peer
    from subsequent dispatches."""

    def __init__(self, message: str, *, failed_peer: str, layer_idx: int) -> None:
        super().__init__(message)
        self.failed_peer = failed_peer
        self.layer_idx = layer_idx
```

- [ ] **Step 4: Update every raise site in `expert_orchestrator.py`**

Find every `raise ExpertRpcFailure(...)` in `expert_orchestrator.py` and add the typed kwargs. There are 3 raise sites in `_gather_with_abort` (around lines 264, 278, 291):

```python
# Site 1 (peer left ALIVE, ~line 264):
raise ExpertRpcFailure(
    f"peer {peer!r} left ALIVE mid-request for layer {layer_idx}",
    failed_peer=peer,
    layer_idx=layer_idx,
)

# Site 2 (peer RPC raised, ~line 278):
raise ExpertRpcFailure(
    f"expert RPC to peer {peer!r} failed for layer {layer_idx}: {e}",
    failed_peer=peer,
    layer_idx=layer_idx,
) from e

# Site 3 (timeout, ~line 291):
raise ExpertRpcFailure(
    f"expert RPC to peer {stuck_peer!r} failed for layer {layer_idx}: "
    f"timeout after {self.rpc_timeout_s}s",
    failed_peer=stuck_peer,
    layer_idx=layer_idx,
)
```

- [ ] **Step 5: Run new test to verify pass**

Run: `uv run pytest tests/test_expert_retry_unit.py -v`
Expected: 2 PASS.

- [ ] **Step 6: Regression check — existing ExpertRpcFailure consumers must still work**

Run:
```bash
uv run pytest tests/test_expert_orchestrator.py tests/test_expert_orchestrator_observer.py tests/test_expert_orchestrator_timeout.py tests/test_expert_rpc_failure.py tests/test_expert_rpc_handler.py -v
```
Expected: all PASS. Any test that catches `ExpertRpcFailure` still does because the class is still a `RuntimeError` subclass with a string message. Any test that constructs an `ExpertRpcFailure` directly must pass the new typed kwargs — if any test file has `ExpertRpcFailure("...")` with no kwargs, those tests will fail with TypeError and you must update them to pass `failed_peer=...` and `layer_idx=...` with whatever placeholder values make sense for the test (e.g., `failed_peer="test-peer", layer_idx=0`).

- [ ] **Step 7: Ruff + mypy clean**

```bash
uv run ruff check src/model_shard/expert_orchestrator.py tests/test_expert_retry_unit.py
uv run mypy src/model_shard/expert_orchestrator.py
```

- [ ] **Step 8: Commit**

```bash
git add src/model_shard/expert_orchestrator.py tests/test_expert_retry_unit.py
git commit -m "Phase 6-A Task 1: typed ExpertRpcFailure.failed_peer + layer_idx"
```

---

### Task 2: `ExpertOrchestrator` retry fields + `Node` env-var plumbing

**Files:**
- Modify: `src/model_shard/expert_orchestrator.py`
- Modify: `src/model_shard/node.py`
- Test: `tests/test_expert_retry_unit.py` (extend)

- [ ] **Step 1: Append failing test**

Append to `tests/test_expert_retry_unit.py`:

```python
from unittest.mock import MagicMock

from model_shard.expert_orchestrator import ExpertOrchestrator


def test_orchestrator_accepts_retry_fields_defaults():
    orch = ExpertOrchestrator(
        self_shard_id="A",
        owners={"A": {3}},
        peer_rpc=MagicMock(),
        rpc_timeout_s=1.0,
    )
    assert orch.retry_max_attempts == 3
    assert orch.retry_backoff_ms == (100, 500)


def test_orchestrator_accepts_explicit_retry_fields():
    orch = ExpertOrchestrator(
        self_shard_id="A",
        owners={"A": {3}},
        peer_rpc=MagicMock(),
        rpc_timeout_s=1.0,
        retry_max_attempts=5,
        retry_backoff_ms=(10, 50, 200),
    )
    assert orch.retry_max_attempts == 5
    assert orch.retry_backoff_ms == (10, 50, 200)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_expert_retry_unit.py::test_orchestrator_accepts_retry_fields_defaults -v`
Expected: AttributeError — `retry_max_attempts` does not exist on `ExpertOrchestrator`.

- [ ] **Step 3: Add retry fields to `ExpertOrchestrator`**

In `src/model_shard/expert_orchestrator.py`, find the `ExpertOrchestrator` dataclass field list (ends with `live_owners_provider` and `heat_observer` — after the Phase 5b additions). Add two new fields BEFORE the `_executor` / `_in_flight*` init=False fields:

```python
    retry_max_attempts: int = 3
    retry_backoff_ms: tuple[int, ...] = (100, 500)
```

- [ ] **Step 4: Add env-var helpers at the bottom of `node.py`**

In `src/model_shard/node.py`, alongside the existing `_gossip_enabled`, `_expert_shard_enabled`, `_partial_load_enabled`, `_dynamic_migration_enabled` helpers, add:

```python
def _expert_retry_enabled() -> bool:
    return os.environ.get("ENABLE_EXPERT_RETRY", "true").lower() in (
        "1", "true", "yes"
    )


def _expert_retry_max_attempts() -> int:
    if not _expert_retry_enabled():
        return 1  # fail-fast, legacy Phase 3 behavior
    return int(os.environ.get("EXPERT_RETRY_MAX_ATTEMPTS", "3"))


def _expert_retry_backoff_ms() -> tuple[int, ...]:
    raw = os.environ.get("EXPERT_RETRY_BACKOFF_MS", "100,500")
    return tuple(int(x.strip()) for x in raw.split(",") if x.strip())
```

- [ ] **Step 5: Wire the env values into `_build_expert_orchestrator`**

Find `_build_expert_orchestrator` in `src/model_shard/node.py` (around line 675). In the `ExpertOrchestrator(...)` constructor call, add two new kwargs:

```python
        return ExpertOrchestrator(
            self_shard_id=self._shard.shard_id,
            owners=owners,
            peer_rpc=TcpPeerRPC(addresses=addresses, timeout_s=30.0),
            rpc_timeout_s=30.0,
            mlx_lock=_MLX_COMPUTE_LOCK,
            loads_provider=_loads_provider,
            rng=_random_mod.Random(),
            live_owners_provider=self.owners_of,  # Phase 5b (unchanged)
            heat_observer=self._heat_tracker.observe,  # Phase 5b (unchanged)
            retry_max_attempts=_expert_retry_max_attempts(),
            retry_backoff_ms=_expert_retry_backoff_ms(),
        )
```

If the existing code has these kwargs in different positions, match that style — the key change is adding the two new retry kwargs.

- [ ] **Step 6: Run tests to verify pass**

Run: `uv run pytest tests/test_expert_retry_unit.py -v`
Expected: 4 PASS (2 from Task 1 + 2 new).

- [ ] **Step 7: Regression**

Run: `uv run pytest tests/test_node_membership.py tests/test_node_load_wiring.py tests/test_node_partial_load_wiring.py tests/test_node_live_experts.py tests/test_dynamic_migration_gate.py tests/test_expert_orchestrator.py -v -m "not slow"`
Expected: all pass (additive fields have safe defaults).

- [ ] **Step 8: Ruff + mypy clean**

```bash
uv run ruff check src/model_shard/expert_orchestrator.py src/model_shard/node.py tests/test_expert_retry_unit.py
uv run mypy src/model_shard/expert_orchestrator.py src/model_shard/node.py
```

- [ ] **Step 9: Commit**

```bash
git add src/model_shard/expert_orchestrator.py src/model_shard/node.py tests/test_expert_retry_unit.py
git commit -m "Phase 6-A Task 2: ExpertOrchestrator retry fields + Node env plumbing"
```

---

### Task 3: Retry loop in `run_split_layer` Phase B + unit tests

**Files:**
- Modify: `src/model_shard/expert_orchestrator.py` (retry loop)
- Modify: `tests/test_expert_retry_unit.py` (append 5 retry-behavior tests)

This is the load-bearing task. TDD: unit tests first, each probing a single invariant.

- [ ] **Step 1: Append the 5 retry-behavior tests**

Append to `tests/test_expert_retry_unit.py`:

```python
import threading
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx


@dataclass
class _FlakyPeerRPC:
    """Test double: peer_rpc that fails on a set of peer shard_ids the first
    time they're called, then succeeds thereafter. Returns per-expert fake
    tensors keyed by expert id."""

    fail_once_for: set[str] = field(default_factory=set)
    _already_failed: set[str] = field(default_factory=set)
    calls: list[tuple[str, list[int]]] = field(default_factory=list)

    def call(
        self,
        peer_shard_id: str,
        request_id: str,
        layer_idx: int,
        expert_ids: list[int],
        h: mx.array,
    ) -> dict[int, mx.array]:
        self.calls.append((peer_shard_id, list(expert_ids)))
        if peer_shard_id in self.fail_once_for and peer_shard_id not in self._already_failed:
            self._already_failed.add(peer_shard_id)
            raise RuntimeError(f"injected failure for {peer_shard_id}")
        return {
            eid: mx.full((1, 1, 8), fill_value=float(eid), dtype=mx.bfloat16)
            for eid in expert_ids
        }


def _run_test_fanout(
    *,
    owners: dict[str, set[int]],
    ids_to_fan: list[int],
    peer_rpc: Any,
    live_owners: dict[int, set[str]],
    max_attempts: int = 3,
    backoff_ms: tuple[int, ...] = (0, 0),
) -> tuple[dict[int, mx.array], Any]:
    """Exercise just the Phase B retry loop by constructing an orchestrator
    and invoking its internal `_phase_b_with_retry` helper. Returns (outputs, orch)."""
    import random

    orch = ExpertOrchestrator(
        self_shard_id="self",
        owners=owners,
        peer_rpc=peer_rpc,
        rpc_timeout_s=1.0,
        rng=random.Random(0),
        live_owners_provider=lambda eid: live_owners.get(eid, set()),
        retry_max_attempts=max_attempts,
        retry_backoff_ms=backoff_ms,
    )
    post_attn = mx.zeros((1, 1, 8), dtype=mx.bfloat16)
    outputs = orch._phase_b_with_retry(
        post_attn=post_attn,
        all_ids=ids_to_fan,
        layer_idx=15,
        request_id="req-1",
        initial_local_ids=[],
        lm=None,  # local retry not exercised in these tests (see separate test).
    )
    orch.close()
    return outputs, orch


def test_retry_succeeds_on_second_attempt_to_replica():
    owners = {"B": {7}, "C": {7}}
    live = {7: {"B", "C"}}
    rpc = _FlakyPeerRPC(fail_once_for={"B"})
    outputs, _ = _run_test_fanout(
        owners=owners, ids_to_fan=[7], peer_rpc=rpc, live_owners=live,
    )
    assert 7 in outputs
    # The rpc was called twice: once to B (failed), once to C (succeeded).
    peers_called = [p for p, _ in rpc.calls]
    assert "B" in peers_called
    assert "C" in peers_called


def test_partial_outputs_preserved_across_retry():
    # B owns {3}, C owns {7}, D owns {11}. B fails; C and D succeed.
    # After retry (B's work re-routed), expect all three outputs.
    owners = {"B": {3}, "C": {7}, "D": {11}}
    # B has a replica on E.
    live = {3: {"B", "E"}, 7: {"C"}, 11: {"D"}}
    rpc = _FlakyPeerRPC(fail_once_for={"B"})
    outputs, _ = _run_test_fanout(
        owners=owners | {"E": {3}}, ids_to_fan=[3, 7, 11],
        peer_rpc=rpc, live_owners=live,
    )
    assert set(outputs.keys()) == {3, 7, 11}
    # C and D each called once — not re-run.
    c_calls = [ids for p, ids in rpc.calls if p == "C"]
    d_calls = [ids for p, ids in rpc.calls if p == "D"]
    assert len(c_calls) == 1
    assert len(d_calls) == 1


def test_retry_exhaustion_raises_typed_failure():
    # Single-owner expert, only owner fails — no replica, exhaust retries.
    owners = {"B": {7}}
    live = {7: {"B"}}
    rpc = _FlakyPeerRPC(fail_once_for={"B", "C"})  # C excluded/not present
    with pytest.raises(ExpertRpcFailure) as excinfo:
        _run_test_fanout(
            owners=owners, ids_to_fan=[7], peer_rpc=rpc, live_owners=live,
            max_attempts=3,
        )
    assert excinfo.value.failed_peer == "B"
    assert excinfo.value.layer_idx == 15


def test_excluded_peer_stays_excluded_within_invocation():
    # B owns {3, 11}; E owns {3}; F owns {11}. B fails once on call routing both.
    # Retry should route 3 to E and 11 to F — not back to B.
    owners = {"B": {3, 11}, "E": {3}, "F": {11}}
    live = {3: {"B", "E"}, 11: {"B", "F"}}
    rpc = _FlakyPeerRPC(fail_once_for={"B"})
    outputs, _ = _run_test_fanout(
        owners=owners, ids_to_fan=[3, 11], peer_rpc=rpc, live_owners=live,
    )
    assert set(outputs.keys()) == {3, 11}
    # On retry, B should not be in the call list (even though the FlakyPeerRPC
    # would now succeed, the excluded_peers set keeps it out).
    second_calls = rpc.calls[1:]  # skip the first call that failed
    assert all(p != "B" for p, _ in second_calls)


def test_retry_disabled_matches_phase5b_behavior():
    # With max_attempts=1, first failure bubbles up immediately.
    owners = {"B": {7}, "C": {7}}
    live = {7: {"B", "C"}}
    rpc = _FlakyPeerRPC(fail_once_for={"B"})
    with pytest.raises(ExpertRpcFailure):
        _run_test_fanout(
            owners=owners, ids_to_fan=[7], peer_rpc=rpc, live_owners=live,
            max_attempts=1,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_expert_retry_unit.py::test_retry_succeeds_on_second_attempt_to_replica -v`
Expected: AttributeError — `_phase_b_with_retry` does not exist.

- [ ] **Step 3: Refactor `run_split_layer` to factor out Phase B into `_phase_b_with_retry`**

In `src/model_shard/expert_orchestrator.py`, `run_split_layer` currently has an inline Phase B (fan-out + gather). Refactor to call a new method `_phase_b_with_retry`. Replace the Phase B block (roughly lines 355-385 — the section between the Phase A `mlx_guard` close and Phase C `mlx_guard` open) with:

```python
        # Phase B — peer fan-out with retry on peer failure. Lock is not held
        # here: peer threads need to acquire it to run their experts.
        outputs = self._phase_b_with_retry(
            post_attn=post_attn,
            all_ids=all_ids,
            layer_idx=layer_idx,
            request_id=request_id,
            initial_local_ids=local_ids,
            lm=lm,
        )
        outputs.update(local_outputs)  # merge local results back in
```

Move the fan-out + gather logic into the new method, wrapping it in a retry loop. The complete `_phase_b_with_retry` implementation:

```python
    def _phase_b_with_retry(
        self,
        post_attn: mx.array,
        all_ids: list[int],
        layer_idx: int,
        request_id: str,
        initial_local_ids: list[int],
        lm: Any,
    ) -> dict[int, mx.array]:
        """Run the peer fan-out with retries on ``ExpertRpcFailure``.

        Preserves partial outputs across retries: experts that already
        completed (in ``outputs``) are never re-dispatched. Each retry
        excludes peers that previously failed in THIS invocation.
        """
        import time as _time

        # ids we still need outputs for (local ids are handled by caller).
        remote_ids_needed = [e for e in all_ids if e not in initial_local_ids]

        # Initial routing.
        peer_loads = self.loads_provider()
        self_load = peer_loads.get(self.self_shard_id, 0)
        by_owner = group_expert_ids_by_owner_loaded(
            remote_ids_needed,
            owners=self.owners,
            peer_loads=peer_loads,
            self_shard_id=self.self_shard_id,
            self_load=self_load,
            rng=self.rng,
            live_owners_provider=self.live_owners_provider,
        )
        # Remove any ids the initial routing put on self (edge case — could
        # happen if live_owners_provider added self as an alternate).
        local_ids_extra = by_owner.pop(self.self_shard_id, [])
        outputs: dict[int, mx.array] = {}
        if local_ids_extra:
            with self._mlx_guard():
                outputs.update(
                    run_selected_experts(lm, post_attn, layer_idx, local_ids_extra)
                )

        futures: dict[str, Future[dict[int, mx.array]]] = {
            peer: self._executor.submit(
                self.peer_rpc.call, peer, request_id, layer_idx, ids, post_attn
            )
            for peer, ids in by_owner.items()
        }
        abort_events: dict[str, threading.Event] = {
            peer: threading.Event() for peer in futures
        }
        if abort_events:
            with self._in_flight_lock:
                self._in_flight[request_id] = abort_events

        excluded_peers: set[str] = set()
        attempts = 0
        try:
            while True:
                attempts += 1
                try:
                    self._gather_with_abort(
                        futures, abort_events, outputs, layer_idx
                    )
                    break
                except ExpertRpcFailure as exc:
                    if attempts >= self.retry_max_attempts:
                        raise
                    excluded_peers.add(exc.failed_peer)
                    # Sleep before the next retry (backoff).
                    backoff_idx = min(
                        attempts - 1, max(0, len(self.retry_backoff_ms) - 1)
                    )
                    if self.retry_backoff_ms:
                        _time.sleep(self.retry_backoff_ms[backoff_idx] / 1000.0)

                    # Figure out which experts still need results.
                    missing = [e for e in remote_ids_needed if e not in outputs]
                    if not missing:
                        break  # Every expert produced an output somehow — done.

                    # Re-route, excluding known-failed peers.
                    def _filtered_provider(eid: int) -> set[str]:
                        base = (
                            self.live_owners_provider(eid)
                            if self.live_owners_provider is not None
                            else set()
                        )
                        return base - excluded_peers

                    filtered_owners = {
                        sid: ids for sid, ids in self.owners.items()
                        if sid not in excluded_peers
                    }
                    by_owner_retry = group_expert_ids_by_owner_loaded(
                        missing,
                        owners=filtered_owners,
                        peer_loads=self.loads_provider(),
                        self_shard_id=self.self_shard_id,
                        self_load=self.loads_provider().get(
                            self.self_shard_id, 0
                        ),
                        rng=self.rng,
                        live_owners_provider=_filtered_provider,
                    )
                    # Experts routed to self on retry run locally under the lock.
                    local_retry = by_owner_retry.pop(self.self_shard_id, [])
                    if local_retry:
                        with self._mlx_guard():
                            outputs.update(
                                run_selected_experts(
                                    lm, post_attn, layer_idx, local_retry
                                )
                            )
                    # Re-submit futures for the remaining peers.
                    futures = {
                        peer: self._executor.submit(
                            self.peer_rpc.call,
                            peer, request_id, layer_idx, ids, post_attn,
                        )
                        for peer, ids in by_owner_retry.items()
                    }
                    abort_events = {
                        peer: threading.Event() for peer in futures
                    }
                    if abort_events:
                        with self._in_flight_lock:
                            self._in_flight[request_id] = abort_events
        finally:
            if abort_events:
                with self._in_flight_lock:
                    self._in_flight.pop(request_id, None)

        return outputs
```

Note the implementation imports `time as _time` inside the function — `time` isn't already used at module scope here; keeping the import local avoids polluting the module namespace. If the linter prefers a top-level import, move it.

- [ ] **Step 4: Update `run_split_layer` to use the new method**

In `run_split_layer`, ensure the Phase B section is replaced per Step 3's snippet. The existing `_gather_with_abort` method, `_in_flight` dict, and orchestrator fields are all used by the new method; no further structural change needed.

Also: the existing `outputs` dict in `run_split_layer` previously was pre-populated with `local_outputs` before fan-out. Preserve that semantic. The refactor moves fan-out into `_phase_b_with_retry` which returns a fresh `outputs` dict (remote results only), and the caller merges `local_outputs` back in:

Find the place in `run_split_layer` that does `outputs: dict[int, mx.array] = dict(local_outputs)`. Change to:

```python
        outputs: dict[int, mx.array] = dict(local_outputs)
        remote_outputs = self._phase_b_with_retry(
            post_attn=post_attn,
            all_ids=all_ids,
            layer_idx=layer_idx,
            request_id=request_id,
            initial_local_ids=local_ids,
            lm=lm,
        )
        outputs.update(remote_outputs)
```

- [ ] **Step 5: Run new tests**

Run: `uv run pytest tests/test_expert_retry_unit.py -v`
Expected: 9 PASS (2 from Task 1 + 2 from Task 2 + 5 new).

- [ ] **Step 6: Regression**

Run: `uv run pytest tests/test_expert_orchestrator.py tests/test_expert_orchestrator_observer.py tests/test_expert_orchestrator_timeout.py tests/test_expert_rpc_failure.py tests/test_expert_rpc_handler.py tests/test_expert_rpc_load_shift.py tests/test_tcp_peer_rpc.py -v -m "not slow"`
Expected: all PASS. The refactor preserves behavior when no failures occur (retry loop exits immediately on first success).

- [ ] **Step 7: Ruff + mypy clean**

```bash
uv run ruff check src/model_shard/expert_orchestrator.py tests/test_expert_retry_unit.py
uv run mypy src/model_shard/expert_orchestrator.py
```

- [ ] **Step 8: Commit**

```bash
git add src/model_shard/expert_orchestrator.py tests/test_expert_retry_unit.py
git commit -m "Phase 6-A Task 3: retry loop in run_split_layer Phase B"
```

---

### Task 4: Fast bit-exact correctness test

**Files:**
- Create: `tests/test_expert_retry_bit_exact.py`

This test asserts that the retry path produces the same output as the no-failure path. Uses a stub `peer_rpc` backed by `run_selected_experts` on a shared `lm`, so the "peers" produce the same tensors the real peers would — just with one controlled failure.

- [ ] **Step 1: Write the slow bit-exact test**

Create `tests/test_expert_retry_bit_exact.py`:

```python
"""Phase 6-A bit-exact correctness: retry output == no-failure output."""
from __future__ import annotations

import random
import threading

import mlx.core as mx
import pytest

from model_shard.expert_orchestrator import ExpertOrchestrator, ExpertRpcFailure
from model_shard.mlx_engine import load_model
from model_shard.moe import run_selected_experts

pytestmark = pytest.mark.slow

_HF_ID = "mlx-community/gemma-4-26b-a4b-it-4bit"
_LAYER = 15


@pytest.fixture(scope="module")
def lm():
    return load_model(_HF_ID)


class _SharedLmPeerRPC:
    """Test double: 'peer RPC' that just runs run_selected_experts on a
    shared lm — mimics what a real peer would produce. Supports one-shot
    failure injection on named peers."""

    def __init__(self, lm, fail_once_for: set[str] | None = None) -> None:
        self._lm = lm
        self._fail_once_for = set(fail_once_for or set())
        self._already_failed: set[str] = set()

    def call(
        self, peer_shard_id, request_id, layer_idx, expert_ids, h
    ):
        if (
            peer_shard_id in self._fail_once_for
            and peer_shard_id not in self._already_failed
        ):
            self._already_failed.add(peer_shard_id)
            raise RuntimeError(f"injected fail on {peer_shard_id}")
        return run_selected_experts(self._lm, h, layer_idx, list(expert_ids))


def test_retry_output_matches_no_failure_output(lm):
    # Setup: 3 "peers" B, C, D each "own" a disjoint subset of experts.
    # live_owners_provider reports B and E both own expert 3 (replica).
    owners = {
        "B": {3, 6},
        "C": {7, 10},
        "D": {11, 14},
        "E": {3},  # replica of 3 — retry target after B fails
    }

    def live_owners(eid: int) -> set[str]:
        return {sid for sid, ids in owners.items() if eid in ids}

    ids_to_fan = [3, 6, 7, 10, 11, 14]

    # Synthetic input: stays on no-sort path per 5a §7.5.
    mx.random.seed(7)
    hidden = lm.text_model.layers[_LAYER].pre_feedforward_layernorm_2.weight.shape[0]
    post_attn = mx.random.normal((1, 3, hidden)).astype(mx.bfloat16)

    # Baseline: no failure.
    rpc_nofail = _SharedLmPeerRPC(lm, fail_once_for=set())
    orch_nofail = ExpertOrchestrator(
        self_shard_id="self",
        owners=owners,
        peer_rpc=rpc_nofail,
        rpc_timeout_s=5.0,
        rng=random.Random(0),
        live_owners_provider=live_owners,
        retry_max_attempts=3,
        retry_backoff_ms=(0, 0),
    )
    baseline = orch_nofail._phase_b_with_retry(
        post_attn=post_attn,
        all_ids=ids_to_fan,
        layer_idx=_LAYER,
        request_id="r-base",
        initial_local_ids=[],
        lm=lm,
    )
    orch_nofail.close()

    # With-failure: peer B fails once on expert 3; retry lands on E.
    rpc_fail = _SharedLmPeerRPC(lm, fail_once_for={"B"})
    orch_fail = ExpertOrchestrator(
        self_shard_id="self",
        owners=owners,
        peer_rpc=rpc_fail,
        rpc_timeout_s=5.0,
        rng=random.Random(0),
        live_owners_provider=live_owners,
        retry_max_attempts=3,
        retry_backoff_ms=(0, 0),
    )
    with_fail = orch_fail._phase_b_with_retry(
        post_attn=post_attn,
        all_ids=ids_to_fan,
        layer_idx=_LAYER,
        request_id="r-fail",
        initial_local_ids=[],
        lm=lm,
    )
    orch_fail.close()

    # Bit-exact.
    assert set(baseline.keys()) == set(with_fail.keys()) == set(ids_to_fan)
    for eid in ids_to_fan:
        assert mx.array_equal(baseline[eid], with_fail[eid]).item(), (
            f"expert {eid} differs between no-failure and with-failure runs"
        )
```

- [ ] **Step 2: Run**

Run: `uv run pytest tests/test_expert_retry_bit_exact.py -v -m slow`
Expected: PASS. Model fixture load is ~30s cold, 5s warm. Test execution <1s.

- [ ] **Step 3: Commit**

```bash
git add tests/test_expert_retry_bit_exact.py
git commit -m "Phase 6-A Task 4: bit-exact retry correctness proof (slow)"
```

---

### Task 5: Slow E2E — kill replica mid-generation

**Files:**
- Create: `tests/test_expert_retry_e2e.py`

- [ ] **Step 1: Write the slow E2E test**

Before writing, reuse the fixture pattern from `tests/test_decode_hang_fix_e2e.py` (Phase 5b Task 22) — it already knows how to spin up 3 Node threads with a configurable shard map and kill one mid-decode.

Create `tests/test_expert_retry_e2e.py`:

```python
"""Phase 6-A E2E: kill a replica peer mid-generation, verify tokens continue.

The shard config overlaps experts 0-2 across two shards, so when one shard
dies the other still owns those experts. Retry should kick in and produce
correct tokens."""
from __future__ import annotations

import random
import socket as _sk
import threading
import time
from pathlib import Path

import pytest

from model_shard.client import Client
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


def test_retry_keeps_generation_alive_after_replica_death(monkeypatch):
    monkeypatch.setenv("ENABLE_GOSSIP", "true")
    monkeypatch.setenv("ENABLE_PARTIAL_LOAD", "false")
    monkeypatch.setenv("ENABLE_DYNAMIC_MIGRATION", "false")
    monkeypatch.setenv("ENABLE_EXPERT_RETRY", "true")
    monkeypatch.setenv("EXPERT_RETRY_BACKOFF_MS", "0,50")  # fast retry for testing

    # Use Phase 4's overlapping shard config. Experts 0, 1, 2 are replicated
    # across two shards each — so killing one shard leaves a replica alive.
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
    time.sleep(3.0)  # SWIM stabilize

    head_spec = next(s for s in specs if s.start_layer == 0)
    client = Client(head_address=head_spec.address)

    # Pick a non-head, non-tail shard to kill; it must hold a replicated expert.
    kill_idx = 1  # middle shard — always holds a replicated expert per Phase 4 config
    killed_sid = specs[kill_idx].shard_id

    # Start a long generation.
    errors: list[Exception] = []
    tokens_received: list[int] = []
    done = threading.Event()

    def drive():
        try:
            # Use a prompt that routes through the replicated expert set.
            tokens = client.generate(
                prompt_tokens=[1, 5674, 1],  # small valid token list
                max_new_tokens=2048,
            )
            tokens_received.extend(tokens)
        except Exception as e:
            errors.append(e)
        finally:
            done.set()

    t = threading.Thread(target=drive, daemon=True)
    t.start()
    time.sleep(1.0)  # let prefill + a couple of decode rounds happen

    # Kill the replica.
    nodes[kill_idx].shutdown()
    threads[kill_idx].join(timeout=3.0)

    # Wait for generation to either complete or error out.
    done.wait(timeout=30.0)

    # Clean up remaining nodes.
    for n, th in zip(nodes, threads):
        if th.is_alive():
            n.shutdown()
            th.join(timeout=3.0)

    # Expected outcome: the generation either completed successfully (retry
    # kept it alive — ideal) OR errored with SHARD_UNAVAILABLE after the
    # tail or head failed to forward (acceptable if the killed peer was
    # critical to a non-replicated path). The key test: the client did NOT
    # hang indefinitely.
    assert done.is_set(), (
        f"client hung after killing {killed_sid!r} — retry or cleanup failed"
    )
    # If retry succeeded, we should have received SOME tokens (at least the
    # ones produced before and after the kill).
    print(
        f"E2E retry result: tokens_received={len(tokens_received)}, "
        f"errors={errors}"
    )
```

The test is structured as a "did-it-hang" liveness check rather than an exact-token-count assertion, because the exact number of tokens produced depends on timing (how many decode rounds fit in 1s). The key invariant: **killing a replica-owning peer does not block the client indefinitely**. If retry works, tokens continue; if retry can't recover (e.g., killed peer was the head's downstream), the client gets `ERR_SHARD_UNAVAILABLE`. Both outcomes are acceptable for this test; the failure mode it guards against is the pre-6-A behavior where the client would hang.

- [ ] **Step 2: Run**

Run: `uv run pytest tests/test_expert_retry_e2e.py -v -m slow`
Expected: PASS. Test runs in ~10-15s (model load + 3s SWIM + 1s setup + up to 30s generation window, most of which is unused if retry succeeds quickly).

- [ ] **Step 3: Commit**

```bash
git add tests/test_expert_retry_e2e.py
git commit -m "Phase 6-A Task 5: slow E2E — retry keeps generation alive after replica death"
```

---

### Task 6: README + memory update + final verification

**Files:**
- Modify: `README.md`
- Modify: `/Users/lukechang/.claude/projects/-Users-lukechang-Github-model-shard/memory/project_gossip_moe.md`

- [ ] **Step 1: Add Phase 6-A status paragraph to README**

Insert after the Phase 5b status paragraph. Match the existing style (a single ~200-word paragraph, no emojis). Cover:

- Scope: expert-peer retry. When a peer fails mid-fan-out, local `ExpertOrchestrator` retries to an alternate replica (Phase 5b's `live_owners_provider`), preserving partial results.
- Gate: `ENABLE_EXPERT_RETRY=true` default. Env vars `EXPERT_RETRY_MAX_ATTEMPTS` (3), `EXPERT_RETRY_BACKOFF_MS` ("100,500").
- Decentralization preserved: retry is local to the node doing the fan-out; no central coordinator.
- Correctness proof: `tests/test_expert_retry_bit_exact.py` — retry output == no-failure output via `mx.array_equal`. Relies on 5b's bit-exactness-across-replicas property.
- E2E: `tests/test_expert_retry_e2e.py` kills a replica mid-generation and verifies the client doesn't hang (retry or clean failure; never indefinite block).
- Non-goals: pipeline-peer failure (separate sub-project); head-peer failure (client story); Byzantine (Phase 6-B).
- Link to spec: `docs/superpowers/specs/2026-04-17-phase6a-expert-retry-design.md`.

- [ ] **Step 2: Update memory file**

Add a Phase 6-A COMPLETE paragraph to `/Users/lukechang/.claude/projects/-Users-lukechang-Github-model-shard/memory/project_gossip_moe.md` parallel to the Phase 5b entry. Cover:

- Date `2026-04-17`, final commit SHA from the verification step.
- All 6 tasks done.
- Link to plan + spec.
- What it enables: "5b's replication now pays off." When a replica peer dies, every node's orchestrator retries locally to an alternate.
- Mechanism: retry loop in `run_split_layer` Phase B, excluded-peer set local to invocation, `live_owners_provider` filters exclusions, bit-exact against no-failure baseline.
- Gate: `ENABLE_EXPERT_RETRY=true` (default); `EXPERT_RETRY_MAX_ATTEMPTS` (3); `EXPERT_RETRY_BACKOFF_MS` ("100,500").
- Decomposition note: Phase 6 is three sub-projects (6-A retry, 6-B provenance, 6-C eviction). 6-A done. 6-B and 6-C remain — each needs its own brainstorm.
- R1-R5 risks from spec §6 acknowledged; mitigations in place for 6-A.

- [ ] **Step 3: Final verification sweep**

```bash
cd /Users/lukechang/Github/model_shard
uv run pytest -q                                                           # fast tests
uv run pytest -m slow -q tests/test_expert_retry_bit_exact.py             # bit-exact
uv run pytest -m slow -q tests/test_expert_retry_e2e.py                   # E2E
uv run pytest -m slow -q tests/test_migration_over_tcp.py                 # 5b regression
uv run ruff check src tests scripts
uv run mypy src
```
Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add README.md "/Users/lukechang/.claude/projects/-Users-lukechang-Github-model-shard/memory/project_gossip_moe.md"
git commit -m "Phase 6-A Task 6: README + memory update; plan complete"
```

---

## Self-Review Notes

**Spec coverage check** — every decision in the spec maps to a task:
- D1 (scope) → all tasks stay within expert-peer retry
- D2 (retries, not circuit-breaker) → Task 3 loop structure
- D3 (placement in Phase B) → Task 3
- D4 (partial-output preservation) → Task 3 (Step 3 code: `outputs` dict preserved across iterations)
- D5 (3 attempts, backoff) → Task 2 (defaults), Task 3 (loop)
- D6 (excluded-peer local to invocation) → Task 3 (`excluded_peers` set scope)
- D7 (API surface) → Task 1 (ExpertRpcFailure), Task 2 (fields)
- D8 (correctness bar bit-exact) → Task 4
- D9 (gate) → Task 2 (env vars)
- D10 (non-goals) → enforced by what's in-scope; nothing in the plan violates

**Placeholder scan:** No "TBD" or "add error handling" steps. Every code step has complete code.

**Type consistency:**
- `ExpertRpcFailure(msg, *, failed_peer, layer_idx)` — Task 1 defines, Task 3 consumes `.failed_peer` ✓
- `retry_max_attempts`, `retry_backoff_ms` — Task 2 defines, Task 3 consumes ✓
- `_phase_b_with_retry` — Task 3 defines, Tasks 3-4 call ✓
- `live_owners_provider` — unchanged from 5b
- `_filtered_provider` — closure local to `_phase_b_with_retry`

**No references to types/methods not defined in the plan.**
