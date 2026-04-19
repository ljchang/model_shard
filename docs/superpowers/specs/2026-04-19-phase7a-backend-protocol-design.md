# Phase 7-A — `Backend` Protocol + `MLXBackend` Refactor

**Status:** draft, 2026-04-19
**Scope:** Introduce a narrow `Backend` protocol that wraps every tensor-level operation the distributed engine performs. Refactor the existing MLX-specific code into `MLXBackend` that implements this protocol. `Node` and `ExpertOrchestrator` call `self._backend.X()` instead of direct module-level imports of `mlx_engine` / `moe` / `partial_load`. Single-platform behavioral no-op — every existing correctness proof (Phase 1-6 bit-exact Tier 1, all sub-project E2E tests) passes unchanged on the default MLX backend. First sub-project of Phase 7 (cross-platform port).

## 1. Background & Decisions

### 1.1 Why now

Phase 7 as originally scoped — "deploy on a real multi-machine cluster" — collides with a platform reality: the only MLX-supported hardware is Apple Silicon. Luke has an M5 plus a DGX Spark (Grace Hopper ARM + CUDA) plus CUDA fleet; a bit-exact-preserving cross-platform cluster requires either (a) sticking to Apple-Silicon-only and buying more Macs, or (b) porting the inference engine to a cross-platform runtime.

Research (see prior brainstorm) concluded:
- **PyTorch + HuggingFace Transformers** is the only engine that supports Mac (MPS) + NVIDIA (CUDA) in one codepath AND exposes per-layer / per-expert forward — the two hard requirements for model_shard's architecture.
- **Bit-exactness across MPS and CUDA is physically impossible** (different reduction kernels, different 4-bit dequant paths). Max achievable cross-platform: `allclose(rtol=1e-2)` + top-1 token agreement on greedy decode. Bit-exactness *within* a platform is still achievable.

Rather than a full MLX→PyTorch rewrite (6-10 weeks), adopt a **hybrid backend architecture**: abstract tensor ops behind a `Backend` protocol, implement `MLXBackend` first (preserving Apple-side bit-exactness and native MLX speed), then `PyTorchBackend` (Phase 7-B), then heterogeneous cluster (Phase 7-C).

Phase 7-A is the gate: introduce the abstraction without changing behavior. This spec covers only 7-A.

### 1.2 Decomposition of Phase 7

- **7-A (this spec).** `Backend` protocol + `MLXBackend`. Pure refactor; zero behavioral change. MLX deployments keep every bit-exact proof intact.
- **7-B (future).** `PyTorchBackend`. Ship CUDA support for DGX Spark / 3090s / RTX 6000 Pro. Verify sharded == atomic bit-exact within the PyTorch backend (CUDA-to-CUDA).
- **7-C (future).** Heterogeneous cluster (M5 MLX + DGX Spark PyTorch). New correctness bar: bit-exact within a backend, `allclose` + top-1 token equality across backends. New test tier.

Each has its own spec → plan → implementation cycle. 7-B and 7-C wait on 7-A landing cleanly.

### 1.3 Decisions

- **D1. Scope.** Behavioral no-op refactor. Zero new features. Zero MLX-code replacement — the existing `mlx_engine.py` / `moe.py` / `partial_load.py` modules stay; `MLXBackend` is a thin wrapper that delegates to them.

- **D2. `Backend` is a stateful class, not a module of free functions.** Each `Backend` instance owns a `LoadedModel` internally (`self._lm` in `MLXBackend`). Consumers call `backend.embed(ids)` rather than `embed(lm, ids)`. This is what enables `PyTorchBackend` later to own a `torch.nn.Module` + `torch.device` without exposing it.

- **D3. Opaque handle types.** `Activation`, `Cache`, `Mask`, `TopK` are `TypeVar`-bound in the protocol, concrete in each backend. Node doesn't inspect them; only passes them between `Backend` calls or serializes via `tensor_to_bytes`. Wire format (bytes + shape + wire-dtype int) remains backend-agnostic.

- **D4. Backend selection.** `Node.__init__` gains `backend: Backend | None = None`. When `None`, defaults to `MLXBackend()`. Env var `BACKEND=mlx` is the explicit form (future: `BACKEND=pytorch`). Legacy callers that supply `loaded_model=<LoadedModel>` are supported via an `MLXBackend.from_loaded_model(lm)` classmethod; this preserves test fixtures that inject a pre-loaded model.

- **D5. `_MLX_COMPUTE_LOCK` stays at the Node level.** The process-wide MLX lock protects MLX's default stream during in-process multi-node tests. `Backend` methods are thread-safe given the lock, but the lock itself remains a `Node` concern (not a backend concern) — rationale: a future `PyTorchBackend` has its own concurrency story, and wrapping the lock inside each backend would force us to name the MLX-specific concept in the protocol.

- **D6. Existing `mlx_engine`, `moe`, `partial_load` modules stay.** Do NOT duplicate logic. `MLXBackend` calls into these existing functions; the refactor is additive. Phase 7-B will NOT call into these modules (they're MLX-coupled). Phase 7-C or later may consider collapsing `mlx_engine` into `MLXBackend` if that becomes cleaner.

- **D7. `ExpertOrchestrator` gains a `backend` field.** Populated from `Node` during construction. Orchestrator's compute calls (`run_attention_and_route`, `run_shared_expert`, `run_selected_experts`, `aggregate_experts`) go through `self.backend`. The existing `heat_observer`, `live_owners_provider`, `peer_rpc` fields are unchanged.

- **D8. Wire dtype unchanged.** The existing `_DTYPE_MAP` + `_mx_to_wire_dtype` in `mlx_engine.py` (consumed by `Backend.dtype_to_wire`) handle all current dtypes. No new wire types in 7-A.

- **D9. Correctness bar: unchanged bit-exact.** Tier 1 prompts on a default-backend `Node` produce tokens byte-identical to the Phase 1 reference. Every existing correctness proof passes.

- **D10. Non-goals (explicit).**
  - No PyTorch / CUDA / ROCm code.
  - No cross-backend interop.
  - No changes to gossip, wire protocol, orchestrator logic, retry, provenance, or eviction.
  - No new config format (`config/shards.yaml` unchanged).
  - No new env vars that enable / disable features — `BACKEND=mlx` is the only new one, defaults preserve today's behavior.
  - No removal of `mlx_engine.py` / `moe.py` / `partial_load.py`.

## 2. Components

### 2.1 `src/model_shard/backends/base.py` (new)

The `Backend` protocol. Uses `typing.Protocol` for structural typing; concrete backends don't need to inherit.

```python
from __future__ import annotations
from typing import Any, Protocol, runtime_checkable


Activation = Any   # Opaque per-backend tensor (mx.array, torch.Tensor, ...).
Cache      = Any
Mask       = Any
TopK       = tuple[Activation, Activation]  # (top_k_indices, top_k_weights)


@runtime_checkable
class Backend(Protocol):
    """Tensor-level operations a Node / ExpertOrchestrator calls.

    Each Backend instance owns one LoadedModel. All methods are thread-
    safe provided the caller holds the Node's _MLX_COMPUTE_LOCK (or the
    backend's own equivalent serialization primitive, documented per
    backend)."""

    name: str  # "mlx" | "pytorch" | "executorch" ...

    # --- Loading ---------------------------------------------------------

    def load(self, hf_id: str) -> None: ...
    def load_partial(
        self, hf_id: str, held_experts_per_layer: dict[int, list[int]]
    ) -> None: ...
    def num_layers(self) -> int: ...
    def held_ids(self, layer_idx: int) -> tuple[int, ...]: ...
    def is_split_layer(self, layer_idx: int) -> bool: ...

    # --- Forward pass primitives -----------------------------------------

    def embed(self, token_ids: list[int]) -> Activation: ...
    def make_cache(self) -> Cache: ...
    def make_masks(self, h: Activation, cache: Cache) -> tuple[Mask, Mask]: ...
    def run_layer_atomic(
        self, layer_idx: int, h: Activation, cache: Cache,
        masks: tuple[Mask, Mask],
    ) -> Activation: ...
    def run_attention_and_route(
        self, layer_idx: int, h: Activation, cache: Cache,
        masks: tuple[Mask, Mask],
        heat_observer: Any = None,
    ) -> tuple[Activation, TopK]: ...
    def run_shared_expert(self, layer_idx: int, h: Activation) -> Activation: ...
    def run_selected_experts(
        self, layer_idx: int, h: Activation, expert_ids: list[int],
    ) -> dict[int, Activation]: ...
    def aggregate_experts(
        self, layer_idx: int,
        expert_outputs: dict[int, Activation],
        top_k_ids: list[int],
        top_k_weights: Activation,
        shared_out: Activation,
    ) -> Activation: ...
    def finalize(self, h: Activation) -> Activation: ...
    def argmax_last(self, logits: Activation) -> int: ...

    # --- Wire serialization ----------------------------------------------

    def tensor_to_bytes(self, h: Activation) -> bytes: ...
    def bytes_to_tensor(
        self, raw: bytes, shape: list[int], dtype: int,
    ) -> Activation: ...
    def dtype_to_wire(self, h: Activation) -> int: ...

    # --- Partial-load / migration ----------------------------------------

    def slice_expert(
        self, layer_idx: int, expert_id: int,
    ) -> list[bytes]: ...
    def attach_expert(
        self, layer_idx: int, expert_id: int, tensor_bytes: list[bytes],
    ) -> None: ...
    def detach_expert(
        self, layer_idx: int, expert_id: int,
    ) -> None: ...
```

### 2.2 `src/model_shard/backends/mlx_backend.py` (new)

```python
from __future__ import annotations
import threading
from typing import Any

import mlx.core as mx

from model_shard import mlx_engine, moe, partial_load


class MLXBackend:
    """MLX implementation of the Backend protocol. Wraps the existing
    `mlx_engine`, `moe`, `partial_load` modules. Zero logic change —
    every method is a thin delegation to the corresponding module-level
    function. Owns a single `LoadedModel` as `self._lm`."""

    name = "mlx"

    def __init__(self) -> None:
        self._lm: mlx_engine.LoadedModel | None = None

    @classmethod
    def from_loaded_model(
        cls, lm: mlx_engine.LoadedModel,
    ) -> "MLXBackend":
        """Construct an MLXBackend around an already-loaded model.
        Used by tests that inject a MagicMock or a real LoadedModel
        via the `loaded_model=` Node kwarg."""
        b = cls()
        b._lm = lm
        return b

    # --- Loading ---------------------------------------------------------

    def load(self, hf_id: str) -> None:
        self._lm = mlx_engine.load_model(hf_id)

    def load_partial(
        self, hf_id: str, held_experts_per_layer: dict[int, list[int]],
    ) -> None:
        self._lm = mlx_engine.load_model_partial(hf_id, held_experts_per_layer)

    def num_layers(self) -> int:
        return self._lm.num_layers

    def held_ids(self, layer_idx: int) -> tuple[int, ...]:
        return self._lm.held_ids_per_layer.get(layer_idx, ())

    def is_split_layer(self, layer_idx: int) -> bool:
        # In Phase 3/4/5 config, split layers are the keys of moe_experts.
        # But MLXBackend doesn't know about the shard's moe_experts —
        # that's a ShardSpec concern. Return False here; callers who care
        # about "is this layer split for THIS shard" should consult
        # the ShardSpec directly.
        return False  # See note; caller-side check preserved.

    # --- Forward pass primitives -----------------------------------------

    def embed(self, token_ids: list[int]) -> mx.array:
        return mlx_engine.embed_tokens(self._lm, mx.array([token_ids]))

    def make_cache(self) -> list[Any]:
        return mlx_engine.make_cache(self._lm)

    def make_masks(self, h, cache):
        return mlx_engine.make_masks(self._lm, h, cache)

    def run_layer_atomic(self, layer_idx, h, cache, masks):
        global_mask, sliding_mask = masks
        # Use the existing per-layer call pattern from run_layers.
        tm = self._lm.text_model
        layer = tm.layers[layer_idx]
        c = cache[tm.layer_idx_to_cache_idx[layer_idx]]
        mask = global_mask if layer.layer_type == "full_attention" else sliding_mask
        return layer(h, mask, c, per_layer_input=None)

    def run_attention_and_route(
        self, layer_idx, h, cache, masks, heat_observer=None,
    ):
        return moe.run_attention_and_route(
            self._lm, h, layer_idx, cache, masks, heat_observer=heat_observer,
        )

    def run_shared_expert(self, layer_idx, h):
        return moe.run_shared_expert(self._lm, h, layer_idx)

    def run_selected_experts(self, layer_idx, h, expert_ids):
        return moe.run_selected_experts(self._lm, h, layer_idx, expert_ids)

    def aggregate_experts(
        self, layer_idx, expert_outputs, top_k_ids, top_k_weights, shared_out,
    ):
        layer = self._lm.text_model.layers[layer_idx]
        return moe.aggregate_experts(
            expert_outputs, top_k_ids, top_k_weights, shared_out,
            layer.post_feedforward_layernorm_2,
        )

    def finalize(self, h):
        return mlx_engine.finalize(self._lm, h)

    def argmax_last(self, logits) -> int:
        return int(mx.argmax(logits[0, -1, :]).item())

    # --- Wire serialization ----------------------------------------------

    def tensor_to_bytes(self, h) -> bytes:
        return mlx_engine.tensor_to_bytes(h)

    def bytes_to_tensor(self, raw, shape, dtype):
        return mlx_engine.bytes_to_tensor(raw, shape, dtype)

    def dtype_to_wire(self, h) -> int:
        return mlx_engine._mx_to_wire_dtype(h.dtype)

    # --- Partial-load / migration ----------------------------------------

    def slice_expert(self, layer_idx, expert_id):
        # slice_expert returns a list[mx.array]; we convert to bytes for
        # wire-neutral passage between Node and Backend. Existing Phase 5b
        # call sites in Node pass mx.array list → convert via tensor_to_bytes
        # at the Node boundary. To preserve the existing semantics in 7-A,
        # return a list of mx.arrays; Node code expects that.
        # Note: if a future Backend abstraction wants bytes, revisit.
        # For 7-A: keep the list[mx.array] shape.
        lock = threading.Lock()  # Placeholder; Node passes its lock in production.
        return partial_load.slice_expert(self._lm, layer_idx, expert_id, lock)

    def attach_expert(self, layer_idx, expert_id, tensor_bytes):
        lock = threading.Lock()
        partial_load.attach_expert(self._lm, layer_idx, expert_id, tensor_bytes, lock)

    def detach_expert(self, layer_idx, expert_id):
        lock = threading.Lock()
        partial_load.detach_expert(self._lm, layer_idx, expert_id, lock)
```

**Design note on `slice_expert` / `attach_expert`:** these currently take a `threading.Lock` kwarg (from Phase 5b/6-C). The `Backend` protocol shouldn't expose backend-specific locks to callers. Two approaches:
1. **Pass the lock via backend state.** `Node` passes its `_MLX_COMPUTE_LOCK` to the backend at init; backend uses it internally. Cleaner.
2. **Backend has its own internal lock.** Create per instance. Fine for standalone use but doesn't serialize with other MLX compute on the same process.

Phase 7-A chooses **option 1**: `MLXBackend.__init__` optionally takes a `mlx_lock: threading.Lock | None = None`; if provided, it's used for `slice_expert`/`attach_expert`/`detach_expert`. If None, a backend-private lock is created. `Node` passes its `_MLX_COMPUTE_LOCK` via this path.

### 2.3 `src/model_shard/node.py` refactor

- `Node.__init__` accepts `backend: Backend | None = None`.
- If `backend is None`:
  - If `loaded_model is None` and partial-load path is eligible: construct `MLXBackend()` and call `backend.load_partial(hf_id, held)`.
  - Else if `loaded_model` is provided: `backend = MLXBackend.from_loaded_model(loaded_model)`.
  - Else: `backend = MLXBackend()` and call `backend.load(hf_id)`.
- `self._backend = backend`. The old `self._lm` is removed; anywhere it was used, replaced with `self._backend.<method>(...)`.
- `_MLX_COMPUTE_LOCK` is passed into the backend via `self._backend._mlx_lock = _MLX_COMPUTE_LOCK` or via an explicit setter. (If using the lock-init kwarg from D2.2, set it before returning from `Node.__init__`.)
- `ExpertOrchestrator` receives `backend=self._backend` in its constructor; use-sites in `run_split_layer` / `_phase_b_with_retry` call `self.backend.run_attention_and_route(...)` etc. instead of `moe.run_attention_and_route(...)`.

Specific Node methods affected (non-exhaustive; real list populated during Task 3):
- `_handle_begin` — `embed_tokens(self._lm, ...)` → `self._backend.embed(token_ids.tolist()[0])`.
- `_run_my_layers` — delegates to `mlx_engine.run_layers`; refactor `run_layers` itself to accept `backend` OR inline the loop in Node using `backend.run_layer_atomic` for non-split layers.
- `_handle_activation` → `finalize(self._lm, h)` → `self._backend.finalize(h)`; `mx.argmax(logits[0, -1, :]).item()` → `self._backend.argmax_last(logits)`.
- `_handle_expert_weight_request` — `slice_expert(self._lm, ...)` → `self._backend.slice_expert(...)`.
- `migration_attach` — `attach_expert(...)` → `self._backend.attach_expert(...)`.
- `migration_detach` — `detach_expert(...)` → `self._backend.detach_expert(...)`.
- `_forward_activation` / wire code — `tensor_to_bytes(h)` → `self._backend.tensor_to_bytes(h)`.

### 2.4 `src/model_shard/expert_orchestrator.py` refactor

- `ExpertOrchestrator` dataclass gains `backend: Backend | None = None` (keyword-only, default None for pre-Phase-7 consumers).
- `run_split_layer` Phase A:
  - `moe.run_attention_and_route(...)` → `self.backend.run_attention_and_route(...)`.
  - `moe.run_shared_expert(...)` → `self.backend.run_shared_expert(...)`.
  - `moe.run_selected_experts(...)` → `self.backend.run_selected_experts(...)`.
- `run_split_layer` Phase C:
  - `moe.aggregate_experts(...)` → `self.backend.aggregate_experts(...)`.

When `backend is None` (e.g., legacy unit tests), the orchestrator falls back to the module-level functions. **This fallback is temporary** — Phase 7-B will remove it.

### 2.5 `src/model_shard/mlx_engine.py` — minimal helper change

Add `run_layer_atomic(lm, layer_idx, h, cache, global_mask, sliding_mask) -> mx.array` as a module-level function that `MLXBackend.run_layer_atomic` delegates to. Preserves the single-source-of-truth pattern and makes the refactor auditable.

Also, expose `_mx_to_wire_dtype` as `mx_to_wire_dtype` (drop leading underscore) so `MLXBackend.dtype_to_wire` doesn't depend on a private name.

### 2.6 Tests

**New fast test:** `tests/test_backend_protocol.py`:
- `test_mlx_backend_implements_backend_protocol` — `isinstance(MLXBackend(), Backend)` via `runtime_checkable`.
- `test_mlx_backend_method_signatures` — construct a backend, verify each method exists and returns types of the right shape (where feasible).
- `test_mlx_backend_from_loaded_model` — construct via `from_loaded_model` with a MagicMock and verify internal `_lm` is set.

**Regression:** every existing fast + slow test passes. Update test fixtures as needed:
- Tests that pass `loaded_model=<mock>` to `Node` — still work because `MLXBackend.from_loaded_model(mock)` wraps them.
- Tests that construct `ExpertOrchestrator` directly without a `backend` — still work via the temporary `backend=None` fallback.

## 3. Wire Protocol

**No changes.** `Activation` bytes + `TensorDescriptor.{shape,dtype,byte_count}` + quant fields carry the same information. `MLXBackend.tensor_to_bytes` / `bytes_to_tensor` produce identical bytes to today's `mlx_engine.tensor_to_bytes` / `bytes_to_tensor` (both delegate to those functions).

## 4. Memory & Performance

Zero runtime overhead. Every `backend.X()` call is a method dispatch + a function call — same cost as today's direct module call. No extra tensor allocation, no re-serialization, no extra copies. Memory profile is identical.

## 5. Testing Strategy

### 5.1 Fast tests
- New: `tests/test_backend_protocol.py` (3 tests, see §2.6).
- All existing fast tests pass unchanged OR with a one-line `backend=MLXBackend()` construction update if they built `Node` with a specific model-injection pattern.

### 5.2 Slow tests
- **Tier 1 bit-exact** (`tests/test_tier1_tokens.py`, `tests/test_partial_load_tier1_e2e.py`) — tokens byte-identical to Phase 1 reference.
- **Every prior E2E proof** — `test_migration_over_tcp.py`, `test_expert_retry_bit_exact.py`, `test_provenance_tier1.py`, `test_provenance_rejection.py`, `test_eviction_e2e.py`, `test_eviction_race_with_expert_request.py` — all green on default `MLXBackend`.

### 5.3 Regression criterion
**If any existing test fails after the refactor, the refactor is wrong.** 7-A ships only when the entire existing test matrix is green.

## 6. Acceptance

1. `ruff check`, `mypy` clean.
2. Full fast suite green.
3. All slow E2E tests green (including the Phase 6 trilogy regression).
4. Tier 1 tokens bit-exact to Phase 1 reference via default `MLXBackend`.
5. README updated with a Phase 7-A status paragraph.
6. Memory file updated.

## 7. Risks & Mitigations

- **R1 — Scope creep into Phase 7-B.** Easy to start adding PyTorch stubs "while we're in here." Hard rule: 7-A is MLX-only. If a decision reveals itself to be PyTorch-specific, defer.
- **R2 — Subtle behavioral drift from added method dispatch.** Method-dispatch isn't free; MLX's lazy graph might get cut at different points if the backend introduces an extra function boundary. Mitigation: benchmark Tier 1 latency before + after. Acceptable delta: within 5% wall-clock (this is a constant-time overhead, not big-O).
- **R3 — Test fixtures with tight coupling to `Node._lm`.** Some tests may assert on `node._lm.held_ids_per_layer` directly. Fix: expose `node._backend.held_ids(L)` as the canonical accessor; update tests. Backward-compat shim: `node._lm` can remain as a `@property` that returns `self._backend._lm` for `MLXBackend` only (marked deprecated). Drop the shim in 7-B.
- **R4 — `ExpertOrchestrator.backend=None` fallback is a wart.** Purely temporary; 7-B removes it by requiring a backend. Document as a FIXME.
- **R5 — Thread-safety of `MLXBackend` slice/attach/detach lock parameter.** Node must pass its `_MLX_COMPUTE_LOCK` via the backend-init kwarg. If a test constructs `MLXBackend` without supplying the lock, a backend-private lock is used — this may mask real races in multi-threaded tests. Mitigation: log a WARNING when the backend-private lock is created.

## 8. References

- Phase 6-C spec: `docs/superpowers/specs/2026-04-18-phase6c-eviction-design.md` (current engine is fully MLX-native)
- `src/model_shard/mlx_engine.py` — the tensor ops being wrapped
- `src/model_shard/moe.py` — MoE primitives being wrapped
- `src/model_shard/partial_load.py` — `slice_expert` / `attach_expert` / `detach_expert`
- Prior brainstorm: cross-platform engine research (PyTorch recommendation)
