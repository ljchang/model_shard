# Phase 7-A Backend Protocol + MLXBackend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce a narrow `Backend` protocol that wraps every tensor-level operation; refactor the existing MLX code into an `MLXBackend` implementing it. Zero behavioral change — Tier 1 bit-exact + every Phase 1-6 correctness proof passes unchanged on the default `MLXBackend`.

**Architecture:** Stateful `Backend` class owns a `LoadedModel` internally; `Node` and `ExpertOrchestrator` call `self._backend.X()` instead of direct module imports. Opaque handle types (`Activation`, `Cache`, `Mask`, `TopK`) — consumers only pass them between Backend calls or serialize via `tensor_to_bytes`. `MLXBackend` delegates to the existing `mlx_engine` / `moe` / `partial_load` modules (no logic duplication).

**Tech Stack:** Python 3.13 with `typing.Protocol`, existing MLX (`mlx.core`), existing `mlx-vlm` for model loading.

**Spec:** `docs/superpowers/specs/2026-04-19-phase7a-backend-protocol-design.md` — decisions D1-D10.

---

## File Structure

**Create:**
- `src/model_shard/backends/__init__.py` — re-exports `Backend` + `MLXBackend`.
- `src/model_shard/backends/base.py` — `Backend` Protocol + opaque type aliases.
- `src/model_shard/backends/mlx_backend.py` — `MLXBackend` implementation (delegates to existing modules).
- `tests/test_backend_protocol.py` — fast tests for protocol conformance.
- `tests/test_mlx_backend.py` — fast unit tests for `MLXBackend` state handling (load, from_loaded_model, held_ids, etc., using mocked `LoadedModel`).

**Modify:**
- `src/model_shard/mlx_engine.py` — extract `run_layer_atomic(lm, layer_idx, h, cache, global_mask, sliding_mask) -> mx.array`; add public alias `mx_to_wire_dtype` (current `_mx_to_wire_dtype`).
- `src/model_shard/node.py` — `Node.__init__` accepts `backend: Backend | None = None`; legacy `loaded_model=` path routes through `MLXBackend.from_loaded_model`; method bodies call `self._backend.X()`.
- `src/model_shard/expert_orchestrator.py` — `ExpertOrchestrator` gains `backend: Backend | None = None` field; `run_split_layer` / `_phase_b_with_retry` prefer `self.backend.X()` over module-level `moe.X()`; temporary fallback when `backend is None`.

**Update at the end:**
- `README.md` — Phase 7-A status paragraph.
- `/Users/lukechang/.claude/projects/-Users-lukechang-Github-model-shard/memory/project_gossip_moe.md` — Phase 7-A COMPLETE entry.

---

## Task ordering

1. `Backend` protocol + opaque type aliases (`backends/base.py`).
2. `mlx_engine.py` helper additions (`run_layer_atomic` extraction + `mx_to_wire_dtype` public alias).
3. `MLXBackend` implementation (`backends/mlx_backend.py`) + fast tests.
4. `Node.__init__` refactor to accept `backend` + legacy `loaded_model=` compat.
5. `Node` method bodies refactor (every `mlx_engine.X(self._lm, ...)` → `self._backend.X(...)`).
6. `ExpertOrchestrator` refactor (`moe.X(...)` → `self.backend.X(...)`, with temporary `None` fallback).
7. Final verification sweep + README + memory update.

---

### Task 1: `Backend` protocol + opaque type aliases

**Files:**
- Create: `src/model_shard/backends/__init__.py`
- Create: `src/model_shard/backends/base.py`
- Test: `tests/test_backend_protocol.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_backend_protocol.py`:

```python
"""Phase 7-A: Backend protocol shape + runtime_checkable behavior."""
from __future__ import annotations

from model_shard.backends import Backend


def test_backend_is_runtime_checkable_protocol():
    """Protocol declared with @runtime_checkable; isinstance() works on it."""
    from typing import get_origin
    # The Backend object exposes _is_runtime_protocol internally when
    # decorated with @runtime_checkable.
    assert getattr(Backend, "_is_runtime_protocol", False) is True


def test_backend_declares_required_methods():
    """Protocol must declare every method Node/ExpertOrchestrator will call."""
    required = {
        "load", "load_partial", "num_layers", "held_ids", "is_split_layer",
        "embed", "make_cache", "make_masks",
        "run_layer_atomic", "run_attention_and_route",
        "run_shared_expert", "run_selected_experts",
        "aggregate_experts", "finalize", "argmax_last",
        "tensor_to_bytes", "bytes_to_tensor", "dtype_to_wire",
        "slice_expert", "attach_expert", "detach_expert",
    }
    # Protocol class exposes its methods via __annotations__ and __dict__.
    declared = {
        name for name in dir(Backend)
        if not name.startswith("_") and callable(getattr(Backend, name, None))
    }
    missing = required - declared
    assert not missing, f"Backend protocol missing methods: {missing}"


def test_backend_has_name_class_attr():
    """Backend declares `name: str` so consumers can log which backend is active."""
    assert "name" in getattr(Backend, "__annotations__", {})


def test_activation_cache_mask_topk_type_aliases_exist():
    """Opaque handle types for tensor/cache/mask/topk results."""
    from model_shard.backends.base import Activation, Cache, Mask, TopK
    # Just verify they are importable. Their actual typing is Any at runtime.
    _ = (Activation, Cache, Mask, TopK)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_backend_protocol.py -v`
Expected: ImportError — `model_shard.backends` does not exist.

- [ ] **Step 3: Create the backends package**

Create `src/model_shard/backends/__init__.py`:

```python
"""Backend protocol and implementations for Phase 7+ multi-backend support.

Phase 7-A ships the protocol and the MLXBackend. Phase 7-B/C add
PyTorchBackend and heterogeneous-cluster support.
"""

from model_shard.backends.base import (
    Activation,
    Backend,
    Cache,
    Mask,
    TopK,
)

__all__ = ["Activation", "Backend", "Cache", "Mask", "TopK"]
```

Create `src/model_shard/backends/base.py`:

```python
"""Phase 7-A Backend protocol.

Each Backend instance owns one LoadedModel-equivalent and exposes the
narrow tensor-level API the distributed engine calls. Consumers (Node,
ExpertOrchestrator) pass opaque handles between method calls and
serialize them at wire boundaries via tensor_to_bytes / bytes_to_tensor.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


# Opaque per-backend handle types. Typed as Any at runtime to keep the
# protocol structural; concrete backends use their native types
# (mx.array, torch.Tensor, etc.) internally.
Activation = Any
Cache = Any
Mask = Any
TopK = tuple[Activation, Activation]  # (top_k_indices, top_k_weights)


@runtime_checkable
class Backend(Protocol):
    """Tensor-level operations a Node / ExpertOrchestrator calls.

    Each Backend instance owns exactly one loaded model. All methods are
    thread-safe provided the caller holds the Node's _MLX_COMPUTE_LOCK
    (or the backend's own equivalent serialization primitive)."""

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
    ) -> list[Activation]: ...
    def attach_expert(
        self, layer_idx: int, expert_id: int, tensors: list[Activation],
    ) -> None: ...
    def detach_expert(
        self, layer_idx: int, expert_id: int,
    ) -> None: ...
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_backend_protocol.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Ruff + mypy clean**

```bash
uv run ruff check src/model_shard/backends tests/test_backend_protocol.py
uv run mypy src/model_shard/backends
```

- [ ] **Step 6: Commit**

```bash
git add src/model_shard/backends tests/test_backend_protocol.py
git commit -m "Phase 7-A Task 1: Backend protocol + opaque type aliases"
```

## Context

- **Working directory:** `/Users/lukechang/Github/model_shard`
- **Branch:** `main`
- **Predecessor commit:** `a3fa1d7` (Phase 7-A spec).
- **Plan file:** this file.
- **Spec:** `docs/superpowers/specs/2026-04-19-phase7a-backend-protocol-design.md` §2.1.

## Your Job

1. Follow Steps 1-6 exactly. TDD.
2. 4 new tests pass.
3. Ruff + mypy clean.
4. Commit with exact message.
5. Report back.

---

### Task 2: `mlx_engine.py` helper additions — `run_layer_atomic` + `mx_to_wire_dtype`

**Files:**
- Modify: `src/model_shard/mlx_engine.py`
- Test: `tests/test_mlx_engine_helpers.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_mlx_engine_helpers.py`:

```python
"""Phase 7-A Task 2: mlx_engine helper additions.

Verifies:
  * `run_layer_atomic` is a module-level function with the right signature.
  * `mx_to_wire_dtype` is a public alias for the pre-existing
    `_mx_to_wire_dtype`.
"""
from __future__ import annotations

import inspect

import mlx.core as mx

from model_shard import mlx_engine


def test_run_layer_atomic_exists_as_public_callable():
    assert callable(getattr(mlx_engine, "run_layer_atomic", None))


def test_run_layer_atomic_signature():
    sig = inspect.signature(mlx_engine.run_layer_atomic)
    params = list(sig.parameters.keys())
    # (lm, layer_idx, h, cache, global_mask, sliding_mask) — 6 positional.
    assert params == ["lm", "layer_idx", "h", "cache", "global_mask", "sliding_mask"]


def test_mx_to_wire_dtype_public_alias():
    # The underscore-prefixed original still exists.
    assert callable(getattr(mlx_engine, "_mx_to_wire_dtype", None))
    # The public alias exists and is the same object.
    assert callable(getattr(mlx_engine, "mx_to_wire_dtype", None))
    assert mlx_engine.mx_to_wire_dtype is mlx_engine._mx_to_wire_dtype


def test_mx_to_wire_dtype_returns_int_for_bfloat16():
    from model_shard._pb import wire_pb2
    assert mlx_engine.mx_to_wire_dtype(mx.bfloat16) == wire_pb2.DTYPE_BFLOAT16
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_mlx_engine_helpers.py -v`
Expected: AttributeError — `run_layer_atomic` and `mx_to_wire_dtype` don't exist yet.

- [ ] **Step 3: Add the two helpers in `src/model_shard/mlx_engine.py`**

Locate the existing `run_layers` function (around line 95-137). Extract the per-layer body into a new `run_layer_atomic` function, then refactor `run_layers` to call it.

Add `run_layer_atomic` right above `run_layers`:

```python
def run_layer_atomic(
    lm: LoadedModel,
    layer_idx: int,
    h: mx.array,
    cache: list[Any],
    global_mask: Any,
    sliding_mask: Any,
) -> mx.array:
    """Run one non-split decoder layer atomically.

    Extracts the inner body of ``run_layers`` for a single layer so a
    Backend can expose this as its own method. ``layer.layer_type`` picks
    the mask; ``cache[tm.layer_idx_to_cache_idx[layer_idx]]`` picks the
    per-layer cache slot."""
    tm = lm.text_model
    layer = tm.layers[layer_idx]
    c = cache[tm.layer_idx_to_cache_idx[layer_idx]]
    mask = global_mask if layer.layer_type == "full_attention" else sliding_mask
    return layer(h, mask, c, per_layer_input=None)  # type: ignore[no-any-return]
```

Inside `run_layers`, replace the else-branch body with a call to the new helper:

Find:
```python
        else:
            layer = tm.layers[i]
            c = cache[tm.layer_idx_to_cache_idx[i]]
            mask = global_mask if layer.layer_type == "full_attention" else sliding_mask
            h = layer(h, mask, c, per_layer_input=None)
```

Replace with:
```python
        else:
            h = run_layer_atomic(lm, i, h, cache, global_mask, sliding_mask)
```

Then add the public alias near the end of the file, after `_mx_to_wire_dtype`:

Find:
```python
def _mx_to_wire_dtype(dtype: mx.Dtype) -> int:
    for wire, (mxt, _, _) in _DTYPE_MAP.items():
        if mxt == dtype:
            return wire
    raise ValueError(f"unsupported mx dtype for wire: {dtype}")
```

Add right below it:
```python
# Phase 7-A: public alias so backends don't depend on a private name.
mx_to_wire_dtype = _mx_to_wire_dtype
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_mlx_engine_helpers.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Regression — ensure `run_layers` still works**

Run: `uv run pytest tests/test_mlx_engine.py tests/test_run_reference_script.py -v -m "not slow"`
Expected: all pass.

- [ ] **Step 6: Ruff + mypy clean**

```bash
uv run ruff check src/model_shard/mlx_engine.py tests/test_mlx_engine_helpers.py
uv run mypy src/model_shard/mlx_engine.py
```

- [ ] **Step 7: Commit**

```bash
git add src/model_shard/mlx_engine.py tests/test_mlx_engine_helpers.py
git commit -m "Phase 7-A Task 2: mlx_engine helpers — run_layer_atomic + mx_to_wire_dtype"
```

## Context

- **Predecessor commit:** Task 1.
- **Plan file:** this file.
- **Spec:** §2.5 specifies these two additions.
- **Purpose:** `MLXBackend.run_layer_atomic` (Task 3) delegates to this; `MLXBackend.dtype_to_wire` delegates to the public alias. Keeping the helpers in `mlx_engine.py` preserves the single-source-of-truth pattern.

## Your Job

1. Follow Steps 1-7 exactly. TDD.
2. Tests pass; regression green.
3. Ruff + mypy clean.
4. Commit.
5. Report back.

---

### Task 3: `MLXBackend` implementation

**Files:**
- Create: `src/model_shard/backends/mlx_backend.py`
- Modify: `src/model_shard/backends/__init__.py` (export `MLXBackend`)
- Test: `tests/test_mlx_backend.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mlx_backend.py`:

```python
"""Phase 7-A Task 3: MLXBackend state handling + protocol conformance."""
from __future__ import annotations

import threading
import types
from unittest.mock import MagicMock

import mlx.core as mx
import pytest

from model_shard.backends import Backend, MLXBackend


def test_mlx_backend_implements_backend_protocol():
    """runtime_checkable Protocol check at instance level."""
    b = MLXBackend()
    assert isinstance(b, Backend)


def test_mlx_backend_name_is_mlx():
    assert MLXBackend.name == "mlx"


def test_mlx_backend_from_loaded_model_wraps_existing_lm():
    """MLXBackend.from_loaded_model(lm) is the test-fixture escape hatch
    that lets callers inject a pre-loaded (or mocked) LoadedModel."""
    lm = MagicMock()
    lm.num_layers = 30
    b = MLXBackend.from_loaded_model(lm)
    assert b._lm is lm
    assert b.num_layers() == 30


def test_mlx_backend_held_ids_delegates_to_lm():
    lm = MagicMock()
    lm.held_ids_per_layer = {15: (0, 3, 6)}
    b = MLXBackend.from_loaded_model(lm)
    assert b.held_ids(15) == (0, 3, 6)
    assert b.held_ids(99) == ()  # absent layer → empty tuple


def test_mlx_backend_is_split_layer_returns_false_by_default():
    """MLXBackend itself doesn't know which layers are split — that's a
    ShardSpec concern. Always returns False; callers consult ShardSpec."""
    lm = MagicMock()
    b = MLXBackend.from_loaded_model(lm)
    assert b.is_split_layer(0) is False
    assert b.is_split_layer(15) is False


def test_mlx_backend_tensor_to_bytes_roundtrips_bfloat16():
    b = MLXBackend()  # No model needed for this method.
    tensor = mx.full((2, 4), 1.5, dtype=mx.bfloat16)
    raw = b.tensor_to_bytes(tensor)
    recovered = b.bytes_to_tensor(raw, shape=[2, 4], dtype=b.dtype_to_wire(tensor))
    assert mx.array_equal(recovered, tensor).item()


def test_mlx_backend_argmax_last_returns_int():
    b = MLXBackend()
    logits = mx.array([[[1.0, 2.0, 3.0]]], dtype=mx.float32)
    assert b.argmax_last(logits) == 2


def test_mlx_backend_accepts_optional_lock():
    """MLXBackend(mlx_lock=existing_lock) uses the caller's lock for
    slice/attach/detach serialization. Default: backend-private lock."""
    lock = threading.Lock()
    b = MLXBackend(mlx_lock=lock)
    assert b._mlx_lock is lock


def test_mlx_backend_creates_private_lock_when_none():
    b = MLXBackend()
    assert isinstance(b._mlx_lock, type(threading.Lock()))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_mlx_backend.py -v`
Expected: ImportError — `MLXBackend` not exported yet.

- [ ] **Step 3: Create `src/model_shard/backends/mlx_backend.py`**

```python
"""Phase 7-A MLXBackend: implementation of the Backend protocol over
the existing mlx_engine / moe / partial_load modules. Thin delegation
layer — zero logic duplication."""

from __future__ import annotations

import threading
from typing import Any

import mlx.core as mx

from model_shard import mlx_engine, moe, partial_load


class MLXBackend:
    """MLX implementation of the Backend protocol.

    Each instance owns one ``LoadedModel`` as ``self._lm``. The optional
    ``mlx_lock`` is used to serialize ``slice_expert`` / ``attach_expert``
    / ``detach_expert`` with concurrent MLX compute (Node passes its
    process-wide ``_MLX_COMPUTE_LOCK`` here in production; unit tests may
    leave it unset and a backend-private lock is created).
    """

    name: str = "mlx"

    def __init__(self, mlx_lock: threading.Lock | None = None) -> None:
        self._lm: mlx_engine.LoadedModel | None = None
        self._mlx_lock: threading.Lock = mlx_lock or threading.Lock()

    @classmethod
    def from_loaded_model(
        cls, lm: mlx_engine.LoadedModel,
        mlx_lock: threading.Lock | None = None,
    ) -> "MLXBackend":
        """Construct an MLXBackend wrapping an existing LoadedModel.
        Used by tests that inject a MagicMock or a real LoadedModel via
        the ``loaded_model=`` Node kwarg."""
        b = cls(mlx_lock=mlx_lock)
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
        assert self._lm is not None
        return int(self._lm.num_layers)

    def held_ids(self, layer_idx: int) -> tuple[int, ...]:
        assert self._lm is not None
        return self._lm.held_ids_per_layer.get(layer_idx, ())

    def is_split_layer(self, layer_idx: int) -> bool:
        # MLXBackend doesn't know which layers are split for a given shard.
        # Phase 7-A: always False; callers consult ShardSpec.moe_experts.
        return False

    # --- Forward pass primitives -----------------------------------------

    def embed(self, token_ids: list[int]) -> mx.array:
        assert self._lm is not None
        return mlx_engine.embed_tokens(self._lm, mx.array([token_ids]))

    def make_cache(self) -> list[Any]:
        assert self._lm is not None
        return mlx_engine.make_cache(self._lm)

    def make_masks(
        self, h: mx.array, cache: list[Any],
    ) -> tuple[Any, Any]:
        assert self._lm is not None
        return mlx_engine.make_masks(self._lm, h, cache)

    def run_layer_atomic(
        self, layer_idx: int, h: mx.array, cache: list[Any],
        masks: tuple[Any, Any],
    ) -> mx.array:
        assert self._lm is not None
        global_mask, sliding_mask = masks
        return mlx_engine.run_layer_atomic(
            self._lm, layer_idx, h, cache, global_mask, sliding_mask,
        )

    def run_attention_and_route(
        self, layer_idx: int, h: mx.array, cache: list[Any],
        masks: tuple[Any, Any], heat_observer: Any = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        assert self._lm is not None
        post_attn, top_k_ids, top_k_weights = moe.run_attention_and_route(
            self._lm, h, layer_idx, cache, masks, heat_observer=heat_observer,
        )
        return post_attn, (top_k_ids, top_k_weights)

    def run_shared_expert(self, layer_idx: int, h: mx.array) -> mx.array:
        assert self._lm is not None
        return moe.run_shared_expert(self._lm, h, layer_idx)

    def run_selected_experts(
        self, layer_idx: int, h: mx.array, expert_ids: list[int],
    ) -> dict[int, mx.array]:
        assert self._lm is not None
        return moe.run_selected_experts(self._lm, h, layer_idx, expert_ids)

    def aggregate_experts(
        self, layer_idx: int,
        expert_outputs: dict[int, mx.array],
        top_k_ids: list[int],
        top_k_weights: mx.array,
        shared_out: mx.array,
    ) -> mx.array:
        assert self._lm is not None
        layer = self._lm.text_model.layers[layer_idx]
        return moe.aggregate_experts(
            expert_outputs, top_k_ids, top_k_weights, shared_out,
            layer.post_feedforward_layernorm_2,
        )

    def finalize(self, h: mx.array) -> mx.array:
        assert self._lm is not None
        return mlx_engine.finalize(self._lm, h)

    def argmax_last(self, logits: mx.array) -> int:
        return int(mx.argmax(logits[0, -1, :]).item())

    # --- Wire serialization ----------------------------------------------

    def tensor_to_bytes(self, h: mx.array) -> bytes:
        return mlx_engine.tensor_to_bytes(h)

    def bytes_to_tensor(
        self, raw: bytes, shape: list[int], dtype: int,
    ) -> mx.array:
        return mlx_engine.bytes_to_tensor(raw, shape, dtype)

    def dtype_to_wire(self, h: mx.array) -> int:
        return mlx_engine.mx_to_wire_dtype(h.dtype)

    # --- Partial-load / migration ----------------------------------------

    def slice_expert(
        self, layer_idx: int, expert_id: int,
    ) -> list[mx.array]:
        assert self._lm is not None
        return partial_load.slice_expert(
            self._lm, layer_idx, expert_id, self._mlx_lock,
        )

    def attach_expert(
        self, layer_idx: int, expert_id: int, tensors: list[mx.array],
    ) -> None:
        assert self._lm is not None
        partial_load.attach_expert(
            self._lm, layer_idx, expert_id, tensors, self._mlx_lock,
        )

    def detach_expert(self, layer_idx: int, expert_id: int) -> None:
        assert self._lm is not None
        partial_load.detach_expert(
            self._lm, layer_idx, expert_id, self._mlx_lock,
        )


__all__ = ["MLXBackend"]
```

Update `src/model_shard/backends/__init__.py` to re-export `MLXBackend`:

```python
"""Backend protocol and implementations for Phase 7+ multi-backend support.

Phase 7-A ships the protocol and the MLXBackend. Phase 7-B/C add
PyTorchBackend and heterogeneous-cluster support.
"""

from model_shard.backends.base import (
    Activation,
    Backend,
    Cache,
    Mask,
    TopK,
)
from model_shard.backends.mlx_backend import MLXBackend

__all__ = [
    "Activation",
    "Backend",
    "Cache",
    "MLXBackend",
    "Mask",
    "TopK",
]
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_mlx_backend.py tests/test_backend_protocol.py -v`
Expected: 9 PASS (4 from Task 1 + 9 from Task 3, minus any overlap — actually 4 + 9 = 13 total).

Wait: test count is 4 + 9 = 13 total. Expected: **13 PASS**.

- [ ] **Step 5: Ruff + mypy clean**

```bash
uv run ruff check src/model_shard/backends tests/test_mlx_backend.py
uv run mypy src/model_shard/backends
```

- [ ] **Step 6: Commit**

```bash
git add src/model_shard/backends/ tests/test_mlx_backend.py
git commit -m "Phase 7-A Task 3: MLXBackend implementation (thin delegation wrapper)"
```

## Context

- **Predecessor commit:** Task 2.
- **Plan file:** this file.
- **Spec:** §2.2 specifies the MLXBackend shape.
- **Design note:** D2 / D5 / D6 — stateful class, delegates to existing modules, takes optional `mlx_lock` for slice/attach/detach serialization.

## Your Job

1. Follow Steps 1-6 exactly. TDD.
2. All 13 tests pass (4 Task 1 + 9 Task 3).
3. Ruff + mypy clean.
4. Commit.
5. Report back.

---

### Task 4: `Node.__init__` refactor — backend kwarg + legacy compat

**Files:**
- Modify: `src/model_shard/node.py`
- Test: `tests/test_node_backend_wiring.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_node_backend_wiring.py`:

```python
"""Phase 7-A Task 4: Node accepts a `backend` kwarg."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from model_shard.backends import Backend, MLXBackend
from model_shard.node import Node
from model_shard.shard_map import NodeAddress, ShardMap, ShardSpec


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("ENABLE_GOSSIP", "false")
    monkeypatch.setenv("ENABLE_PARTIAL_LOAD", "false")
    monkeypatch.setenv("ENABLE_DYNAMIC_MIGRATION", "false")
    monkeypatch.setenv("ENABLE_EXPERT_RETRY", "false")
    yield


def _mk_spec(sid: str, port: int) -> ShardSpec:
    return ShardSpec(
        shard_id=sid,
        address=NodeAddress(host="127.0.0.1", port=port),
        start_layer=0, end_layer=30, moe_experts={},
    )


def test_node_default_backend_is_mlx_backend_wrapping_loaded_model():
    """Legacy path: Node(loaded_model=<mock>) wraps it in MLXBackend.from_loaded_model."""
    spec_a = _mk_spec("A", 32000)
    spec_b = _mk_spec("B", 32001)
    sm = ShardMap({"A": spec_a, "B": spec_b})
    lm = MagicMock()
    n = Node(shard=spec_a, shard_map=sm, loaded_model=lm, total_layers=30)
    assert isinstance(n._backend, Backend)
    assert n._backend.name == "mlx"
    assert n._backend._lm is lm  # MLXBackend holds the lm internally


def test_node_accepts_explicit_backend():
    """Explicit backend kwarg is honored over loaded_model."""
    spec_a = _mk_spec("A", 32002)
    spec_b = _mk_spec("B", 32003)
    sm = ShardMap({"A": spec_a, "B": spec_b})
    lm = MagicMock()
    b = MLXBackend.from_loaded_model(lm)
    n = Node(shard=spec_a, shard_map=sm, backend=b, total_layers=30)
    assert n._backend is b


def test_node_lm_property_is_backend_lm_for_backcompat():
    """Pre-Phase-7 code that reads node._lm still works; it's a deprecated
    shim that returns the backend's loaded model. Only valid for MLXBackend."""
    spec_a = _mk_spec("A", 32004)
    spec_b = _mk_spec("B", 32005)
    sm = ShardMap({"A": spec_a, "B": spec_b})
    lm = MagicMock()
    n = Node(shard=spec_a, shard_map=sm, loaded_model=lm, total_layers=30)
    assert n._lm is lm  # property returns the underlying LoadedModel


def test_node_passes_mlx_lock_into_backend():
    """Node's _MLX_COMPUTE_LOCK is passed into MLXBackend.__init__ so
    slice/attach/detach serialize against concurrent compute."""
    from model_shard.node import _MLX_COMPUTE_LOCK
    spec_a = _mk_spec("A", 32006)
    spec_b = _mk_spec("B", 32007)
    sm = ShardMap({"A": spec_a, "B": spec_b})
    lm = MagicMock()
    n = Node(shard=spec_a, shard_map=sm, loaded_model=lm, total_layers=30)
    # Either the Node set the backend's lock to _MLX_COMPUTE_LOCK at init,
    # or the backend's lock is a private one. Prefer the former.
    assert n._backend._mlx_lock is _MLX_COMPUTE_LOCK
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_node_backend_wiring.py -v`
Expected: AttributeError — `Node._backend` doesn't exist.

- [ ] **Step 3: Refactor `Node.__init__`**

In `src/model_shard/node.py`:

Add to the imports near the top:
```python
from model_shard.backends import Backend, MLXBackend
```

Locate `Node.__init__`. Today it accepts `shard`, `shard_map`, `loaded_model`, `total_layers` (and possibly others). Extend the signature to accept `backend: Backend | None = None` before `total_layers`:

```python
    def __init__(
        self,
        shard: ShardSpec,
        shard_map: ShardMap,
        loaded_model: Any | None = None,
        backend: Backend | None = None,
        total_layers: int = 0,
    ) -> None:
```

Inside `__init__`, replace the existing code that computes `self._lm` with the following backend-construction block. This preserves all prior behavior (including the Phase 5a `ENABLE_PARTIAL_LOAD` path):

```python
        # Phase 7-A: Backend construction.
        # Precedence:
        #   1. Explicit `backend` kwarg wins.
        #   2. Else, if `loaded_model` was passed, wrap it in MLXBackend.
        #   3. Else, construct MLXBackend and call .load() / .load_partial().
        if backend is not None:
            self._backend: Backend = backend
        elif loaded_model is not None:
            self._backend = MLXBackend.from_loaded_model(
                loaded_model, mlx_lock=_MLX_COMPUTE_LOCK
            )
        else:
            b = MLXBackend(mlx_lock=_MLX_COMPUTE_LOCK)
            if _partial_load_enabled() and shard.moe_experts:
                held = {L: list(ids) for L, ids in shard.moe_experts.items()}
                b.load_partial("mlx-community/gemma-4-26b-a4b-it-4bit", held)
            else:
                b.load("mlx-community/gemma-4-26b-a4b-it-4bit")
            self._backend = b
```

Add the deprecated back-compat property for `_lm`. Near the class-body end (before any methods), add:

```python
    @property
    def _lm(self) -> Any:
        """Deprecated: use self._backend methods directly. Kept for
        Phase 1-6 callers that read node._lm as a LoadedModel. Only
        meaningful for MLXBackend; other backends return whatever they
        hold internally (may be a torch.nn.Module etc.)."""
        return getattr(self._backend, "_lm", None)
```

Delete any old `self._lm = ...` assignment elsewhere in `__init__`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_node_backend_wiring.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Regression on existing Node construction**

Run: `uv run pytest tests/test_node_membership.py tests/test_node_load_wiring.py tests/test_node_partial_load_wiring.py tests/test_node_live_experts.py tests/test_decode_hang_fix.py tests/test_dynamic_migration_gate.py tests/test_node_eviction.py tests/test_node_backend_wiring.py tests/test_handle_expert_request_authority.py -v -m "not slow"`
Expected: all pass. Legacy `loaded_model=` callers work via `MLXBackend.from_loaded_model`.

- [ ] **Step 6: Ruff + mypy clean**

```bash
uv run ruff check src/model_shard/node.py tests/test_node_backend_wiring.py
uv run mypy src/model_shard/node.py
```

- [ ] **Step 7: Commit**

```bash
git add src/model_shard/node.py tests/test_node_backend_wiring.py
git commit -m "Phase 7-A Task 4: Node.__init__ accepts backend kwarg + legacy _lm compat"
```

## Context

- **Predecessor commit:** Task 3.
- **Plan file:** this file.
- **Spec:** §2.3, §D4.
- **Important:** Tasks 5 and 6 refactor method bodies to call `self._backend.X()`. This task only changes `__init__`; `self._lm` property is temporary back-compat glue.

## Your Job

1. Follow Steps 1-7 exactly. TDD.
2. All new tests pass; regression green.
3. Ruff + mypy clean.
4. Commit.
5. Report back.

---

### Task 5: `Node` method bodies refactor — route through `self._backend`

**Files:**
- Modify: `src/model_shard/node.py`

- [ ] **Step 1: Identify every call site**

Run a survey grep to find every place `Node` calls into `mlx_engine` / `moe` / `partial_load` directly:

```bash
cd /Users/lukechang/Github/model_shard
grep -n "mlx_engine\." src/model_shard/node.py
grep -n "moe\." src/model_shard/node.py
grep -n "partial_load\." src/model_shard/node.py
grep -nE "\bembed_tokens\(|\bfinalize\(|\btensor_to_bytes\(|\bbytes_to_tensor\(|\bmake_cache\(|\bmake_masks\(|\brun_layers\(|\bslice_expert\(|\battach_expert\(|\bdetach_expert\(|\bmx_to_wire_dtype\(|\b_mx_to_wire_dtype\(" src/model_shard/node.py
```

Expected sites (approximate, verify against actual code):
- `_handle_begin`: `embed_tokens(self._lm, token_ids)` → `self._backend.embed(token_ids.tolist()[0])`  *(Caveat: the current code passes `mx.array([prompt_tokens])` of shape `[1, L]`; the backend's `embed` takes a `list[int]` and wraps internally. Preserve shape via whichever interface is easier.)*
- `_handle_begin`: `make_cache(self._lm)` → `self._backend.make_cache()`
- `_handle_begin` / `_drive_decode_loop`: `embed_tokens(self._lm, mx.array([[token_id]]))` → keep MLX-native for simplicity; see Step 2 workaround.
- `_handle_activation`: tail branch `finalize(self._lm, h)` → `self._backend.finalize(h)`
- `_handle_activation`: `mx.argmax(logits[0, -1, :]).item()` → `self._backend.argmax_last(logits)`
- `_forward_activation` / `_activation_envelope`: `tensor_to_bytes(h)` → `self._backend.tensor_to_bytes(h)`
- `_handle_activation`: `bytes_to_tensor(tensor_bytes, shape=..., dtype=...)` → `self._backend.bytes_to_tensor(raw, shape, dtype)`
- `_handle_expert_request` / `_handle_expert_weight_request`: same two byte conversions.
- `_handle_expert_request`: `run_selected_experts(self._lm, h, layer_idx, requested)` → `self._backend.run_selected_experts(layer_idx, h, requested)`
- `_handle_expert_weight_request`: `slice_expert(self._lm, layer_idx, expert_id, _MLX_COMPUTE_LOCK)` → `self._backend.slice_expert(layer_idx, expert_id)`
- `_dtype_to_wire(t.dtype)` / `_mx_to_wire_dtype(...)` call sites in node.py → `self._backend.dtype_to_wire(t)` (note the signature change: takes the tensor, not the dtype).
- `migration_attach`: `attach_expert(self._lm, ..., _MLX_COMPUTE_LOCK)` → `self._backend.attach_expert(layer_idx, expert_id, tensors)`
- `migration_detach`: `detach_expert(self._lm, ..., _MLX_COMPUTE_LOCK)` → `self._backend.detach_expert(layer_idx, expert_id)`

Also `_run_my_layers` currently calls `run_layers(self._lm, ...)`. `run_layers` (modified in Task 2 to accept `provenance_chain` + `node_id`, and to split between atomic and split layers) is still MLX-specific. **Do not route `run_layers` through the backend in Phase 7-A.** Keep `_run_my_layers` calling `run_layers` directly; pass `self._backend` is not needed for the atomic path because `run_layers` itself already has `run_layer_atomic`. The orchestrator fan-out (split layers) does go through the backend (Task 6).

- [ ] **Step 2: Pick a minimal-change approach for `embed`**

The current `_handle_begin` does:
```python
token_ids = mx.array([prompt_tokens])
h = embed_tokens(self._lm, token_ids)
```

And `_drive_decode_loop` does:
```python
h = embed_tokens(self._lm, mx.array([[token_id]]))
```

`Backend.embed(token_ids: list[int])` takes a Python list and wraps internally into a `[1, L]` shape. This matches `_handle_begin`'s first call but needs adaptation in `_drive_decode_loop`.

**Approach:** update both sites to call `self._backend.embed([<ids>])`:
- `_handle_begin`: `h = self._backend.embed(list(prompt_tokens))`
- `_drive_decode_loop`: `h = self._backend.embed([token_id])`

- [ ] **Step 3: Apply the refactor**

Go through the list from Step 1 methodically. At every call site, replace the direct function call with the backend method. Some specific rewrites:

In `_handle_begin`, find:
```python
        cache = make_cache(self._lm)
        ...
        prompt_tokens = list(req.prompt_token_ids)
        token_ids = mx.array([prompt_tokens])
        h = embed_tokens(self._lm, token_ids)
```

Replace with:
```python
        cache = self._backend.make_cache()
        ...
        prompt_tokens = list(req.prompt_token_ids)
        h = self._backend.embed(prompt_tokens)
```

In `_drive_decode_loop`, find:
```python
                h = embed_tokens(self._lm, mx.array([[token_id]]))
```

Replace with:
```python
                h = self._backend.embed([token_id])
```

In `_handle_activation`, find the tail-path finalize:
```python
        if self.is_tail:
            logits = finalize(self._lm, h)
            token_id = int(mx.argmax(logits[0, -1, :]).item())
```

Replace with:
```python
        if self.is_tail:
            logits = self._backend.finalize(h)
            token_id = self._backend.argmax_last(logits)
```

In `_activation_envelope` (module-level helper) — the function takes `h` but doesn't have access to a backend. Since it's module-level and consumed by `Node._forward_activation`, change `_activation_envelope` to also take the backend:

Find:
```python
def _activation_envelope(
    request_id: str, next_layer: int, h: mx.array
) -> tuple[wire_pb2.Envelope, bytes]:
    raw = tensor_to_bytes(h)
    env = wire_pb2.Envelope()
    env.activation.protocol_version = _PROTOCOL_VERSION
    env.activation.request_id = request_id
    env.activation.next_layer_idx = next_layer
    env.activation.tensor.shape.extend(list(h.shape))
    env.activation.tensor.dtype = _dtype_to_wire(h.dtype)
    env.activation.tensor.quant = wire_pb2.QUANT_NONE
    env.activation.tensor.byte_count = len(raw)
    return env, raw
```

Replace with:
```python
def _activation_envelope(
    request_id: str, next_layer: int, h: Any, backend: Backend,
) -> tuple[wire_pb2.Envelope, bytes]:
    raw = backend.tensor_to_bytes(h)
    env = wire_pb2.Envelope()
    env.activation.protocol_version = _PROTOCOL_VERSION
    env.activation.request_id = request_id
    env.activation.next_layer_idx = next_layer
    env.activation.tensor.shape.extend(list(h.shape))
    env.activation.tensor.dtype = backend.dtype_to_wire(h)
    env.activation.tensor.quant = wire_pb2.QUANT_NONE
    env.activation.tensor.byte_count = len(raw)
    return env, raw
```

Update the one call site in `Node._forward_activation`:
```python
        env, raw = _activation_envelope(
            request_id, self._shard.end_layer, h, self._backend,
        )
```

In `_handle_activation`, find:
```python
        h = bytes_to_tensor(
            tensor_bytes, shape=list(act.tensor.shape), dtype=act.tensor.dtype
        )
```

Replace with:
```python
        h = self._backend.bytes_to_tensor(
            tensor_bytes, shape=list(act.tensor.shape), dtype=int(act.tensor.dtype),
        )
```

In `_handle_expert_request`, find the call to `run_selected_experts`:
```python
                    outputs = run_selected_experts(
                        self._lm, h, layer_idx, requested
                    )
```

Replace with:
```python
                    outputs = self._backend.run_selected_experts(
                        layer_idx, h, requested,
                    )
```

Find `bytes_to_tensor` call in the same function:
```python
        h = bytes_to_tensor(
            tensor_bytes,
            shape=list(req.h_spec.shape),
            dtype=req.h_spec.dtype,
        )
```

Replace with:
```python
        h = self._backend.bytes_to_tensor(
            tensor_bytes, shape=list(req.h_spec.shape), dtype=int(req.h_spec.dtype),
        )
```

Find the response-side `tensor_to_bytes` / dtype conversion:
```python
                    raw = tensor_to_bytes(stacked)
            ...
            resp.expert_response.outputs_spec.dtype = _dtype_to_wire(stacked.dtype)
```

Replace with:
```python
                    raw = self._backend.tensor_to_bytes(stacked)
            ...
            resp.expert_response.outputs_spec.dtype = self._backend.dtype_to_wire(stacked)
```

(Similar treatment for `_handle_expert_weight_request` — `slice_expert`, `tensor_to_bytes`, `dtype_to_wire`.)

Find `slice_expert` call:
```python
            tensors = slice_expert(
                self._lm, layer_idx, expert_id, _MLX_COMPUTE_LOCK
            )
```

Replace with:
```python
            tensors = self._backend.slice_expert(layer_idx, expert_id)
```

Find `migration_attach` / `migration_detach` body calls to `attach_expert` / `detach_expert`:
```python
        attach_expert(
            self._lm, layer_idx, expert_id, tensors, _MLX_COMPUTE_LOCK
        )
```
Replace with:
```python
        self._backend.attach_expert(layer_idx, expert_id, tensors)
```

```python
        detach_expert(self._lm, layer_idx, expert_id, _MLX_COMPUTE_LOCK)
```
Replace with:
```python
        self._backend.detach_expert(layer_idx, expert_id)
```

In `_run_my_layers`, keep calling `run_layers(self._lm, ...)` — this function's signature is still MLX-specific and already uses `run_layer_atomic` internally (Task 2). **Do not change `_run_my_layers` in Task 5.**

Finally, drop any `from model_shard.mlx_engine import embed_tokens, make_cache, ...` import lines that are no longer used. Keep `run_layers` import because `_run_my_layers` still needs it.

Also drop the module-level `_dtype_to_wire` in `node.py` if it was only used by the refactored code (spec D8 notes this). Keep it if `_handle_expert_request`'s response still needs module-level access — probably not after the refactor.

- [ ] **Step 4: Run full Node regression**

```bash
uv run pytest tests/test_node_membership.py tests/test_node_load_wiring.py tests/test_node_partial_load_wiring.py tests/test_node_live_experts.py tests/test_node_eviction.py tests/test_node_expert_weight_handler.py tests/test_decode_hang_fix.py tests/test_dynamic_migration_gate.py tests/test_node_backend_wiring.py tests/test_handle_expert_request_authority.py tests/test_provenance_integration_unit.py -v -m "not slow"
```
Expected: all pass.

- [ ] **Step 5: Ruff + mypy clean**

```bash
uv run ruff check src/model_shard/node.py
uv run mypy src/model_shard/node.py
```

- [ ] **Step 6: Commit**

```bash
git add src/model_shard/node.py
git commit -m "Phase 7-A Task 5: Node method bodies route through self._backend"
```

## Context

- **Predecessor commit:** Task 4.
- **Plan file:** this file.
- **Spec:** §2.3.
- **Critical:** this is a large, mechanical diff. Every call site should have a matching replacement; search for leftover bare function calls after you're done.

## Your Job

1. Follow Steps 1-6. Survey call sites first (Step 1); apply mechanically (Step 3).
2. Regression green.
3. Ruff + mypy clean.
4. Commit.
5. Report back with the survey result (how many call sites you changed).

---

### Task 6: `ExpertOrchestrator` refactor — route through `self.backend`

**Files:**
- Modify: `src/model_shard/expert_orchestrator.py`
- Modify: `src/model_shard/node.py` (pass `backend=self._backend` to the orchestrator)

- [ ] **Step 1: Add `backend` field to `ExpertOrchestrator`**

In `src/model_shard/expert_orchestrator.py`, add at the top-level imports:

```python
from model_shard.backends import Backend
```

In the `ExpertOrchestrator` dataclass, add a field:

```python
    backend: Backend | None = None
```

Place it after the existing fields (e.g., after `heat_observer`, before `retry_max_attempts`) so no existing kwarg ordering breaks. Default `None` preserves pre-Phase-7 call-sites.

- [ ] **Step 2: Refactor compute calls in `run_split_layer` Phase A**

Find the Phase A block that computes `post_attn`, `shared_out`, `local_outputs`. Today it calls `moe.run_attention_and_route`, `moe.run_shared_expert`, `moe.run_selected_experts`. Replace with backend calls when `self.backend is not None`; fall back to `moe.X` when `None` (temporary; 7-B removes fallback).

Find the existing code that looks roughly like:
```python
        with self._mlx_guard():
            post_attn, top_k_ids, top_k_weights = run_attention_and_route(
                lm, h, layer_idx, cache, masks,
                heat_observer=self.heat_observer,
            )
            ...
            shared_out = run_shared_expert(lm, post_attn, layer_idx)
            local_outputs = run_selected_experts(lm, post_attn, layer_idx, local_ids)
```

Replace with:
```python
        with self._mlx_guard():
            if self.backend is not None:
                post_attn, top_k = self.backend.run_attention_and_route(
                    layer_idx, h, cache, masks,
                    heat_observer=self.heat_observer,
                )
                top_k_ids, top_k_weights = top_k
            else:
                post_attn, top_k_ids, top_k_weights = run_attention_and_route(
                    lm, h, layer_idx, cache, masks,
                    heat_observer=self.heat_observer,
                )
            ...
            if self.backend is not None:
                shared_out = self.backend.run_shared_expert(layer_idx, post_attn)
                local_outputs = self.backend.run_selected_experts(
                    layer_idx, post_attn, local_ids,
                )
            else:
                shared_out = run_shared_expert(lm, post_attn, layer_idx)
                local_outputs = run_selected_experts(lm, post_attn, layer_idx, local_ids)
```

- [ ] **Step 3: Refactor `aggregate_experts` call in Phase C**

Find the Phase C block that aggregates. Today's code constructs `post_ffn_ln_2` from the layer module and passes it to `aggregate_experts`. Replace with backend call:

Find:
```python
                    agg = aggregate_experts(
                        per_pos, ids, weights, per_pos_shared, post_ffn_ln_2
                    )
```

Replace with:
```python
                    if self.backend is not None:
                        agg = self.backend.aggregate_experts(
                            layer_idx, per_pos, ids, weights, per_pos_shared,
                        )
                    else:
                        agg = aggregate_experts(
                            per_pos, ids, weights, per_pos_shared, post_ffn_ln_2,
                        )
```

- [ ] **Step 4: Also handle the retry path's local-expert compute**

`_phase_b_with_retry` has two places where it calls `run_selected_experts` directly (the initial local-route block and the retry-local block). Both currently look like:

```python
                with self._mlx_guard():
                    outputs.update(
                        run_selected_experts(lm, post_attn, layer_idx, local_retry)
                    )
```

Replace with:
```python
                with self._mlx_guard():
                    if self.backend is not None:
                        outputs.update(
                            self.backend.run_selected_experts(
                                layer_idx, post_attn, local_retry,
                            )
                        )
                    else:
                        outputs.update(
                            run_selected_experts(lm, post_attn, layer_idx, local_retry)
                        )
```

Apply to BOTH call sites (initial + retry).

- [ ] **Step 5: Wire `backend` through from Node**

In `src/model_shard/node.py`, find the `ExpertOrchestrator(...)` constructor call in `_build_expert_orchestrator`. Add `backend=self._backend`:

```python
            return ExpertOrchestrator(
                self_shard_id=self._shard.shard_id,
                owners=owners,
                peer_rpc=TcpPeerRPC(addresses=addresses, timeout_s=30.0),
                rpc_timeout_s=30.0,
                mlx_lock=_MLX_COMPUTE_LOCK,
                loads_provider=_loads_provider,
                rng=_random_mod.Random(),
                live_owners_provider=self.owners_of,
                heat_observer=self._heat_tracker.observe,
                retry_max_attempts=_expert_retry_max_attempts(),
                retry_backoff_ms=_expert_retry_backoff_ms(),
                backend=self._backend,
            )
```

- [ ] **Step 6: Run regression**

```bash
uv run pytest tests/test_expert_orchestrator.py tests/test_expert_retry_unit.py tests/test_expert_rpc_load_shift.py tests/test_orchestrator_live_owners.py tests/test_tcp_peer_rpc.py tests/test_expert_weight_peer_rpc.py tests/test_provenance_integration_unit.py -v -m "not slow"
```
Expected: all pass. The `backend=None` fallback keeps existing `ExpertOrchestrator(...)`-from-tests construction working without explicit backend args.

- [ ] **Step 7: Ruff + mypy clean**

```bash
uv run ruff check src/model_shard/expert_orchestrator.py src/model_shard/node.py
uv run mypy src/model_shard/expert_orchestrator.py src/model_shard/node.py
```

- [ ] **Step 8: Commit**

```bash
git add src/model_shard/expert_orchestrator.py src/model_shard/node.py
git commit -m "Phase 7-A Task 6: ExpertOrchestrator routes compute through self.backend"
```

## Context

- **Predecessor commit:** Task 5.
- **Plan file:** this file.
- **Spec:** §2.4, §D7.
- **FIXME note:** The `if self.backend is not None:` fallback is temporary. Phase 7-B will make `backend` required. Document this with a FIXME comment near the field declaration so it's visible to future readers.

## Your Job

1. Follow Steps 1-8.
2. Regression green (existing orchestrator + retry + RPC tests all pass with default `backend=None`).
3. Ruff + mypy clean.
4. Commit.
5. Report back.

---

### Task 7: Final verification sweep + README + memory update

**Files:**
- Modify: `README.md`
- Modify: `/Users/lukechang/.claude/projects/-Users-lukechang-Github-model-shard/memory/project_gossip_moe.md`

- [ ] **Step 1: Add Phase 7-A status paragraph to README**

Insert after the existing Phase 6-C status paragraph. Match the existing style (~200 words, no emojis). Cover:

- Scope: introduces a `Backend` protocol + `MLXBackend` that wraps every tensor-level operation. Zero behavioral change on default MLX deployments.
- Architecture: stateful Backend class owns the `LoadedModel`; opaque `Activation`/`Cache`/`Mask`/`TopK` handles; consumers pass handles between backend calls or serialize via `tensor_to_bytes`.
- `Node.__init__(backend=None)` defaults to `MLXBackend()`. Legacy `loaded_model=` callers supported via `MLXBackend.from_loaded_model`.
- `ExpertOrchestrator.backend` field added; compute calls go through `self.backend.X()`. Temporary `backend=None` fallback preserves Phase 1-6 construction patterns; Phase 7-B will remove the fallback.
- `mlx_engine.py` gained a public `run_layer_atomic` helper and a `mx_to_wire_dtype` alias so backends don't depend on private names.
- Correctness preserved: Tier 1 bit-exact to Phase 1 reference via default `MLXBackend`. All Phase 1-6 E2E tests (migration, retry, provenance, eviction) pass unchanged.
- Purpose: opens the seam for Phase 7-B (`PyTorchBackend` for CUDA / DGX Spark) and Phase 7-C (heterogeneous cluster with `allclose` + top-1 correctness bar across platforms).
- Link to spec: `docs/superpowers/specs/2026-04-19-phase7a-backend-protocol-design.md`.

- [ ] **Step 2: Update memory file**

Add a Phase 7-A COMPLETE paragraph to `project_gossip_moe.md`, parallel to the Phase 6-C entry. Cover:

- Date `2026-04-19`, final commit SHA (fill in after your own commit).
- 7 tasks done.
- Links to plan + spec.
- What it enables: an abstraction seam for cross-platform work. Phase 7-B will add `PyTorchBackend`; Phase 7-C heterogeneous cluster.
- What changed technically: `Backend` protocol in `src/model_shard/backends/base.py`; `MLXBackend` in `src/model_shard/backends/mlx_backend.py`; Node refactor; ExpertOrchestrator refactor; mlx_engine helpers (`run_layer_atomic`, `mx_to_wire_dtype`).
- What didn't change: wire protocol, gossip, orchestrator logic, retry, provenance, eviction, bit-exact correctness bar, Tier 1 reference tokens.
- Phase 7 decomposition: 7-A (protocol + MLXBackend) ✅; 7-B (PyTorchBackend for CUDA) — next; 7-C (heterogeneous cluster) — after.
- Technical debt noted: `ExpertOrchestrator.backend=None` fallback is temporary; `Node._lm` property is temporary. Both removed in 7-B.
- Next: Phase 7-B brainstorm — `PyTorchBackend` on DGX Spark. Or defer, if more Apple Silicon acquisition is simpler.

- [ ] **Step 3: Full verification sweep**

Run from `/Users/lukechang/Github/model_shard`:

```bash
uv run pytest -q                                              # fast
uv run pytest -m slow -q tests/test_tier1_tokens.py           # Tier 1 bit-exact
uv run pytest -m slow -q tests/test_partial_load_bit_exact_per_expert.py  # 5a regression
uv run pytest -m slow -q tests/test_migration_over_tcp.py     # 5b regression
uv run pytest -m slow -q tests/test_expert_retry_bit_exact.py # 6-A regression
uv run pytest -m slow -q tests/test_provenance_tier1.py       # 6-B regression
uv run pytest -m slow -q tests/test_eviction_e2e.py           # 6-C regression
uv run ruff check src tests scripts
uv run mypy src
```

Expected: all green. Known Phase 3 Metal in-process artifact on `test_partial_load_tier1_migration.py` may need to be run in isolation (matches prior-phase behavior).

- [ ] **Step 4: Commit**

```bash
git add README.md "/Users/lukechang/.claude/projects/-Users-lukechang-Github-model-shard/memory/project_gossip_moe.md"
git commit -m "Phase 7-A Task 7: README + memory update; backend abstraction landed"
```

- [ ] **Step 5: Report**

Include:
- The Phase 7-A README paragraph text.
- Verification results per bucket.
- Final commit SHA.
- Phase 7-A commit list: `git log --grep "Phase 7-A" --oneline`.

## Context

- **Predecessor commit:** Task 6.
- **Plan file:** this file.
- **Spec:** §6 acceptance criteria.

## Your Job

1. README paragraph (match existing style).
2. Memory update.
3. Full verification sweep.
4. Single commit.
5. Full report.

---

## Self-Review Notes

**Spec coverage:**
- D1 (behavioral no-op refactor) → enforced by the "all existing tests green" acceptance criterion in every task.
- D2 (stateful Backend class) → Task 1 defines; Task 3 implements.
- D3 (opaque handles) → Task 1 declares `Activation`/`Cache`/`Mask`/`TopK` as `Any`.
- D4 (backend selection + legacy compat) → Task 4.
- D5 (`_MLX_COMPUTE_LOCK` at Node level, passed into backend) → Task 3 (`mlx_lock` kwarg) + Task 4 (Node passes it).
- D6 (existing modules stay) → MLXBackend delegates to `mlx_engine`/`moe`/`partial_load` in Task 3.
- D7 (ExpertOrchestrator gains `backend`) → Task 6.
- D8 (wire dtype unchanged) → Task 2 aliases `_mx_to_wire_dtype` publicly; no wire-level change.
- D9 (correctness bar) → Task 7 verification sweep.
- D10 (non-goals) → plan excludes PyTorch / CUDA / cross-backend work; fallback in Orchestrator is flagged as temporary.

**Placeholder scan:** No "TBD" / "add error handling" / vague steps. Every code step has complete code or a clear rewrite recipe. Task 5's "identify every call site" is a bounded mechanical search, not a vague step.

**Type consistency:**
- `Backend` protocol methods use `Activation`/`Cache`/`Mask`/`TopK` consistently across Tasks 1, 3, 5, 6.
- `MLXBackend.from_loaded_model` / `MLXBackend(mlx_lock=...)` constructors referenced in Tasks 3 and 4.
- `backend.tensor_to_bytes(h)` vs module-level `tensor_to_bytes(h)` — same signature, zero-arg rename at call site.
- `backend.dtype_to_wire(h)` takes tensor, whereas `_mx_to_wire_dtype(h.dtype)` takes dtype — Task 5 notes this signature change.

No type-name drift. No references to methods not declared in Task 1's protocol or implemented in Task 3's MLXBackend.
