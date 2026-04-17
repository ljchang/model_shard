# Phase 5a — Partial Expert Weight Loading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A node loads only the routed experts listed in its `ShardSpec.moe_experts` (custom safetensors slice-reader) rather than the full 128-expert stack per layer; `run_selected_experts` translates global expert ids to compact local slot indices before dispatch; bit-exact against the full-loaded path.

**Architecture:** New `partial_load.py` walks the model's safetensors files and, for each stacked `experts.switch_glu.*` tensor at a held layer, reads only the subset of rows (axis 0) corresponding to held expert ids. The stock mlx-vlm `Experts` / `SwitchLinear` modules are untouched; after construction the compact `(k, out, in)` tensor is swapped in via `model.load_weights(...)`. Index remapping lives in `moe.run_selected_experts`. The full router + `per_expert_scale[128]` stays on every node (router runs identically everywhere). Opt-in via `ENABLE_PARTIAL_LOAD=true`.

**Tech Stack:** mlx-vlm (Gemma 4), safetensors (`safe_open` + `get_slice`), MLX 4-bit quantization, Python 3.13, protobuf unchanged (no wire format changes in 5a).

**Design spec:** `docs/superpowers/specs/2026-04-17-phase5a-partial-expert-loading-design.md`

---

## File Structure

**New files:**
- `src/model_shard/partial_load.py` — safetensors slice-reader + `load_model_partial(hf_id, held_experts_per_layer)`.
- `tests/test_partial_load_slice_math.py` — fast unit tests on the pure slice helper.
- `tests/test_partial_load_bit_exact_per_expert.py` — slow: sliced vs full, per-expert bit-exact.
- `tests/test_partial_load_split_equivalence.py` — slow: three sliced LMs (mod-3 at layer 15) composed vs atomic full.
- `tests/test_partial_load_missing_expert_raises.py` — slow: unknown global id → `KeyError`.
- `tests/test_partial_load_tier1_e2e.py` — slow: Tier 1 still bit-exact under `ENABLE_PARTIAL_LOAD=true`.

**Modified:**
- `src/model_shard/mlx_engine.py` — `LoadedModel.held_ids_per_layer` field; `load_model_partial` wrapper.
- `src/model_shard/moe.py` — global→local translation in `run_selected_experts`.
- `src/model_shard/node.py` — `_partial_load_enabled()` + conditional load.
- `README.md` — Phase 5a status paragraph.

---

## Task Overview

| # | Task | Blocker |
|---|---|---|
| 1 | Recon: resolve spec §7 open questions (mlx-vlm internals) | — |
| 2 | Pure slice helper `_slice_stacked_by_axis0` + fast tests | 1 |
| 3 | `load_model_partial` — sliced safetensors read + weight swap | 1, 2 |
| 4 | `LoadedModel.held_ids_per_layer` field | 3 |
| 5 | `run_selected_experts` global→local translation | 4 |
| 6 | `KeyError` on unknown global id | 5 |
| 7 | Slow bit-exact per-expert test | 3, 5 |
| 8 | Slow split-equivalence under sliced load | 3, 5 |
| 9 | Node integration: `ENABLE_PARTIAL_LOAD` env var + conditional load | 4 |
| 10 | Phase 3/4 regression pass (default-OFF) | 9 |
| 11 | Slow Tier 1 E2E under `ENABLE_PARTIAL_LOAD=true` | 9 |
| 12 | Final acceptance — lint, types, README, memory | all |

---

## Task 1: Recon — resolve spec §7 open questions

**Files:**
- Read only: `.venv/lib/python3.13/site-packages/mlx_vlm/utils.py`, `mlx_vlm/models/gemma4/language.py`, `mlx_lm/models/switch_layers.py`.
- Modify: `docs/superpowers/specs/2026-04-17-phase5a-partial-expert-loading-design.md` — replace §7 with resolved answers.

This is pure reconnaissance — no production code. The design doc has three open questions; Task 1 resolves them so downstream tasks don't fail mysteriously.

- [ ] **Step 1: Confirm SwitchLinear accepts post-init weight shape change**

Open `mlx_lm/models/switch_layers.py`. Find `SwitchLinear.__init__`, `SwitchLinear.__call__`, and the `num_experts` property. Verify:
1. `num_experts` is a `@property` returning `self.weight.shape[0]` (not a field set at init).
2. `__call__` uses `self["weight"]` (dynamic lookup), not a captured reference.
3. Quantized variant `QuantizedSwitchLinear` likewise.

If any of these uses a captured `num_experts` or a frozen shape assumption, note it — we'd need a different strategy (subclass instead of weight replacement).

- [ ] **Step 2: Determine the safetensors slice API**

Run:
```bash
uv run python -c "
import safetensors
from huggingface_hub import snapshot_download
path = snapshot_download('mlx-community/gemma-4-26b-a4b-it-4bit')
import glob, os
f = sorted(glob.glob(os.path.join(path, '*.safetensors')))[0]
with safetensors.safe_open(f, framework='np') as sf:
    key = 'language_model.model.layers.0.experts.switch_glu.gate_proj.weight'
    s = sf.get_slice(key)
    print('shape:', s.get_shape())
    print('dtype:', s.get_dtype())
    partial = s[[0, 3, 6], :, :]   # read rows 0, 3, 6 only
    print('partial shape:', partial.shape)
    print('partial dtype:', partial.dtype)
"
```

Record: does `get_slice(...)[rows, :, :]` work with a Python list of ints? If the shape is `(128, O, I)` in 4-bit packed format (likely dtype uint32 with width `I//8` along last dim), confirm the slice math still makes sense. Record the exact dtype and shape.

- [ ] **Step 3: Confirm mlx-vlm's weight-loading entry point**

Find where `mlx_vlm.utils.load_model` calls into the model class to attach weights. Specifically, locate the `model.load_weights(list(weights.items()))` or equivalent. Confirm:
1. The `weights` dict is a `{key: mx.array}` map.
2. `model.load_weights(...)` uses `mlx.nn.Module.load_weights`, which accepts mismatched shapes only when `strict=False`. Under the default `strict=True`, our compact `(k, out, in)` tensor would FAIL because the model class constructed with `num_experts=128` has `SwitchLinear.weight.shape == (128, out, in)`.

If strict mode rejects the shape mismatch, we have options:
- **Option A:** Call `load_weights(..., strict=False)` and rely on the subsequent shape propagation through `num_experts` being a property.
- **Option B:** Construct the model with a patched config where each held layer's `num_experts` is `k`. This requires model-level cooperation; Gemma 4's config may not support per-layer variance.
- **Option C:** Post-load weight replacement: load with full 128 (peak memory blip), then overwrite the `weight` / `scales` / `biases` fields with compact tensors. This is essentially brainstorm-option-A despite us choosing option-B; spec is still honored because the final resident state is partial.

Record which option works.

- [ ] **Step 4: Write resolved §7 into the spec**

Edit `docs/superpowers/specs/2026-04-17-phase5a-partial-expert-loading-design.md` §7 ("Open Technical Questions") into a new §7 ("Resolved Technical Choices"). For each question, state the answer in 1-2 sentences naming the exact mlx / mlx-vlm API. Example:

```markdown
## 7. Resolved Technical Choices (2026-04-17)

**SwitchLinear post-init weight replacement.** `num_experts` is a `@property`
returning `self.weight.shape[0]`, and `__call__` uses `self["weight"]` (dynamic
lookup). Replacing `layer.experts.switch_glu.gate_proj.weight` with a compact
`(k, O, I)` mx.array works: `num_experts` then reports `k`. `QuantizedSwitchLinear`
has the same pattern for `weight`, `scales`, `biases`. No subclass needed.

**Safetensors partial read.** `safetensors.safe_open(..., framework="np").get_slice(key)`
returns a SliceObject that supports Python-list indexing on axis 0:
`slice_obj[held_ids, :, :]` returns a numpy array of the sliced rows. 4-bit
quantized weights are stored as uint32 with last-dim packed to `I//8`; axis-0
slicing leaves the per-row quantization groups intact, so no requantization is
needed.

**mlx-vlm weight attachment.** `mlx_vlm.utils.load_model` calls
`model.load_weights(list(weights.items()))` with default `strict=True`. For our
compact weights to be accepted, we use **Option C (post-load replacement)**:
load normally with full 128 via `mx.load`, construct the model with stock
`num_experts=128`, call `model.load_weights(full_weights)`, THEN iterate the
held layers and do `layer.experts.switch_glu.<proj>.weight = compact_tensor`.
Because mlx arrays are not Python-side refcounted the way Python objects are,
we also call `mx.metal.clear_cache()` after the reassignment to release the
full stacked tensors. Peak memory blips to 14 GB during load but drops to
chassis + held for the rest of the process lifetime.
```

(Adjust the actual findings to match what you observed in steps 1-3.)

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-04-17-phase5a-partial-expert-loading-design.md
git commit -m "Phase 5a: resolve §7 — mlx-vlm partial-load APIs confirmed"
```

---

## Task 2: `_slice_stacked_by_axis0` pure helper

**Files:**
- Create: `src/model_shard/partial_load.py`
- Create: `tests/test_partial_load_slice_math.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_partial_load_slice_math.py`:

```python
"""Fast unit tests for the pure axis-0 slice helper used by the partial loader."""

from __future__ import annotations

import numpy as np
import pytest

from model_shard.partial_load import _slice_stacked_by_axis0


def test_slice_3d_by_ids() -> None:
    arr = np.arange(128 * 4 * 5, dtype=np.int32).reshape(128, 4, 5)
    out = _slice_stacked_by_axis0(arr, [0, 3, 127])
    assert out.shape == (3, 4, 5)
    assert np.array_equal(out[0], arr[0])
    assert np.array_equal(out[1], arr[3])
    assert np.array_equal(out[2], arr[127])


def test_slice_2d_by_ids() -> None:
    arr = np.arange(128 * 7, dtype=np.int32).reshape(128, 7)
    out = _slice_stacked_by_axis0(arr, [5, 42])
    assert out.shape == (2, 7)
    assert np.array_equal(out[0], arr[5])
    assert np.array_equal(out[1], arr[42])


def test_slice_preserves_dtype() -> None:
    arr = np.zeros((128, 3), dtype=np.uint32)
    out = _slice_stacked_by_axis0(arr, [1])
    assert out.dtype == np.uint32
    assert out.shape == (1, 3)


def test_slice_empty_ids_returns_empty_axis0() -> None:
    arr = np.zeros((128, 3), dtype=np.int32)
    out = _slice_stacked_by_axis0(arr, [])
    assert out.shape == (0, 3)


def test_slice_preserves_id_order() -> None:
    """Order of returned rows follows the caller's id order, not sorted."""
    arr = np.arange(128 * 2, dtype=np.int32).reshape(128, 2)
    out = _slice_stacked_by_axis0(arr, [10, 5, 100])
    assert np.array_equal(out[0], arr[10])
    assert np.array_equal(out[1], arr[5])
    assert np.array_equal(out[2], arr[100])


def test_slice_out_of_bounds_raises() -> None:
    arr = np.zeros((128, 3), dtype=np.int32)
    with pytest.raises((IndexError, ValueError)):
        _slice_stacked_by_axis0(arr, [999])
```

- [ ] **Step 2: Run — expect failure**

Run: `uv run pytest tests/test_partial_load_slice_math.py -v`
Expected: `ModuleNotFoundError: No module named 'model_shard.partial_load'`.

- [ ] **Step 3: Create `src/model_shard/partial_load.py`**

```python
"""Partial expert-weight loading for Phase 5a.

A shard can declare which routed experts it holds per layer (via
ShardSpec.moe_experts). This module provides a custom safetensors reader
that slices the stacked (128, out, in) expert projection tensors at load
time so the shard's resident memory contains only the held experts'
weights.

Chassis weights (attention, dense mlp, norms, embeddings, LM head, router)
load unchanged on every node.
"""

from __future__ import annotations

import numpy as np


def _slice_stacked_by_axis0(
    arr: np.ndarray, ids: list[int]
) -> np.ndarray:
    """Return the rows of `arr` at positions `ids` along axis 0.

    Order is preserved: the returned array's row `i` is `arr[ids[i]]`.
    Raises IndexError or ValueError if any id is out of bounds.
    """
    if not ids:
        # numpy doesn't special-case empty fancy-index; use shape-preserving take.
        return arr[0:0]
    return arr[ids]


__all__ = ["_slice_stacked_by_axis0"]
```

- [ ] **Step 4: Run — expect pass**

Run: `uv run pytest tests/test_partial_load_slice_math.py -v`
Expected: 6 passed.

- [ ] **Step 5: Lint + types**

```
uv run ruff check src/model_shard/partial_load.py tests/test_partial_load_slice_math.py
uv run mypy src/model_shard/partial_load.py
```

Both clean.

- [ ] **Step 6: Commit**

```bash
git add src/model_shard/partial_load.py tests/test_partial_load_slice_math.py
git commit -m "Phase 5a: partial_load._slice_stacked_by_axis0 + fast tests"
```

---

## Task 3: `load_model_partial` — sliced safetensors read + post-load weight swap

**Files:**
- Modify: `src/model_shard/partial_load.py`
- Modify: `src/model_shard/mlx_engine.py`

No new test file here — Task 7's bit-exact test is the true proof. This task is the load-path scaffolding.

- [ ] **Step 1: Implement `load_model_partial`**

Append to `src/model_shard/partial_load.py`:

```python
import glob
import logging
import os
import re
from pathlib import Path
from typing import Any

import mlx.core as mx
import safetensors
from huggingface_hub import snapshot_download

from model_shard.mlx_engine import LoadedModel

_LOG = logging.getLogger(__name__)

_EXPERT_KEY_RE = re.compile(
    r"^language_model\.model\.layers\.(\d+)\.experts\.switch_glu\."
    r"(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$"
)


def load_model_partial(
    hf_id: str,
    held_experts_per_layer: dict[int, list[int]],
) -> LoadedModel:
    """Load the Gemma 4 26B model with routed-expert weights restricted
    to the held subset per layer.

    Layers absent from `held_experts_per_layer` load full 128-expert
    stacks (same as `load_model`). Chassis weights (attention, norms,
    embeddings, LM head, router) always load fully.

    Strategy: use mlx-vlm's standard `load()` to construct the full model
    normally (peak memory blip ~14 GB), then iterate the held layers and
    replace each layer's `experts.switch_glu.<proj>.{weight, scales,
    biases}` with a compact (k, ...) tensor sliced along axis 0.
    """
    from mlx_vlm import load as _mlx_vlm_load

    model, processor = _mlx_vlm_load(hf_id)
    language_model = model.language_model
    text_model = language_model.model
    num_layers = len(text_model.layers)

    # Slice the stacked tensors on held layers.
    for layer_idx, ids in held_experts_per_layer.items():
        if not ids:
            continue
        layer = text_model.layers[layer_idx]
        switch_glu = layer.experts.switch_glu
        # Slice weight, scales, biases for each of gate/up/down projections.
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            proj = getattr(switch_glu, proj_name)
            # Each attribute is an mx.array with leading dim num_experts.
            # Replace with compact tensor by row indexing.
            for attr in ("weight", "scales", "biases"):
                if not hasattr(proj, attr):
                    continue
                full = getattr(proj, attr)
                if full is None:
                    continue
                # mx.array supports fancy indexing via mx.take.
                held = mx.take(full, mx.array(list(ids)), axis=0)
                setattr(proj, attr, held)
        _LOG.info(
            "partial_load: layer %d sliced to %d experts (from 128)",
            layer_idx,
            len(ids),
        )

    # Release the full-stacked tensors that are no longer referenced.
    mx.metal.clear_cache()

    # Normalize held_ids: tuple per layer for immutability.
    held_ids_norm = {k: tuple(v) for k, v in held_experts_per_layer.items()}

    return LoadedModel(
        mlx_model=model,
        language_model=language_model,
        text_model=text_model,
        processor=processor,
        num_layers=num_layers,
        held_ids_per_layer=held_ids_norm,
    )


__all__ = ["_slice_stacked_by_axis0", "load_model_partial"]
```

**NOTE:** this implementation uses the **post-load replacement** strategy (Task 1 Step 3 Option C). If Task 1 confirms that the true safetensors-skip strategy (never materialize the full tensor) is feasible without fighting `model.load_weights` strict mode, rewrite this function to bypass `mlx_vlm.load()` entirely and construct the model + weights dict manually. For Phase 5a the post-load strategy is acceptable: peak memory blips to 14 GB briefly, resident drops after `mx.metal.clear_cache()`.

- [ ] **Step 2: Add `held_ids_per_layer` to `LoadedModel` and export the new helper**

Modify `src/model_shard/mlx_engine.py`. Change the `LoadedModel` dataclass:

```python
from dataclasses import dataclass, field


@dataclass
class LoadedModel:
    """Thin handle over mlx-vlm's loaded Gemma 4 model."""

    mlx_model: Any
    language_model: Any
    text_model: Any
    processor: Any
    num_layers: int
    # Phase 5a: per-layer list of held routed-expert global ids. Empty dict
    # (or an absent layer_idx key) means that layer holds all 128 experts.
    held_ids_per_layer: dict[int, tuple[int, ...]] = field(default_factory=dict)
```

Existing `load_model` callers don't pass `held_ids_per_layer` — the default makes old call-sites source-compatible.

At the bottom of `mlx_engine.py`, add:

```python
def load_model_partial(
    hf_id: str,
    held_experts_per_layer: dict[int, list[int]],
) -> LoadedModel:
    """Phase 5a wrapper. See partial_load.load_model_partial for semantics."""
    from model_shard.partial_load import load_model_partial as _impl
    return _impl(hf_id, held_experts_per_layer)
```

- [ ] **Step 3: Smoke-test the loader**

Add an ad-hoc slow test at `tests/test_partial_load_smoke.py`:

```python
"""Smoke test: load_model_partial returns a LoadedModel with correct shape."""

from __future__ import annotations

import mlx.core as mx
import pytest

from model_shard.mlx_engine import load_model_partial


@pytest.mark.slow
def test_partial_load_slices_layer_experts() -> None:
    held = {15: [0, 3, 6, 9]}
    lm = load_model_partial("mlx-community/gemma-4-26b-a4b-it-4bit", held)

    assert lm.num_layers == 30
    assert lm.held_ids_per_layer == {15: (0, 3, 6, 9)}

    layer15 = lm.text_model.layers[15]
    w = layer15.experts.switch_glu.gate_proj.weight
    # Held-layer weight has leading dim == len(held_ids).
    assert w.shape[0] == 4

    # A non-held layer retains full 128.
    layer0 = lm.text_model.layers[0]
    w0 = layer0.experts.switch_glu.gate_proj.weight
    assert w0.shape[0] == 128
```

Run: `uv run pytest -m slow tests/test_partial_load_smoke.py -v`
Expected: pass.

- [ ] **Step 4: Lint + types**

```
uv run ruff check src/model_shard/partial_load.py src/model_shard/mlx_engine.py tests/test_partial_load_smoke.py
uv run mypy src/model_shard/partial_load.py src/model_shard/mlx_engine.py
```

Both clean.

- [ ] **Step 5: Commit**

```bash
git add src/model_shard/partial_load.py src/model_shard/mlx_engine.py tests/test_partial_load_smoke.py
git commit -m "Phase 5a: load_model_partial — post-load slice of stacked expert weights"
```

---

## Task 4: `LoadedModel.held_ids_per_layer` field plumbing

(Rolled into Task 3 Step 2 — Task 4 is a no-op in this plan. The field was added alongside `load_model_partial`. Skip Task 4; renumber downstream if needed, or leave this as an explicit no-op acknowledgement.)

---

## Task 5: `run_selected_experts` global→local index translation

**Files:**
- Modify: `src/model_shard/moe.py`
- Create: `tests/test_partial_load_run_selected.py`

- [ ] **Step 1: Write the failing test**

```python
"""Unit test that run_selected_experts translates global ids to local slots
when lm.held_ids_per_layer[layer_idx] is non-empty."""

from __future__ import annotations

import mlx.core as mx
import pytest

from model_shard.mlx_engine import load_model, load_model_partial
from model_shard.moe import run_selected_experts


@pytest.mark.slow
def test_run_selected_experts_sliced_lm_returns_correct_outputs(loaded_model) -> None:
    """Given the same input h, run_selected_experts on a sliced model for a
    held id returns the same tensor as on the full model for that id."""
    lm_full = loaded_model
    held_ids = [0, 3, 6, 9]
    lm_part = load_model_partial(
        "mlx-community/gemma-4-26b-a4b-it-4bit",
        {15: held_ids},
    )
    try:
        h = mx.random.normal((1, 3, lm_full.text_model.config.hidden_size)).astype(mx.bfloat16)
        out_full = run_selected_experts(lm_full, h, layer_idx=15, expert_ids=[3])
        out_part = run_selected_experts(lm_part, h, layer_idx=15, expert_ids=[3])
        mx.eval(out_full[3], out_part[3])
        assert mx.array_equal(out_full[3], out_part[3]), (
            f"bit-exact failure for expert 3; max abs diff = "
            f"{mx.max(mx.abs(out_full[3] - out_part[3])).item()}"
        )
    finally:
        # Attempt to release the partial model's memory.
        del lm_part
        mx.metal.clear_cache()
```

- [ ] **Step 2: Run — expect failure**

Run: `uv run pytest -m slow tests/test_partial_load_run_selected.py -v`
Expected: either the test passes accidentally (if run_selected_experts happens to use local gather indexing) or it fails with an out-of-bounds or mismatched-shape gather error. Most likely failure: "Index out of bounds" inside `SwitchLinear.__call__` because `run_selected_experts` passes global id 3 but the sliced tensor only has 4 rows (slots 0..3).

- [ ] **Step 3: Modify `run_selected_experts` to translate**

Read the current `run_selected_experts` (in `src/model_shard/moe.py`, added in Phase 3 Task 7). It builds `indices = mx.full((B * L, 1), eid, dtype=mx.int32)` where `eid` is the global expert id. Replace the iteration to translate:

```python
def run_selected_experts(
    lm: Any,
    h: mx.array,
    layer_idx: int,
    expert_ids: list[int],
) -> dict[int, mx.array]:
    """..."""
    if not expert_ids:
        return {}
    layer = lm.text_model.layers[layer_idx]
    h_normed = layer.pre_feedforward_layernorm_2(h)
    B, L, H = h_normed.shape

    # Phase 5a: if this shard holds only a subset of experts for this layer,
    # translate each requested global id to its local slot in the compact
    # stacked weight tensor.
    held = lm.held_ids_per_layer.get(layer_idx)
    if held:
        global_to_local = {gid: li for li, gid in enumerate(held)}
    else:
        global_to_local = None

    per_expert: dict[int, mx.array] = {}
    one_weight = mx.ones((B * L, 1), dtype=h_normed.dtype)
    h_flat = h_normed.reshape(B * L, H)
    for eid in expert_ids:
        if global_to_local is not None:
            try:
                slot = global_to_local[int(eid)]
            except KeyError as e:
                raise KeyError(
                    f"expert {eid} not held on this shard "
                    f"(layer {layer_idx} held ids: {held})"
                ) from e
        else:
            slot = int(eid)
        idx = mx.full((B * L, 1), slot, dtype=mx.int32)
        out_flat = layer.experts(h_flat[:, None, :], idx, one_weight)
        per_expert[int(eid)] = out_flat.reshape(B, L, H)
    return per_expert
```

(Reuse whatever exact shape conventions the Phase 3 implementation committed — don't accidentally change other aspects. The only change is the `slot = global_to_local[...]` lookup when `held` is present.)

- [ ] **Step 4: Run — expect pass**

Run: `uv run pytest -m slow tests/test_partial_load_run_selected.py -v`
Expected: pass. `mx.array_equal` holds.

- [ ] **Step 5: Phase 3/4 regression**

Run:
```
uv run pytest -m slow tests/test_moe_run_experts.py tests/test_moe_split_equivalence.py tests/test_expert_orchestrator.py -v
```
Expected: all pass. When `held_ids_per_layer` is empty (the `loaded_model` fixture case), `global_to_local` is None and behavior is identical to Phase 3.

- [ ] **Step 6: Lint + types**

```
uv run ruff check src/model_shard/moe.py tests/test_partial_load_run_selected.py
uv run mypy src/model_shard/moe.py
```

- [ ] **Step 7: Commit**

```bash
git add src/model_shard/moe.py tests/test_partial_load_run_selected.py
git commit -m "Phase 5a: run_selected_experts — global->local slot translation"
```

---

## Task 6: `KeyError` on unknown global id

**Files:**
- Create: `tests/test_partial_load_missing_expert_raises.py`

This test exercises the Task 5 error path explicitly.

- [ ] **Step 1: Write the test**

```python
"""run_selected_experts must raise KeyError when given a global id
not in the shard's held_ids_per_layer[layer_idx]."""

from __future__ import annotations

import mlx.core as mx
import pytest

from model_shard.mlx_engine import load_model_partial
from model_shard.moe import run_selected_experts


@pytest.mark.slow
def test_run_selected_experts_unknown_global_id_raises() -> None:
    held = {15: [0, 3, 6]}
    lm = load_model_partial("mlx-community/gemma-4-26b-a4b-it-4bit", held)
    try:
        h = mx.random.normal((1, 2, lm.text_model.config.hidden_size)).astype(mx.bfloat16)
        with pytest.raises(KeyError, match="expert 42 not held on this shard"):
            run_selected_experts(lm, h, layer_idx=15, expert_ids=[42])
    finally:
        del lm
        mx.metal.clear_cache()


@pytest.mark.slow
def test_run_selected_experts_held_layer_unaffected_elsewhere() -> None:
    """If layer 15 is subset-loaded but layer 20 is not, requests for experts
    on layer 20 still work with any global id."""
    held = {15: [0, 3]}
    lm = load_model_partial("mlx-community/gemma-4-26b-a4b-it-4bit", held)
    try:
        h = mx.random.normal((1, 2, lm.text_model.config.hidden_size)).astype(mx.bfloat16)
        # Layer 20 has no slice; global id 99 is still valid (full stack).
        out = run_selected_experts(lm, h, layer_idx=20, expert_ids=[99])
        assert 99 in out
    finally:
        del lm
        mx.metal.clear_cache()
```

- [ ] **Step 2: Run — both pass**

Run: `uv run pytest -m slow tests/test_partial_load_missing_expert_raises.py -v`
Expected: 2 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/test_partial_load_missing_expert_raises.py
git commit -m "Phase 5a: test — unknown global id on sliced layer raises KeyError"
```

---

## Task 7: Slow bit-exact per-expert test (comprehensive)

**Files:**
- Create: `tests/test_partial_load_bit_exact_per_expert.py`

Task 5's test proved bit-exactness for a single expert (id 3). Task 7 generalizes: for every held expert, output matches.

- [ ] **Step 1: Write the test**

```python
"""Bit-exact per-expert equivalence between full-loaded and sliced model."""

from __future__ import annotations

import mlx.core as mx
import pytest

from model_shard.mlx_engine import load_model_partial
from model_shard.moe import run_selected_experts


@pytest.mark.slow
def test_every_held_expert_matches_full_model(loaded_model) -> None:
    lm_full = loaded_model
    held_ids = [0, 3, 6, 9, 12, 15, 42, 127]
    lm_part = load_model_partial(
        "mlx-community/gemma-4-26b-a4b-it-4bit",
        {15: held_ids},
    )
    try:
        h = mx.random.normal(
            (1, 5, lm_full.text_model.config.hidden_size)
        ).astype(mx.bfloat16)

        out_full = run_selected_experts(lm_full, h, layer_idx=15, expert_ids=held_ids)
        out_part = run_selected_experts(lm_part, h, layer_idx=15, expert_ids=held_ids)

        for eid in held_ids:
            mx.eval(out_full[eid], out_part[eid])
            assert mx.array_equal(out_full[eid], out_part[eid]), (
                f"expert {eid}: bit-exact failure; max abs diff = "
                f"{mx.max(mx.abs(out_full[eid] - out_part[eid])).item()}"
            )
    finally:
        del lm_part
        mx.metal.clear_cache()
```

- [ ] **Step 2: Run — expect pass**

Run: `uv run pytest -m slow tests/test_partial_load_bit_exact_per_expert.py -v`
Expected: pass.

**If it fails:** the slice math is wrong. Most likely causes:
- `mx.take(full, ..., axis=0)` on a quantized tensor doesn't slice scales/biases/weight consistently. Inspect each of `layer.experts.switch_glu.gate_proj.{weight, scales, biases}` and confirm all three were sliced identically.
- The `mx.metal.clear_cache()` released a still-in-use tensor and MLX silently re-reads garbage. Remove the `clear_cache()` call and re-run; if it now passes, the cache-clear is the problem.

- [ ] **Step 3: Commit**

```bash
git add tests/test_partial_load_bit_exact_per_expert.py
git commit -m "Phase 5a: bit-exact per-expert — sliced model == full model"
```

---

## Task 8: Slow split-equivalence under sliced load

**Files:**
- Create: `tests/test_partial_load_split_equivalence.py`

This is the capstone proof. Three sliced LoadedModels (mod-3 partition at layer 15), atomic full model. The split reconstruction via `run_attention_and_route + run_selected_experts + run_shared_expert + aggregate_experts + outer ops` must match atomic layer 15 bit-exactly — same as Phase 3 Task 9, but now each expert is only available on its owner's sliced model.

- [ ] **Step 1: Write the test**

```python
"""Load-bearing Phase 5a proof:

Three sliced LoadedModels (mod-3 at layer 15) + one full LoadedModel.
For each token, top-k ids are partitioned by mod-3 owner, each sliced
LM computes its share of expert outputs, aggregation runs, outer ops run.
Result must match atomic layer 15 on the full model bit-for-bit.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from model_shard.mlx_engine import embed_tokens, load_model_partial, make_cache, make_masks
from model_shard.moe import (
    aggregate_experts,
    run_attention_and_route,
    run_selected_experts,
    run_shared_expert,
)


def _ids_mod3(r: int) -> list[int]:
    return [e for e in range(128) if e % 3 == r]


@pytest.mark.slow
def test_three_sliced_shards_compose_bit_exact(loaded_model) -> None:
    lm_full = loaded_model
    HF = "mlx-community/gemma-4-26b-a4b-it-4bit"
    lm_shards = [
        load_model_partial(HF, {15: _ids_mod3(0)}),
        load_model_partial(HF, {15: _ids_mod3(1)}),
        load_model_partial(HF, {15: _ids_mod3(2)}),
    ]
    try:
        layer_idx = 15
        tokens = mx.array([[1, 42, 99, 7, 13, 256, 500]])

        # Atomic on the full model: run layers 0..14, then layer 15 atomically.
        h_atom = embed_tokens(lm_full, tokens)
        cache_atom = make_cache(lm_full)
        gm, sm = make_masks(lm_full, h_atom, cache_atom)
        tm = lm_full.text_model
        for i in range(layer_idx):
            layer = tm.layers[i]
            c = cache_atom[tm.layer_idx_to_cache_idx[i]]
            mask = gm if layer.layer_type == "full_attention" else sm
            h_atom = layer(h_atom, mask, c, per_layer_input=None)
        layer15 = tm.layers[layer_idx]
        c15 = cache_atom[tm.layer_idx_to_cache_idx[layer_idx]]
        mask15 = gm if layer15.layer_type == "full_attention" else sm
        out_atomic = layer15(h_atom, mask15, c15, per_layer_input=None)

        # Split across 3 sliced shards. Attention+router+dense branch runs
        # on shard 0 (matches Phase 3's pattern where mid owns these).
        lm_router = lm_shards[1]  # mid-equivalent — holds {1,4,...} plus runs attention
        # Replay layers 0..14 on the router shard too (same operator path).
        h_split = embed_tokens(lm_router, tokens)
        cache_split = make_cache(lm_router)
        gm2, sm2 = make_masks(lm_router, h_split, cache_split)
        for i in range(layer_idx):
            layer = lm_router.text_model.layers[i]
            c = cache_split[lm_router.text_model.layer_idx_to_cache_idx[i]]
            mask = gm2 if layer.layer_type == "full_attention" else sm2
            h_split = layer(h_split, mask, c, per_layer_input=None)

        post_attn, top_k_ids, top_k_weights = run_attention_and_route(
            lm_router, h_split, layer_idx, cache_split, (gm2, sm2)
        )
        mx.eval(top_k_ids)

        # Collect expert outputs across the 3 shards.
        all_ids = sorted({int(eid) for eid in top_k_ids.reshape(-1).tolist()})
        expert_outputs: dict[int, mx.array] = {}
        for shard_lm in lm_shards:
            held = set(shard_lm.held_ids_per_layer.get(layer_idx, ()))
            mine = [e for e in all_ids if e in held]
            if not mine:
                continue
            contribution = run_selected_experts(shard_lm, post_attn, layer_idx, mine)
            expert_outputs.update(contribution)

        shared_out = run_shared_expert(lm_router, post_attn, layer_idx)
        post_ffn_ln_2 = lm_router.text_model.layers[layer_idx].post_feedforward_layernorm_2

        # Per-position aggregate + outer ops — same pattern as Phase 3 Task 9.
        h1_plus_h2 = mx.zeros_like(post_attn)
        for b in range(top_k_ids.shape[0]):
            for l in range(top_k_ids.shape[1]):
                ids_l = [int(x) for x in top_k_ids[b, l].tolist()]
                weights = top_k_weights[b : b + 1, l : l + 1, :]
                per_pos = {
                    eid: expert_outputs[eid][b : b + 1, l : l + 1, :] for eid in ids_l
                }
                per_pos_shared = shared_out[b : b + 1, l : l + 1, :]
                agg = aggregate_experts(
                    per_pos, ids_l, weights, per_pos_shared, post_ffn_ln_2
                )
                h1_plus_h2 = mx.concatenate(
                    [h1_plus_h2[:, :l, :], agg, h1_plus_h2[:, l + 1 :, :]],
                    axis=1,
                ) if h1_plus_h2.shape[1] > 1 else agg

        layer_router = lm_router.text_model.layers[layer_idx]
        out_split = layer_router.post_feedforward_layernorm(h1_plus_h2)
        out_split = post_attn + out_split
        if layer_router.layer_scalar is not None:
            out_split = out_split * layer_router.layer_scalar

        mx.eval(out_atomic, out_split)
        assert mx.array_equal(out_atomic, out_split), (
            f"sliced-split != atomic; max abs diff = "
            f"{mx.max(mx.abs(out_atomic - out_split)).item()}"
        )
    finally:
        for lm in lm_shards:
            del lm
        mx.metal.clear_cache()
```

- [ ] **Step 2: Run — expect pass**

Run: `uv run pytest -m slow tests/test_partial_load_split_equivalence.py -v`
Expected: pass.

**Memory note:** this test holds FOUR LoadedModels simultaneously (1 full + 3 sliced). On the 128 GB M5 this is fine (~14 + 3×7.5 = ~36 GB resident). If memory is tight, split into three separate test functions, each loading one shard.

- [ ] **Step 3: Commit**

```bash
git add tests/test_partial_load_split_equivalence.py
git commit -m "Phase 5a: split-equivalence under sliced load (3 shards mod-3)"
```

---

## Task 9: Node integration — `ENABLE_PARTIAL_LOAD` + conditional load

**Files:**
- Modify: `src/model_shard/node.py`
- Create: `tests/test_node_partial_load_wiring.py`

- [ ] **Step 1: Write the failing test**

```python
"""Node picks load_model vs load_model_partial based on ENABLE_PARTIAL_LOAD."""

from __future__ import annotations

import random
import socket
from typing import Any

import pytest

from model_shard.node import Node
from model_shard.shard_map import NodeAddress, ShardMap, ShardSpec


def _free_port() -> int:
    for _ in range(100):
        p = random.randint(30000, 60000)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", p))
            return p
        except OSError:
            continue
    raise RuntimeError("no free port")


@pytest.mark.slow
def test_node_partial_load_active_when_enabled_and_moe_experts_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENABLE_PARTIAL_LOAD", "true")
    monkeypatch.setenv("ENABLE_GOSSIP", "false")
    port = _free_port()
    spec = ShardSpec(
        shard_id="solo",
        address=NodeAddress("127.0.0.1", port),
        start_layer=0,
        end_layer=30,
        moe_experts={15: (0, 3, 6, 9)},
    )
    peer_port = _free_port()
    peer = ShardSpec(
        shard_id="peer",
        address=NodeAddress("127.0.0.1", peer_port),
        start_layer=30,
        end_layer=30,
    )
    sm = ShardMap({"solo": spec, "peer": peer})
    node = Node(
        shard=spec, shard_map=sm,
        loaded_model=None,   # forced reload via env
        total_layers=30,
    )
    try:
        # After construction, the loaded model should have the sliced layer-15.
        lm = node._lm
        assert lm.held_ids_per_layer == {15: (0, 3, 6, 9)}
        layer15 = lm.text_model.layers[15]
        assert layer15.experts.switch_glu.gate_proj.weight.shape[0] == 4
    finally:
        node.shutdown()
```

**Note on `loaded_model=None`:** The existing `Node.__init__` signature takes `loaded_model: Any`. Phase 1 tests pass a pre-loaded `loaded_model` via the session fixture. For the Phase 5a integration to work, `Node.__init__` either (a) accepts `loaded_model=None` and loads via `_partial_load_enabled()` branching, OR (b) the caller always loads externally and `Node` just accepts it. Option (a) is cleaner for this test; if node.py currently always expects a live `loaded_model`, adjust either the signature or the test.

If changing `Node.__init__` is too invasive, alternative: have the test instantiate the model explicitly via `load_model_partial` and pass it in, then just assert `node._lm.held_ids_per_layer == {15: (0,3,6,9)}`.

- [ ] **Step 2: Run — expect failure**

- [ ] **Step 3: Modify `src/model_shard/node.py`**

Add near the top (with other helpers):

```python
def _partial_load_enabled() -> bool:
    return os.environ.get("ENABLE_PARTIAL_LOAD", "false").lower() in ("1", "true", "yes")
```

In `Node.__init__`, find the line that stores `self._lm = loaded_model`. Replace with:

```python
        if loaded_model is None and _partial_load_enabled() and shard.moe_experts:
            from model_shard.mlx_engine import load_model_partial
            held = {k: list(v) for k, v in shard.moe_experts.items()}
            self._lm = load_model_partial(
                "mlx-community/gemma-4-26b-a4b-it-4bit",
                held,
            )
        else:
            self._lm = loaded_model
```

If `Node.__init__` already computes its own `hf_id` from a config or constant, reuse that instead of hardcoding the string.

- [ ] **Step 4: Run — expect pass**

- [ ] **Step 5: Commit**

```bash
git add src/model_shard/node.py tests/test_node_partial_load_wiring.py
git commit -m "Phase 5a: Node conditionally invokes load_model_partial"
```

---

## Task 10: Phase 3/4 regression pass (default-OFF)

No code changes — just a verification.

- [ ] **Step 1: Run regression**

```
uv run pytest -m slow tests/test_moe_split_equivalence.py tests/test_expert_rpc_handler.py tests/test_expert_orchestrator.py tests/test_expert_orchestrator_timeout.py tests/test_expert_orchestrator_observer.py tests/test_routing_correctness.py tests/test_expert_rpc_load_shift.py -v
```

Expected: all pass. With `ENABLE_PARTIAL_LOAD=false` (default), every code path that checked `lm.held_ids_per_layer` sees an empty dict and behaves exactly as Phase 3/4.

- [ ] **Step 2: If anything regresses**

Most likely suspects:
- `run_selected_experts` sees `held = lm.held_ids_per_layer.get(...)` returning an empty dict — `.get(15)` returns `None`, so `global_to_local = None` and the path reverts to Phase 3 behavior. Confirm this with a print.
- `LoadedModel` construction somewhere now fails because the caller doesn't pass `held_ids_per_layer`. Default factory should prevent this; check for positional-arg callers.

- [ ] **Step 3: No commit if regression clean**

---

## Task 11: Slow Tier 1 E2E under `ENABLE_PARTIAL_LOAD=true`

**Files:**
- Create: `tests/test_partial_load_tier1_e2e.py`

- [ ] **Step 1: Write the test**

```python
"""Tier 1 E2E with partial-load enabled: each of 3 in-process nodes runs
with a sliced layer-15 expert subset, total coverage = all 128 experts,
tokens must match the Phase 1 reference manifest exactly."""

from __future__ import annotations

import json
import os
import socket
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from model_shard.client import Client
from model_shard.mlx_engine import load_model_partial
from model_shard.node import Node
from model_shard.shard_map import NodeAddress, ShardMap, ShardSpec

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "artifacts" / "ref" / "manifest.json"
MAX_TOK = 32


def _ids_mod3(r: int) -> tuple[int, ...]:
    return tuple(e for e in range(128) if e % 3 == r)


@pytest.fixture(scope="module")
def three_node_pipeline_partial_load() -> Iterator[Any]:
    import random
    os.environ["ENABLE_EXPERT_SHARD"] = "true"
    os.environ["ENABLE_PARTIAL_LOAD"] = "true"

    def free_port() -> int:
        for _ in range(100):
            p = random.randint(30000, 60000)
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
        raise RuntimeError("no free port")

    ports = [free_port() for _ in range(3)]
    specs = [
        ShardSpec(
            shard_id="layer_0-10",
            address=NodeAddress("127.0.0.1", ports[0]),
            start_layer=0, end_layer=10,
            moe_experts={15: _ids_mod3(0)},
        ),
        ShardSpec(
            shard_id="layer_10-20",
            address=NodeAddress("127.0.0.1", ports[1]),
            start_layer=10, end_layer=20,
            moe_experts={15: _ids_mod3(1)},
        ),
        ShardSpec(
            shard_id="layer_20-30",
            address=NodeAddress("127.0.0.1", ports[2]),
            start_layer=20, end_layer=30,
            moe_experts={15: _ids_mod3(2)},
        ),
    ]
    shard_map = ShardMap({s.shard_id: s for s in specs})
    # Each node loads its own sliced model.
    nodes = {
        spec.shard_id: Node(
            shard=spec, shard_map=shard_map,
            loaded_model=None,  # force partial load
            total_layers=30,
        )
        for spec in specs
    }
    threads = [
        threading.Thread(target=n.serve_forever, daemon=True)
        for n in nodes.values()
    ]
    for t in threads:
        t.start()

    from tests.conftest import _wait_for_listening
    for s in specs:
        _wait_for_listening(s.address.host, s.address.port)

    try:
        from tests.conftest import DistributedCluster
        yield DistributedCluster(shard_map=shard_map, nodes_by_id=nodes)
    finally:
        for n in nodes.values():
            n.shutdown()
        for t in threads:
            t.join(timeout=3.0)


@pytest.mark.slow
@pytest.mark.parametrize("prompt_idx", range(5))
def test_tier1_under_partial_load(
    three_node_pipeline_partial_load: Any,
    prompt_idx: int,
) -> None:
    if not MANIFEST.exists():
        pytest.skip("reference manifest missing")
    manifest = json.loads(MANIFEST.read_text())
    record = manifest["prompts"][prompt_idx]
    prompt_tokens = list(record["prompt_tokens"])
    expected = list(record["generated_tokens"])[:MAX_TOK]

    head = three_node_pipeline_partial_load.shard_map.lookup("layer_0-10")
    got = Client(head_address=head.address).generate(prompt_tokens, max_new_tokens=MAX_TOK)
    assert got == expected, (
        f"prompt {prompt_idx}: distributed {got[:10]}... != reference {expected[:10]}..."
    )
```

- [ ] **Step 2: Run**

Run: `uv run pytest -m slow tests/test_partial_load_tier1_e2e.py -v`

Expected: 5 pass (one per prompt). Each node loads ~7.5 GB instead of 14 GB.

**Memory consideration:** 3 nodes × 7.5 GB = ~22.5 GB. Plus the `loaded_model` session fixture for other tests may still be in memory if this test runs after them. If memory exceeds the machine's comfort zone, run this test in isolation with `-p no:cacheprovider` to avoid fixture reuse.

- [ ] **Step 3: If any prompt diverges**

The Phase 1 reference was generated on the full model. A sliced run that produces identical tokens is the proof Phase 5a works. If a prompt diverges:
- Task 7's bit-exact test should have caught per-expert drift. Re-run it.
- Check that the three shards' moe_experts together cover all 128 experts (the `_ids_mod3(0/1/2)` helper must partition {0..127}). Assert this in the fixture.
- If tokens diverge only late in the sequence, suspect KV-cache state — each node has its own KV cache per its `[start, end)` range, and the partial-load doesn't touch attention weights.

- [ ] **Step 4: Commit**

```bash
git add tests/test_partial_load_tier1_e2e.py
git commit -m "Phase 5a: Tier 1 E2E bit-exact under ENABLE_PARTIAL_LOAD=true"
```

---

## Task 12: Final acceptance — lint, types, tests, README, memory

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

Expected: all existing + new fast tests pass (partial_load slice-math adds 6 fast tests).

- [ ] **Step 3: Phase 5a slow suite**

```
uv run pytest -m slow tests/test_partial_load_slice_math.py tests/test_partial_load_smoke.py \
  tests/test_partial_load_run_selected.py tests/test_partial_load_missing_expert_raises.py \
  tests/test_partial_load_bit_exact_per_expert.py tests/test_partial_load_split_equivalence.py \
  tests/test_node_partial_load_wiring.py tests/test_partial_load_tier1_e2e.py -v
```

Expected: all Phase 5a slow tests pass.

- [ ] **Step 4: Phase 3/4 regression**

```
uv run pytest -m slow tests/test_moe_split_equivalence.py tests/test_expert_rpc_handler.py \
  tests/test_expert_orchestrator.py tests/test_expert_orchestrator_timeout.py \
  tests/test_expert_orchestrator_observer.py tests/test_routing_correctness.py -v
```

Expected: all pass.

- [ ] **Step 5: README update**

Append:

```markdown
## Phase 5a status: Partial Expert Weight Loading — complete

A node can now load only the routed experts listed in its shard's
`moe_experts` YAML instead of the full 128-expert stack per layer.
Opt-in via `ENABLE_PARTIAL_LOAD=true`. Resident memory per shard drops
from ~14 GB to chassis (~4.5 GB) + `k/128 × 9 GB` for routed experts,
which is the unlock for eventual 24 GB-VRAM deployments. Correctness
is proven by `tests/test_partial_load_bit_exact_per_expert.py` (per
expert bit-exact vs full load) and `tests/test_partial_load_split_equivalence.py`
(three mod-3 sliced shards compose bit-exact to atomic layer 15).
See `docs/superpowers/specs/2026-04-17-phase5a-partial-expert-loading-design.md`.
```

Commit:

```bash
git add README.md
git commit -m "Phase 5a complete: partial expert weight loading"
```

- [ ] **Step 6: Update memory**

Tell the operator: Phase 5a is complete. Update `~/.claude/projects/-Users-lukechang-Github-model-shard/memory/project_gossip_moe.md` to mark Phase 5a done and Phase 5b (dynamic migration + heat tracking + decode-loop hang fix, built on 5a's per-expert loading) next. Phase 5b requires a fresh brainstorming cycle.

---

## Self-Review

### 1. Spec coverage

| Spec § | Implemented in tasks |
|---|---|
| D1 per-shard partial load | Task 3, 9 |
| D2 custom safetensors slice-reader (option B, revised to post-load slice) | Task 1, 3 |
| D3 `moe_experts` YAML semantic extension | Task 9 |
| D4 global→local index remap in `run_selected_experts` | Task 5 |
| D5 KeyError on unknown id | Task 6 |
| D6 bit-exact correctness | Tasks 7, 8 |
| D7 non-goals (no migration, no streaming) | — (by omission) |
| D8 `ENABLE_PARTIAL_LOAD` env var | Task 9 |
| §3.1 `partial_load.py` | Tasks 2, 3 |
| §3.2 `LoadedModel.held_ids_per_layer` | Task 3 |
| §3.3 `run_selected_experts` translation | Task 5 |
| §3.4 node integration | Task 9 |
| §5 testing | Tasks 2, 5-8, 10, 11 |
| §6 acceptance | Task 12 |
| §7 open questions → resolved | Task 1 |

### 2. Placeholder scan

- Task 3's implementation uses `mx.take(...)` rather than `_slice_stacked_by_axis0` because at that point `mx.array` is what we hold (not numpy). Clarified inline — the fast unit test covers numpy; the production path uses `mx.take`. Both are axis-0 row selection; semantics identical.
- Task 9 Step 1 discusses a potential `Node.__init__` signature change (`loaded_model: Any = None`). If the current signature doesn't accept None, the engineer must choose between (a) adjusting the signature or (b) loading externally and passing in. Both are named; no placeholder.
- Task 1 is recon with concrete commands; no placeholder in the plan. Task 1's output (revised §7 of the spec) is itself the "fill in" — this is the correct structure.

### 3. Type / name consistency

- `LoadedModel.held_ids_per_layer: dict[int, tuple[int, ...]]` — defined in Task 3, used in Tasks 5, 9, 11.
- `load_model_partial(hf_id: str, held_experts_per_layer: dict[int, list[int]])` — defined in Task 3, used in Tasks 5, 6, 7, 8, 9, 11.
- `_slice_stacked_by_axis0(arr, ids)` — Task 2 only; not called in production path (which uses `mx.take`).
- `_partial_load_enabled()` — Task 9; reads `ENABLE_PARTIAL_LOAD`.
- `run_selected_experts` signature unchanged across all tasks.

### 4. Scope check

Plan is one cohesive subsystem (partial load + id remap + proof). 12 tasks; ~3 mechanical (2, 4, 6), ~3 integration (3, 5, 9), ~4 slow tests (7, 8, 11, 12), 1 recon (1), 1 acceptance (12). Appropriate for one implementation cycle.

### 5. Memory and cost

Task 8's split-equivalence test holds 4 LoadedModels. On M5 128 GB: fine. On a lower-memory dev box: flag this as a concern — the test could be reorganized as three separate tests each loading one shard, with atomic comparisons streamed through an intermediate reference file.
