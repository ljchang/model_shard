# Phase 5a — Partial Expert Weight Loading

**Status:** draft, 2026-04-17
**Scope:** A node loads only the routed experts listed in its shard's `moe_experts` YAML, not the full 128-expert stack. Bit-exact against the full-loaded path. No migration, no streaming, no heat tracking — Phase 5b handles those.

## 1. Background & Decisions

### 1.1 Why now
Phase 1 through Phase 4 had every node load the full 14 GB model. Phase 3 introduced the `moe_experts` YAML to declare *which* experts a node serves; Phase 4 added multi-owner routing via power-of-two-choices. But resident memory per node stayed at 14 GB because weights were fully loaded. Phase 5a changes that: if the YAML says a shard only serves experts {0, 3, 6, ...}, the node resident set becomes the chassis plus just those experts.

This is a precondition for meaningful dynamic migration (Phase 5b) — you cannot meaningfully "stream an expert to a target node" unless the target doesn't already have it.

### 1.2 Weight layout (observed in mlx-community/gemma-4-26b-a4b-it-4bit)
Each MoE layer has three stacked projection tensors for the routed experts:
- `language_model.model.layers.<L>.experts.switch_glu.gate_proj.{weight, scales, biases}` — leading dim 128
- `...up_proj.{weight, scales, biases}` — leading dim 128
- `...down_proj.{weight, scales, biases}` — leading dim 128

Plus `router.per_expert_scale` (128-element). Chassis weights (attention, `mlp` dense branch, all norms, embeddings, LM head, `router.proj`) have no per-expert dimension and load identically on every node.

### 1.3 Decisions
- **D1 Load strategy.** Custom safetensors slice-reader (option B from brainstorming). For each stacked expert tensor in `experts.switch_glu.*`, read only the rows corresponding to `held_ids` along axis 0. All other keys load unchanged via `mx.load`. Peak memory is chassis + held-expert-subset from the start.
- **D2 Config source.** Existing `ShardSpec.moe_experts: dict[int, tuple[int, ...]]`. **New semantic:** listed means "only these experts loaded for this layer"; absent means "full 128 stack loaded." This is a superset of Phase 3/4 semantics (where `moe_experts` declared ownership for routing) — now it also controls what's resident.
- **D3 Global→local index remapping.** Happens in `moe.run_selected_experts`, not inside `Experts.__call__`. Each layer keeps `held_ids_per_layer[layer_idx] = [...]`; callers pass global expert ids; the function translates to compact local slots before invoking `layer.experts(h, indices, weights)`. The stock `Experts` and `SwitchLinear` modules are untouched.
- **D4 Correctness bar.** Bit-exact. Two proofs:
  1. **Per-expert equivalence:** for any held expert `e`, the sliced model's `run_selected_experts(..., [e])` output is `mx.array_equal` to the full model's `run_selected_experts(..., [e])` on the same input.
  2. **Distributed split-equivalence under sliced load:** the Phase 3 Task 9 proof, repeated with three sliced `LoadedModel` instances (each holding its mod-3 partition), produces output identical to atomic layer 15 on the full model.
- **D5 Routing safety.** If `run_selected_experts` receives a global id not in `held_ids_per_layer[layer_idx]`, raise `KeyError`. In practice, Phase 4's orchestrator filters by ownership so this never fires — but it guards against silent routing-logic drift.
- **D6 Scope.** Startup-time partial load only. No migration, no streaming, no runtime reload.
- **D7 Non-goals.** Dynamic shard-map updates, runtime memory freeing, per-expert safetensors key layout (upstream format stays stacked).
- **D8 Migration / rollback.** New env var `ENABLE_PARTIAL_LOAD=false` (default). When false, `Node.__init__` calls the existing `load_model` and resident memory is unchanged. When true AND `shard.moe_experts` is non-empty, calls `load_model_partial` with the YAML subset.

## 2. Components

### 2.1 `src/model_shard/partial_load.py` (new)

```python
def load_model_partial(
    hf_id: str,
    held_experts_per_layer: dict[int, list[int]],
) -> LoadedModel
```

- Resolves `hf_id` via `huggingface_hub.snapshot_download` (same path mlx-vlm uses).
- Walks the safetensors files. For each key:
  - If it matches `language_model.model.layers.<L>.experts.switch_glu.*` AND `L` has a held_ids entry: read only rows `held_ids[L]` of the stacked tensor.
  - Else: read in full via `mx.load`.
- Constructs the mlx-vlm Gemma 4 model shell (same config, same `num_experts=128`).
- Before calling `model.load_weights(weights)`, replaces the stacked-tensor entries in the `weights` dict with the sliced tensors.
- Walks the model post-load, for each held layer mutates `layer.experts.switch_glu.<proj>.weight` to the compact `(k, out, in)` tensor (and `scales` / `biases` likewise).
- Returns a `LoadedModel` with a new field `held_ids_per_layer: dict[int, tuple[int, ...]]`.

### 2.2 `src/model_shard/mlx_engine.py`

- `LoadedModel` gains `held_ids_per_layer: dict[int, tuple[int, ...]] = field(default_factory=dict)`.
- New helper `load_model_partial(hf_id, held_experts_per_layer)` delegates to `partial_load.load_model_partial`. Existing `load_model` unchanged; its `LoadedModel` has `held_ids_per_layer = {}`.

### 2.3 `src/model_shard/moe.py` — `run_selected_experts`

Before calling `layer.experts(h_normed, indices, weights)`:

```python
held = lm.held_ids_per_layer.get(layer_idx)
if held:
    global_to_local = {gid: li for li, gid in enumerate(held)}
    try:
        local_ids = [global_to_local[eid] for eid in expert_ids]
    except KeyError as e:
        missing = e.args[0]
        raise KeyError(f"expert {missing} not held on this shard") from e
    indices = mx.array([[local_ids[i]] for i in range(len(local_ids))])  # shape TBD; match current form
else:
    # No slicing active — use global ids directly, as today.
    ...
```

The exact shape of `indices` matches what `SwitchLinear.__call__` expects (single-slot per token with the K=1 strategy from Phase 3 Task 7). Plan task 1 confirms the shape wiring.

### 2.4 `src/model_shard/node.py`

In `Node.__init__`, replace the unconditional `load_model(hf_id)` call with:

```python
held = dict(shard.moe_experts)
if _partial_load_enabled() and held:
    lm = load_model_partial(hf_id, {k: list(v) for k, v in held.items()})
else:
    lm = load_model(hf_id)
```

With `_partial_load_enabled()` reading `ENABLE_PARTIAL_LOAD` env var (default false).

When a node serves an `ExpertRequest`, `_handle_expert_request` → `moe.run_selected_experts` handles remapping via D3. No direct `node.py` change beyond the constructor.

### 2.5 Config

No change to `config/shards.yaml`. The existing overlapping moe_experts from Phase 4 works: each shard lists its ~44 held ids for layer 15, plus full stacks for layers 0-14 and 16-29 (which are absent from moe_experts, so they stay full-loaded).

## 3. Wire Protocol

**No changes.** Phase 5a is startup-time only.

## 4. Memory Model

| Component | Size (4-bit bf16) |
|---|---|
| Chassis (attention × 30, dense mlp × 30, norms, embed, LM head, router) | ~4.5 GB |
| Full routed-expert stack (30 layers × 128 experts × 3 projections) | ~9 GB |
| Sliced expert stack at 43/128 per layer × 30 layers | ~3 GB |

A shard holding 43/128 experts per layer has resident ~7.5 GB instead of 14 GB. On the M5 128 GB this is moot; on future 24 GB 3090s it's the unlock.

## 5. Testing Strategy

### 5.1 Fast tests
- `test_partial_load_slice_math.py` — synthetic stacked tensors, unit test the axis-0 slice helper in isolation.

### 5.2 Slow tests
- `test_partial_load_bit_exact_per_expert.py` — load `lm_full` and `lm_sliced` (with held_ids = {0, 3, 6, 9}). For each held id, call `run_selected_experts(lm, h, 15, [id])` and assert `mx.array_equal`.
- `test_partial_load_split_equivalence.py` — three sliced LoadedModels (mod-3 partition at layer 15), each runs its share of top-k experts; aggregate; compare to atomic layer 15 on full model via `mx.array_equal`.
- `test_partial_load_missing_expert_raises.py` — call `run_selected_experts` with a global id not in `held_ids`, assert `KeyError`.

### 5.3 Regression
- Every Phase 3/4 slow test must still pass with `ENABLE_PARTIAL_LOAD=false` (default).
- With `ENABLE_PARTIAL_LOAD=true`, the Tier 1 E2E test (5 prompts) must still produce tokens bit-exact to the Phase 1 reference.

## 6. Acceptance

1. `ruff check`, `mypy` clean.
2. Fast suite green.
3. All new Phase 5a slow tests pass.
4. `ENABLE_PARTIAL_LOAD=false` (default) → Phase 3/4 slow suite unchanged.
5. `ENABLE_PARTIAL_LOAD=true` with Phase 4's config/shards.yaml → Tier 1 E2E still bit-exact.
6. README updated with a Phase 5a status paragraph.

## 7. Open Technical Questions (to resolve during Task 1 recon)

- **Does mlx-vlm's model constructor work when we replace `SwitchLinear.weight` post-init with a compact `(k, out, in)`?** `SwitchLinear` reads `num_experts` from `weight.shape[0]`, so shrinking should be fine, but assertions elsewhere in the layer (e.g. inside `Experts.__call__` checking `top_k_indices` bounds) might reference the original 128. Task 1 reads the relevant mlx-vlm source.
- **Does `mx.load(safetensors_file)` support partial key loading?** If not, we fall back to `safetensors.safe_open(...)` + `f.get_slice(key)[rows, :, :]` and then `mx.array(sliced_np)`. This is the most likely path; confirm during Task 1.
- **Quantized tensor slicing correctness:** `scales` / `biases` share the leading dim 128 with `weight` and slice identically; but the quant group alignment might care. We verify by round-tripping one expert's output through `gather_mm` and comparing to the atomic layer's computation for that expert.

## 8. References

- Phase 3 spec: `docs/superpowers/specs/2026-04-16-phase3-expert-sharding-design.md`
- Phase 4 spec: `docs/superpowers/specs/2026-04-16-phase4-load-aware-routing-design.md`
- mlx-vlm loader: `.venv/lib/python3.13/site-packages/mlx_vlm/utils.py` (`load_model`)
- mlx_lm switch layers: `.venv/lib/python3.13/site-packages/mlx_lm/models/switch_layers.py` (`SwitchLinear`, `SwitchGLU`, `QuantizedSwitchLinear`)
- Gemma 4 model: `.venv/lib/python3.13/site-packages/mlx_vlm/models/gemma4/language.py` (`Experts`, `Router`, `DecoderLayer`)
- Spec §10 (dynamic migration — Phase 5b scope)
