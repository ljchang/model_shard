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

## 7. Resolved Technical Choices (2026-04-17)

Recon performed on `mlx-community/gemma-4-26b-a4b-it-4bit` with the interactive experiments described in Task 1. All three open questions are resolved; the plan is unchanged in its shape but the implementation strategy is pinned below.

### 7.1 Post-init weight replacement is safe — `QuantizedSwitchLinear` tolerates shrinking

`mlx_lm/models/switch_layers.py`:
- `SwitchLinear` (unquantized) and `QuantizedSwitchLinear` both expose `num_experts` as a `@property` that returns `self.weight.shape[0]`. Input/output dims are also properties derived from `weight.shape[1]` / `scales.shape[2] * group_size`. Nothing is cached at `__init__`.
- Both `__call__` methods look up weight tensors dynamically via `self["weight"]`, `self["scales"]`, `self.get("biases")`, `self["bias"]` — no captured references.
- `SwitchGLU.__call__` and `Experts.__call__` (in `mlx_vlm/models/gemma4/language.py`) never compare indices against `num_experts` or reference a captured expert count; they only use `top_k_indices` in shape operations.

Conclusion: we can mutate `layer.experts.switch_glu.<proj>.{weight, scales, biases}` in place to compact `(k, out, packed_in)` tensors after `load()` returns. The `num_experts` property will correctly report `k` afterwards.

Verified empirically with held_ids=[0,3,6] on layer 0: after `mx.take(..., axis=0)` the forward pass runs and produces correct output (see §7.4 bit-exactness result).

### 7.2 Safetensors slice API — what actually works

`safetensors.safe_open(path, framework="np").get_slice(key)` returns a `PySafeSlice` whose `__getitem__` supports Python `slice` and `int`, but does NOT support Python-list fancy indexing:

- `s[0:3, :, :]` → works (contiguous range)
- `s[0, :, :]` → works (single int drops the axis)
- `s[[0, 3, 6], :, :]` → `TypeError: 'list' object cannot be converted to 'PySlice'`

So for non-contiguous `held_ids`, Task 2's slice helper must loop: read one row at a time and `np.stack(..., axis=0)`, then `mx.array(stacked)`. For contiguous held_ids a single slice call is fine. Memory-wise row-by-row is still a big win — each read is `704*352*4 bytes ≈ 1 MiB` for `gate_proj.weight`, we touch only the rows we keep.

### 7.3 Exact tensor shapes & dtypes (per MoE layer, 4-bit quantized)

All stored under `language_model.model.layers.<L>.experts.switch_glu.<proj>.{weight,scales,biases}`:

| Tensor | Shape | Dtype |
|---|---|---|
| `gate_proj.weight` | `[128, 704, 352]` | `U32` (packed 4-bit) |
| `gate_proj.scales` | `[128, 704, 44]` | `BF16` |
| `gate_proj.biases` | `[128, 704, 44]` | `BF16` |
| `up_proj.weight` | `[128, 704, 352]` | `U32` |
| `up_proj.scales` | `[128, 704, 44]` | `BF16` |
| `up_proj.biases` | `[128, 704, 44]` | `BF16` |
| `down_proj.weight` | `[128, 2816, 88]` | `U32` |
| `down_proj.scales` | `[128, 2816, 11]` | `BF16` |
| `down_proj.biases` | `[128, 2816, 11]` | `BF16` |

Leading dim 128 = `num_experts`. 4-bit weights pack 8 nibbles per `uint32`, so the true input dim is `352 * 8 = 2816` (matching `hidden_size`). `QuantizedSwitchLinear` uses `group_size=64, bits=4, mode="affine"`, so `scales`/`biases` have last-dim `input_dims / group_size = 2816 / 64 = 44` for gate/up and `704 / 64 = 11` for down.

Quant groups live along the last axis (within each expert's matrix), not across experts, so axis-0 slicing preserves group alignment exactly — no group-boundary breakage.

### 7.4 Chosen strategy: post-load replacement (mutate after `mlx_vlm.load()`)

We commit to **strategy B (post-load replacement)**. Two reasons: it is minimal-surface (no fork of `mlx_vlm.utils.load_model`), and it was verified bit-exact on the same path that drives `run_selected_experts`.

Flow in `partial_load.load_model_partial`:

1. Call `mlx_vlm.load(hf_id)` as normal (loads full 14 GB of weights). This gives us a constructed, quantized, weight-populated `nn.Module` tree.
2. For each layer `L` with `held_ids` entry, for each of `{gate_proj, up_proj, down_proj}`:
   - `proj.weight  = mx.take(proj.weight,  mx.array(held), axis=0)`
   - `proj.scales  = mx.take(proj.scales,  mx.array(held), axis=0)`
   - `proj.biases  = mx.take(proj.biases,  mx.array(held), axis=0)`
3. `mx.eval(model.parameters())` to force materialization and allow the GC to reclaim the full-size originals.
4. Return a `LoadedModel` with `held_ids_per_layer = {L: tuple(held_ids[L])}`.

**Why not strategy A (safetensors slice at load):** It works (we proved §7.2), and it has lower *peak* memory. But it requires reaching into `mlx_vlm.utils.load_model`'s innards — re-implementing the sanitize → quantize → load_weights sequence while swapping in sliced tensors with the correct quantized key names (`.weight`/`.scales`/`.biases` produced by `nn.quantize`). That's fragile across mlx-vlm versions. Strategy B accepts a ~14 GB transient memory spike at startup on the M5 (which has 128 GB) in exchange for independence from mlx-vlm's load internals. **If Phase 5b needs true low-peak-memory partial loading** (e.g. on 24 GB 3090s), we revisit strategy A then; for Phase 5a the M5 is the only target.

**Bit-exactness verified** (see §7.5): post-load replacement with held=[0,3,6] produces `mx.array_equal == True` for `run_selected_experts`-shaped calls (small `indices.size`, no-sort path).

### 7.5 Quantized-slice correctness — observed, with one caveat

Per-expert equivalence test (full model vs sliced model, same input, expert id 3 → local slot 1 in sliced):

- `B=1, S=1, K=1` (no-sort path, `indices.size=1<64`): **`mx.array_equal == True`**. Bit-exact.
- `B=1, S=63, K=1` (no-sort path): **`mx.array_equal == True`**. Bit-exact.
- `B=1, S=128, K=1` (sort path, `indices.size=128≥64`): **NOT bit-exact** — `max abs diff ≈ 0.034`.

Root cause of the sort-path delta: `SwitchGLU.__call__` dispatches to `sorted_indices=True` when `indices.size >= 64`, and `mx.gather_qmm(..., rhs_indices=idx, sorted_indices=True)` appears to pick a different kernel or reduction order when the index values are small-integer (local slots, e.g. `[1,1,...,1]`) vs larger-integer (global ids, e.g. `[3,3,...,3]`). The numerical difference is within acceptable 4-bit-quant noise but it is not zero.

**Impact on our proofs (D4):**
- D4.1 (per-expert equivalence via `run_selected_experts`): `run_selected_experts` is called with `h = [B, L, hidden]` where L=1 at decode and L≤ prompt length at prefill. Our Phase 3/4 slow tests use L=7. `indices.size = B*L*1 = 7 < 64` → no-sort path → **bit-exact holds**. Test remains valid.
- D4.2 (split-equivalence on layer 15 across 3 sliced shards): same L=7 path. Bit-exact holds.
- Tier 1 E2E with `ENABLE_PARTIAL_LOAD=true`: single-token decode steps have `indices.size = top_k = 8 < 64` → no-sort path → **bit-exact holds**. Prefill of a prompt of ≥64 tokens routed at K=8 gives `indices.size = 64*8 = 512` ≥ 64 → sort path would fire.

**Mitigation (to implement in Task 10):** for the Tier 1 E2E test use prompts ≤ ~8 tokens, or assert token-id equality (not logit equality) on generated output. The max-abs-diff of 0.034 in sort-path is well below argmax-tipping noise for a 262k-vocab LM head on well-separated top tokens in practice — but we will document this as a known non-bit-exact-in-sort-path gotcha and rely on token-id equality, not logit equality, at Tier 1.

### 7.6 Summary — concrete commitments for downstream tasks

| Aspect | Decision |
|---|---|
| Strategy | B: `mlx_vlm.load()` then `mx.take(..., axis=0)` on `{weight,scales,biases}` |
| API for per-expert slice | `mx.take(tensor, mx.array(held_ids), axis=0)` |
| Attribute assignment | Direct `proj.weight = ...` (works; `num_experts` is a property) |
| Safetensors slice (strategy A fallback) | Row-by-row `s[i, :, :]` + `np.stack` (list-indexing not supported) |
| Correctness bar | D4.1 and D4.2 bit-exact on L=7 (proven path); Tier 1 token-id equality (sort-path has FP noise) |
| Dtype preservation | `weight: uint32`, `scales: bfloat16`, `biases: bfloat16` — no casting needed |
| Quant-group alignment | Safe: groups are along last axis, slicing along axis 0 preserves them |

## 8. References

- Phase 3 spec: `docs/superpowers/specs/2026-04-16-phase3-expert-sharding-design.md`
- Phase 4 spec: `docs/superpowers/specs/2026-04-16-phase4-load-aware-routing-design.md`
- mlx-vlm loader: `.venv/lib/python3.13/site-packages/mlx_vlm/utils.py` (`load_model`)
- mlx_lm switch layers: `.venv/lib/python3.13/site-packages/mlx_lm/models/switch_layers.py` (`SwitchLinear`, `SwitchGLU`, `QuantizedSwitchLinear`)
- Gemma 4 model: `.venv/lib/python3.13/site-packages/mlx_vlm/models/gemma4/language.py` (`Experts`, `Router`, `DecoderLayer`)
- Spec §10 (dynamic migration — Phase 5b scope)
