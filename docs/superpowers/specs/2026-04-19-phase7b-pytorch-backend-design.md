# Phase 7-B: PyTorchBackend + DGX Spark Single-Node Tier-1 — Design

**Status:** Draft, awaiting user review.
**Date:** 2026-04-19
**Phase predecessor:** 7-A (Backend protocol + MLXBackend, commits `9b8218a` through `6a8c56a`).
**Phase successors:** 7-C (heterogeneous cluster + cross-backend correctness harness).

## 1. Goal

Add a PyTorch implementation of the `Backend` protocol (introduced in Phase 7-A) so the distributed engine runs on NVIDIA DGX Spark (GB10 Grace Blackwell, SM_121, 128 GB unified LPDDR5X). Full parity with `MLXBackend`: every protocol method implemented, including partial-load / slice / attach / detach for Phase 5a/5b/6-C features. Single-node Tier-1-equivalent generation verified on actual Spark silicon. Remove the temporary Phase 7-A shims (`ExpertOrchestrator.backend=None` fallback, `Node._lm` property) now that a second backend exists to validate the abstraction.

## 2. Architecture

### 2.1 Module layout — mirror the MLX side

```
src/model_shard/
  pytorch_engine.py        # mirror of mlx_engine.py
  pt_moe.py                # mirror of moe.py
  pt_partial_load.py       # mirror of partial_load.py
  backends/
    pytorch_backend.py     # PyTorchBackend class
    __init__.py            # re-export PyTorchBackend
```

Each mirrored module keeps function signatures close to its MLX twin, substituting `torch.Tensor` for `mx.array`. The two backends remain interchangeable from the orchestrator's perspective because both implement the same `Backend` protocol with opaque `Activation` / `Cache` / `Mask` / `TopK` handles.

### 2.2 Model loading

HF transformers ≥ 5.5.0 ships `Gemma4ForCausalLM` natively (model directory `src/transformers/models/gemma4/`). Loader:

```python
AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-26B-A4B-it",
    torch_dtype=torch.bfloat16,   # float16 on MPS; MPS has no bf16
    device_map=self._device,
).eval()
```

No `trust_remote_code`, no custom modeling file, no quantization library. `use_cache=True` at every forward call (works around open transformers bug #45242 on Gemma 4 with `use_cache=False`).

### 2.3 Expert weight layout — identical to MLX

HF uses stacked tensors, not `nn.ModuleList`:

```python
layer.experts.gate_up_proj  # shape [num_experts, 2*moe_intermediate_size, hidden_size]
layer.experts.down_proj     # shape [num_experts, hidden_size, moe_intermediate_size]
```

Expert `k` in layer `i` is addressed as `model.model.layers[i].experts.gate_up_proj[k]`, `model.model.layers[i].experts.down_proj[k]`. This is the same dim-0 slicing pattern the MLX `partial_load.py` already uses; `pt_partial_load.py` ports the algorithm verbatim.

### 2.4 Shared expert — parallel dense MLP

On MoE layers, `Gemma4TextDecoderLayer.forward` runs both `self.mlp(...)` (dense path, per-token) **and** `self.experts(...)` (sparse top-8 path) and sums the outputs. `self.mlp` is the shared expert; `self.experts` is the sparse branch. `pt_moe.run_shared_expert` calls `layer.mlp(h)` directly; `pt_moe.run_selected_experts` bypasses `MixtralExperts.forward` and does per-expert `F.linear(h, gate_up_proj[k])` — matching how MLX routes per-expert work and avoiding HF's Python-per-expert dispatch loop.

### 2.5 Router

`Gemma4TextRouter` is not a plain `nn.Linear`. It contains:

- `self.norm = Gemma4RMSNorm(..., with_scale=False)` — input normalization.
- `self.proj = nn.Linear(hidden_size, num_experts, bias=False)`.
- `self.scale` — per-dim learnable scale applied to the pre-proj hidden.
- `self.per_expert_scale` — per-expert learnable scale applied to the top-k weights.

`pt_moe.run_attention_and_route` calls `layer.router(h_post_attn)` and returns `(post_attn, (top_k_ids, top_k_weights))` matching the Backend protocol's `TopK = tuple[Activation, Activation]` shape.

### 2.6 KV cache

`DynamicCache` from `transformers.cache_utils` — simpler than MLX's `HybridCache`-equivalent. Per-layer slots accessed as `past_key_values.key_cache[layer_idx]` / `.value_cache[layer_idx]`. Identity map; no `layer_idx_to_cache_idx` translation needed on the PyTorch side.

### 2.7 Layer-type introspection

`config.layer_types: list[str]` — each entry is `"full_attention"` or `"sliding_attention"`. Per-layer: `layer.layer_type`. The existing MLX check `layer.layer_type == "full_attention"` ports verbatim.

### 2.8 Device selection

Single class for all PyTorch devices:

```python
PyTorchBackend(device="cuda")   # default on Spark / CUDA hosts
PyTorchBackend(device="mps")    # Apple PyTorch path (dev/test only)
PyTorchBackend(device="cpu")    # fallback
```

Auto-detect logic in `_default_device()`: CUDA > MPS > CPU. Dtype: `torch.bfloat16` on CUDA/CPU, `torch.float16` on MPS.

### 2.9 Wire dtype

bf16 bytes are identical on both backends (IEEE 754 bf16 is bf16 regardless of runtime), so a PyTorch node can actually exchange tensor bytes with an MLX node cleanly at the wire layer. That is what unlocks Phase 7-C (heterogeneous cluster).

## 3. PyTorchBackend protocol conformance

### 3.1 Method mapping

| Protocol | Delegates to |
|---|---|
| `load(hf_id)` | HF `AutoModelForCausalLM.from_pretrained` |
| `load_partial(hf_id, held)` | Load full; zero-out non-held `gate_up_proj[k]` / `down_proj[k]`; populate `_held_experts_per_layer`. |
| `num_layers()` | `self._model.config.num_hidden_layers` |
| `held_ids(L)` | `self._held_experts_per_layer.get(L, ())` |
| `is_split_layer(L)` | always `False` — ShardSpec decides, matching MLXBackend. |
| `embed(token_ids)` | `pytorch_engine.embed_tokens(self._model, token_ids)` |
| `make_cache()` | `pytorch_engine.make_cache(self._model)` (returns `DynamicCache()`) |
| `make_masks(h, cache)` | `pytorch_engine.make_masks(self._model, h, cache)` |
| `run_layer_atomic(...)` | `pytorch_engine.run_layer_atomic(...)` |
| `run_attention_and_route(...)` | `pt_moe.run_attention_and_route(...)` |
| `run_shared_expert(L, h)` | `pt_moe.run_shared_expert(...)` → `layer.mlp(h)` |
| `run_selected_experts(L, h, ids)` | `pt_moe.run_selected_experts(...)` |
| `aggregate_experts(...)` | `pt_moe.aggregate_experts(...)` |
| `finalize(h)` | `pytorch_engine.finalize(self._model, h)` |
| `argmax_last(logits)` | `int(torch.argmax(logits[0, -1, :]).item())` |
| `tensor_to_bytes(h)` | `h.contiguous().view(torch.uint8).cpu().numpy().tobytes()` |
| `bytes_to_tensor(raw, shape, dtype)` | `torch.frombuffer(bytearray(raw), dtype=_wire_to_torch(dtype)).reshape(shape).to(self._device)` |
| `dtype_to_wire(h)` | torch dtype → wire int; bf16 → `DTYPE_BFLOAT16`, fp16 → `DTYPE_FLOAT16`, fp32 → `DTYPE_FLOAT32` |
| `slice_expert(L, E)` | `pt_partial_load.slice_expert(self._model, L, E, self._torch_lock)` — returns `[gate_up_proj[E].detach().cpu(), down_proj[E].detach().cpu()]` |
| `attach_expert(L, E, tensors)` | `pt_partial_load.attach_expert(self._model, L, E, tensors, self._torch_lock)` — writes in-place under lock |
| `detach_expert(L, E)` | `pt_partial_load.detach_expert(self._model, L, E, self._torch_lock)` — zero-out slices; update `_held_experts_per_layer` |

### 3.2 Lock discipline

`PyTorchBackend(torch_lock: threading.Lock | None = None)` mirrors `MLXBackend(mlx_lock=...)`. When `Node.__init__` constructs the backend, it passes its process-wide compute lock (probably renamed from `_MLX_COMPUTE_LOCK` to `_COMPUTE_LOCK` to be backend-neutral). The lock serializes slice/attach/detach against concurrent forward passes. `threading.Lock` is non-reentrant — verified empirically that no production call site acquires the lock externally around a backend slice/attach/detach call.

### 3.3 `from_loaded_model` escape hatch

```python
@classmethod
def from_loaded_model(cls, model, device=None, torch_lock=None) -> "PyTorchBackend":
    b = cls(device=device, torch_lock=torch_lock)
    b._model = model
    return b
```

Used by tests that inject a pre-built `Gemma4ForCausalLM` or a mock.

## 4. Node & orchestrator wiring changes

### 4.1 Backend selection at `Node.__init__`

Extend the current three-way precedence (explicit `backend=` > `loaded_model=` > auto-load) to pick PyTorch vs MLX in the auto-load branch:

```python
def _default_backend() -> Backend:
    env = os.environ.get("MODEL_SHARD_BACKEND", "").lower()
    if env == "pytorch":
        return PyTorchBackend(torch_lock=_COMPUTE_LOCK)
    if env == "mlx":
        return MLXBackend(mlx_lock=_COMPUTE_LOCK)
    # Auto: prefer MLX on Apple Silicon, PyTorch elsewhere.
    try:
        import mlx.core as mx
        if mx.metal.is_available():
            return MLXBackend(mlx_lock=_COMPUTE_LOCK)
    except ImportError:
        pass
    return PyTorchBackend(torch_lock=_COMPUTE_LOCK)
```

The `_COMPUTE_LOCK` rename from `_MLX_COMPUTE_LOCK` reflects that the lock is backend-neutral. Old name kept as an alias for one release for any external consumer.

Legacy `loaded_model=` path remains MLX-only (a loaded model is always a `LoadedModel` MLX struct; PyTorch callers pass `backend=PyTorchBackend.from_loaded_model(...)` explicitly). No type-sniffing.

### 4.2 Remove 7-A shims

**`ExpertOrchestrator.backend=None` fallback — removed.** The field becomes `backend: Backend` (required, no default). Five `if self.backend is not None:` / `else:` branches in `run_split_layer` / `_phase_b_with_retry` / Phase-C aggregate are collapsed to the backend-true arm. The `lm` parameter threaded through `run_split_layer` / `_phase_b_with_retry` for the fallback is removed. Tests that construct `ExpertOrchestrator(...)` directly pass `backend=MagicMock(spec=Backend)` (with `.run_attention_and_route`, `.run_shared_expert`, etc. returning shaped mocks) — a handful of `tests/test_expert_*.py` files need minor updates.

**`Node._lm` @property — removed.** Phase 7-A added this as back-compat for pre-Phase-7 code reading `node._lm` directly. The only production Node-internal consumer is `_run_my_layers`, which calls `run_layers(...)` because that helper carries Phase 6-B provenance and stays backend-specific per §4.3 (D6). Post-refactor, `_run_my_layers` dispatches on backend type: `isinstance(self._backend, MLXBackend)` → `mlx_engine.run_layers(self._backend._lm, ...)`; `isinstance(self._backend, PyTorchBackend)` → `pytorch_engine.run_layers(self._backend._model, ...)`. Split-layer paths use the backend protocol methods directly and don't need `_lm`. External tests that read `n._lm` are rewritten to `n._backend._lm` (MLX-only) or `n._backend._model` (PyTorch-only) — type-narrowed at the test site.

### 4.3 Does `run_layers` need a PyTorch port?

`run_layers` is the non-split atomic multi-layer path used by `_run_my_layers`. It carries Phase 6-B provenance-chain append. Two options:

- **Option A (recommended):** port `run_layers` to PyTorch as `pytorch_engine.run_layers`, provenance append identical. `_run_my_layers` dispatches based on `isinstance(self._backend, MLXBackend)` / `PyTorchBackend`. Keeps `run_layers` backend-specific (outside the protocol) but working on both.

- **Option B:** extend the Backend protocol with a `run_layers(layer_range, h, cache, masks, provenance_chain, node_id) -> (h, provenance_chain)` method. Cleaner long-term but expands the protocol surface.

Phase 7-B picks **Option A** to minimize protocol-level churn. Option B can be evaluated in Phase 7-C if cross-backend orchestration needs it.

## 5. Testing & correctness bar

### 5.1 Fast unit tests (CPU, run on every commit, Apple + CUDA hosts)

- `tests/test_pytorch_backend.py` — protocol conformance via `isinstance(b, Backend)`, `from_loaded_model`, `tensor_to_bytes` roundtrip, `argmax_last`, `held_ids` delegation, `is_split_layer` False, optional `torch_lock`. Uses `MagicMock` for the model. Mirrors `test_mlx_backend.py`.
- `tests/test_pt_partial_load.py` — `slice_expert` / `attach_expert` / `detach_expert` with a tiny synthetic `nn.Module` holding stacked `gate_up_proj` `[4, 8, 4]` and `down_proj` `[4, 4, 4]` tensors. Verifies shape, values, thread-lock acquisition.
- `tests/test_pt_moe_unit.py` — `run_attention_and_route` shape check; `run_shared_expert` calls `layer.mlp`; `run_selected_experts` does per-expert `F.linear` on the stacked tensor.
- `tests/test_pytorch_engine.py` — `embed_tokens`, `make_cache`, `make_masks`, `run_layer_atomic`, `finalize`, wire dtype roundtrip.

### 5.2 Slow integration tests (DGX Spark only, marked `@pytest.mark.slow @pytest.mark.cuda`)

- `tests/test_pytorch_tier1.py` — load `google/gemma-4-26B-A4B-it` bf16 on CUDA. Run 3 fixture prompts (same prompts as MLX Tier 1 `tests/test_tier1_tokens.py`). **Correctness bar:** top-1 token agreement on positions 0-10 against a pre-generated fixture `tests/fixtures/pytorch_tier1_tokens.json`. Fixture is generated once on Spark via `scripts/generate_pytorch_tier1_fixture.py`, committed, then the test just compares. Gives us PyTorch-side regression testing without requiring MLX-PyTorch cross-equivalence.
- `tests/test_pytorch_migration_e2e.py` — starts a 2-Node localhost PyTorch cluster (using an MPS or CPU backend for the test to avoid requiring actual CUDA; or gated on CUDA if we want full Spark-only). Triggers `migration_attach`, then `migration_detach`. Verifies decode continues correctly across the migration. Exercises full MLXBackend parity claim.

### 5.3 MLX regression preservation

Phase 7-B's shim removals could regress the MLX path. The full existing MLX slow buckets must stay green:

```
uv run pytest -m slow -q tests/test_tier1_tokens.py
uv run pytest -m slow -q tests/test_partial_load_bit_exact_per_expert.py
uv run pytest -m slow -q tests/test_migration_over_tcp.py
uv run pytest -m slow -q tests/test_expert_retry_bit_exact.py
uv run pytest -m slow -q tests/test_provenance_tier1.py
uv run pytest -m slow -q tests/test_eviction_e2e.py
```

All must be green on the MLX dev box.

### 5.4 Environment setup on DGX Spark

- Upstream PyTorch ≥ 2.6 + CUDA 12.9 wheel (not NGC container — bf16 matmul works on SM_120 forward-compat even without SM_121-specific kernels; no FP4 needed).
- `pyproject.toml` adds an optional dependency group:
  ```toml
  [project.optional-dependencies]
  pytorch = ["torch>=2.6", "transformers>=5.5.0", "accelerate>=1.0"]
  ```
  Mac-only dev doesn't pull CUDA wheels; `uv sync --extra pytorch` installs them on Spark.
- `scripts/spark_smoke_test.py` — manual "does it generate tokens" sanity script run after first deploy.

### 5.5 Success criterion for Phase 7-B "done"

1. Fast unit tests green on all platforms.
2. `tests/test_pytorch_tier1.py` green on DGX Spark.
3. `tests/test_pytorch_migration_e2e.py` green.
4. Existing MLX slow regression buckets all green post-shim-removal.
5. `README.md` has a Phase 7-B status paragraph.
6. Memory file `project_gossip_moe.md` has a Phase 7-B COMPLETE entry.

## 6. Risks & mitigations

| Risk | Mitigation |
|---|---|
| HF `Gemma4ForCausalLM` has a subtle bug that breaks forward pass | `use_cache=True` always (works around known #45242). Pin `transformers==5.5.X`. Fallback: copy modeling file locally and patch. |
| bf16 numerics differ across PyTorch devices | Don't test across devices. Tier-1 fixture is generated on the same device it's tested on (CUDA). |
| Concurrent slice/attach/detach corrupts stacked tensors | `torch_lock` held during mutation. Matches MLX pattern exactly. |
| `MixtralExperts.forward` Python-per-expert loop is slow | Bypass it: `pt_moe.run_selected_experts` loops itself, matching distributed routing. |
| `DynamicCache` layout mismatches MLX cache API | HF exposes `past_key_values.key_cache[i]` as direct integer index. Identity map on PyTorch side. |
| Upstream PyTorch wheel lacks SM_121 kernels on Spark | Falls back to SM_120 forward-compat for bf16 matmul (our hot path). Acceptable. |
| `_COMPUTE_LOCK` rename breaks external users reading `_MLX_COMPUTE_LOCK` | Keep old name as an alias for one release. Spec calls out the alias explicitly. |
| Removing `ExpertOrchestrator.backend=None` fallback breaks unit tests that construct orchestrators directly | Update tests to pass `backend=MagicMock(spec=Backend)`. Enumerated in Task 6. |

## 7. Non-goals

Explicitly out of scope for Phase 7-B (deferred to later phases):

- Cross-backend correctness harness (top-1 MLX vs PyTorch on identical prompts) — **Phase 7-C**.
- Heterogeneous gossip cluster (MLX head + PyTorch tail) — **Phase 7-C**.
- 4-bit quantization on PyTorch (NVFP4, GPTQ, bitsandbytes, AutoRound, AWQ) — deferred. bf16 only for 7-B.
- Performance optimizations: CUDA graphs, torch.compile, FlashAttention-3, custom kernels. Correctness first.
- Windows support.
- macOS MPS validation beyond the CI fast-unit-test bar. Primary validation target is Spark CUDA.

## 8. Decision log

- **D1 — bf16 on PyTorch, 4-bit on MLX is acceptable asymmetry.** Cross-framework bit-exactness is physically impossible (different reduction kernels, different bf16 accumulation paths). No PyTorch quant is bit-exact with MLX 4-bit. The "same quantization" goal is relaxed to "both backends implement the same protocol" with Phase 7-C's top-1 / allclose harness as the cross-backend correctness bar.
- **D2 — HF transformers native, no custom modeling.** `Gemma4ForCausalLM` shipped in transformers v5.5.0 (2026-04-01). No `trust_remote_code`, no vendored modeling file. Reduces the project from ~6 weeks to ~2-3 weeks.
- **D3 — Mirror MLX module layout.** `pytorch_engine.py` / `pt_moe.py` / `pt_partial_load.py` / `backends/pytorch_backend.py`. Symmetric with MLX side makes the backend abstraction's value visible: orchestrator code is identical, module code is backend-specific.
- **D4 — Full protocol parity (all 20 methods).** Partial-load, slice, attach, detach all implemented — enables Phase 5a/5b/6-C features on DGX Spark. Unblocks heterogeneous cluster work in 7-C.
- **D5 — Remove 7-A shims in 7-B.** `ExpertOrchestrator.backend=None` fallback and `Node._lm` property were flagged temporary in 7-A's spec. Now that PyTorchBackend exists, we have the second implementation that validates the abstraction; keeping the shims wastes maintenance surface.
- **D6 — `run_layers` stays backend-specific (Option A).** Port it to `pytorch_engine.run_layers` rather than lifting into the Backend protocol. `_run_my_layers` dispatches on backend type. Revisit in 7-C if cross-backend orchestration needs a protocol-level `run_layers`.
- **D7 — `_COMPUTE_LOCK` rename from `_MLX_COMPUTE_LOCK`.** Lock is backend-neutral; old name aliased for one release.
- **D8 — Optional-dependency group for PyTorch.** Mac-only dev doesn't pull CUDA wheels; `uv sync --extra pytorch` on Spark installs the full stack.
- **D9 — Correctness bar: pre-generated Spark fixture + top-1 on first 10 positions.** Internal regression (PyTorch stable across commits). Cross-backend equivalence is Phase 7-C.
- **D10 — Auto-detect backend with env var override.** `MODEL_SHARD_BACKEND=pytorch|mlx` as the explicit knob. Auto prefers MLX on Apple Silicon, PyTorch elsewhere.

## 9. Task decomposition

Seven tasks, sized similarly to Phase 7-A:

1. Scaffolding: `optional-dependencies.pytorch` group + `pytorch_engine.py` skeleton + `cuda` pytest marker config + `_COMPUTE_LOCK` alias.
2. `pytorch_engine.py`: `load_model`, `embed_tokens`, `make_cache`, `make_masks`, `run_layer_atomic`, `run_layers`, `finalize`, `tensor_to_bytes`, `bytes_to_tensor`, `torch_to_wire_dtype` + fast unit tests with synthetic `nn.Module`.
3. `pt_moe.py`: `run_attention_and_route`, `run_shared_expert`, `run_selected_experts`, `aggregate_experts` + fast unit tests.
4. `pt_partial_load.py`: `slice_expert`, `attach_expert`, `detach_expert`, `load_model_partial` + fast unit tests.
5. `backends/pytorch_backend.py`: `PyTorchBackend` class + protocol conformance tests.
6. Node & orchestrator refactor: `_default_backend()` with auto-detect + env var. **Remove** `ExpertOrchestrator.backend=None` fallback (5 branches). **Remove** `Node._lm` @property. Rename `_MLX_COMPUTE_LOCK` → `_COMPUTE_LOCK` with alias. Update affected tests. MLX regression must stay green.
7. DGX Spark integration: `test_pytorch_tier1.py` + `test_pytorch_migration_e2e.py` + `scripts/generate_pytorch_tier1_fixture.py` + `scripts/spark_smoke_test.py` + README Phase 7-B paragraph + memory update. Full verification sweep.

Each task follows TDD + subagent-driven development with two-stage review (spec + quality) per the Phase 7-A workflow.

## 10. Open questions

None at spec time. User-confirmed answers to brainstorming questions:

- Scope: Single-node Tier-1-equivalent on DGX Spark (B from Q1).
- Weights: bf16 native HF, no quantization (A from Q3).
- Feature parity: Full MLXBackend parity including migration (A from Q4).

The spec has no placeholders, no TBDs, and no unresolved design choices.
