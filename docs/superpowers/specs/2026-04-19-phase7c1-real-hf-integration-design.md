# Phase 7-C-1: Real HF Integration + DGX Spark Tier-1 — Design

**Status:** Draft, awaiting user review.
**Date:** 2026-04-19
**Phase predecessor:** 7-B (PyTorchBackend thin delegation wrapper, commits `b8253f6` through `4af9ac1`).
**Phase successors:** 7-C-2 (cross-backend correctness harness), 7-C-3 (heterogeneous cluster), 7-C-4 (cleanup).

## 1. Goal

Make the PyTorchBackend generate coherent tokens on real Gemma 4 26B A4B weights loaded from HF. Phase 7-B shipped the backend with synthetic-test coverage only; on the real HF model, `run_layer_atomic` and `pt_moe.*` silently produce wrong output because they bypass `past_key_values` / `position_embeddings` / layernorm sequencing. 7-C-1 closes that gap: single-node Tier 1 passes on DGX Spark, with a committed fixture-based regression test so future code changes don't silently regress the forward pass.

## 2. Architecture

### 2.1 Position state threading via the `masks` tuple (A, from Q3)

The `Backend.run_layer_atomic(layer_idx, h, cache, masks)` protocol slot `masks: tuple[Mask, Mask]` stays a 2-tuple — but the PyTorch backend repurposes the tuple to carry `(cos, sin)` rotary embeddings, while MLX keeps returning `(global_mask, sliding_mask)`. Both fit `tuple[Any, Any]`; consumers never mix them across backends.

- `pytorch_engine.make_masks(model, h, cache)` computes `position_ids` from `cache.get_seq_length() + h.shape[1]` then calls `model.model.rotary_emb(h, position_ids)` and returns the resulting `(cos, sin)` tuple. Cadence: once per iteration (same as MLX).
- `pytorch_engine.run_layer_atomic(model, layer_idx, h, cache, global_mask, sliding_mask)` treats `(global_mask, sliding_mask)` as `(cos, sin)`, derives `position_ids` + `cache_position` from cache state internally, calls `layer(hidden_states=h, position_embeddings=(cos, sin), attention_mask=None, position_ids=..., past_key_value=cache, use_cache=True, cache_position=...)`, and unpacks the tuple return.

No Backend-protocol signature changes. No Node / Orchestrator signature changes beyond §2.3's outer-residual fix.

### 2.2 `pt_moe.*` HF-correct forward replication (from Section 2)

The split-layer path (layer 15, MoE) runs even on single-node because `ExpertOrchestrator.run_split_layer` fans out based on shard-map, not local-ownership. The four `pt_moe.*` functions must exactly slice HF's `Gemma4TextDecoderLayer.forward`:

**`run_attention_and_route(model, h, layer_idx, cache, masks, heat_observer=None)`** — attention sub-block only:
```
residual = h
x = input_layernorm(h)
attn_out = self_attn(x, position_embeddings=(cos,sin), position_ids=..., past_key_value=cache,
                     use_cache=True, cache_position=..., attention_mask=None)
h = residual + attn_out
post_attn = post_attention_layernorm(h)
router_in = pre_feedforward_layernorm_2(post_attn)
(top_k_ids, top_k_weights) = router(router_in)
return (post_attn, top_k_ids, top_k_weights)
```

**`run_shared_expert(model, h, layer_idx)`** — dense-branch pre-norm + MLP:
```
normed = pre_feedforward_layernorm(h)
return mlp(normed)
```

**`run_selected_experts(model, h, layer_idx, expert_ids)`** — per-expert pre-norm + gated MLP:
```
normed = pre_feedforward_layernorm_2(h)
for k in expert_ids:
    gu = F.linear(normed, experts.gate_up_proj[k])
    g, u = gu.chunk(2, dim=-1)
    mid = F.silu(g) * u
    out[k] = F.linear(mid, experts.down_proj[k])
```

**`aggregate_experts(model, layer_idx, expert_outputs, top_k_ids, top_k_weights, shared_out)`** — post-norm + combine:
```
moe_branch = weighted_sum(expert_outputs, top_k_weights)  # per-position
dense_normed = post_feedforward_layernorm_1(shared_out)
moe_normed = post_feedforward_layernorm_2(moe_branch)
return dense_normed + moe_normed
```

**Key change from Phase 7-B:** `run_shared_expert` applies `pre_feedforward_layernorm` internally; `run_selected_experts` applies `pre_feedforward_layernorm_2` internally. Phase 7-B had these as no-ops (synthetic test didn't care). Real HF needs them.

Task 1 does an HF source-read to confirm the exact layernorm sequence and `self_attn` signature before code lands.

### 2.3 Outer residual in `ExpertOrchestrator.run_split_layer`

HF Gemma 4's MoE block structure:
```
hidden_states = attention_sub_block(h)     # post_attn out of run_attention_and_route
residual = post_attn
block_out = dense_normed + moe_normed      # out of aggregate_experts
hidden_states = residual + block_out       # outer residual — MISSING in current orchestrator
```

Current `run_split_layer` returns the aggregate output directly; on real HF that omits the outer `post_attn + block_out` residual and produces wrong hidden states for downstream layers.

**Fix:** after `aggregate_experts`, add `h = post_attn + agg`. MLX's existing path must be audited to see whether this residual is already baked into MLX's layernorm convention. Two cases:

- **Case A** (MLX already includes outer residual internally): the orchestrator adds it unconditionally on both sides. The MLX add is algebraically a no-op because MLX's aggregate already includes `post_attn`; MLX slow regression verifies this.
- **Case B** (MLX relies on orchestrator): the orchestrator adds it unconditionally, and MLX slow regression improves (or was passing by accident).
- **Case C** (MLX doesn't need it, PyTorch does): guard with `isinstance(self.backend, PyTorchBackend)` — PyTorch adds, MLX skips. One-line branch.

Task 4 empirically determines which case by reading `moe.py` and running MLX slow regression with each option.

### 2.4 Tiny-HF integration test on Mac (new)

`tests/test_pytorch_tiny_hf_integration.py` — a slow-marked test that builds a minimal Gemma4ForCausalLM from a hand-rolled config:

```python
Gemma4TextConfig(
    hidden_size=64, num_hidden_layers=2,
    num_experts=4, top_k_experts=2, moe_intermediate_size=16,
    layer_types=["full_attention", "full_attention"],
    num_attention_heads=4, head_dim=16, num_key_value_heads=2,
    vocab_size=256, intermediate_size=32,
    # any other required Gemma4TextConfig fields with minimal values
)
```

Random-init. Runs on CPU, takes seconds. Verifies `PyTorchBackend.embed → make_masks → run_layer_atomic loop → finalize` produces a finite `[1, L, V]` logits tensor without exceptions. Covers what synthetic tests miss: real `self_attn` signature, real `rotary_emb` behavior, real layernorm sequencing.

Marked `@pytest.mark.slow` — CI fast loop skips it; developers run `uv run pytest -m slow tests/test_pytorch_tiny_hf_integration.py` before any 7-C-1 commit.

### 2.5 Spark fixture generation + Tier-1 regression (from Q2 bar B)

`scripts/generate_pytorch_tier1_fixture.py` (already shipped as a stub in 7-B Task 7) runs on DGX Spark once: loads the real 54 GB model, greedy-decodes 10 tokens for each of 3 canonical prompts, writes `tests/fixtures/pytorch_tier1_tokens.json` (replacing the 7-B placeholder).

Commit the generated fixture. Then `tests/test_pytorch_tier1.py` (also from 7-B Task 7) becomes a permanent regression test — subsequent PRs that break PyTorch forward semantics fail this test on Spark.

Task 6 does this. Requires DGX Spark reachable via SSH or Tailscale. **Tailscale is preferable** but not required — a one-shot manual SSH session + `scp` suffices.

## 3. Testing bar

| Tier | Scope | When |
|---|---|---|
| Synthetic unit tests | `test_pytorch_engine.py`, `test_pt_moe_unit.py`, `test_pt_partial_load.py` + updates for new signatures | Every commit (fast) |
| Tiny-HF integration | `test_pytorch_tiny_hf_integration.py` | Before 7-C-1 commits (Mac CPU, slow) |
| Spark Tier-1 | `test_pytorch_tier1.py` with real fixture | Spark only, after Task 6 |
| MLX regression | Existing 6 slow buckets | Every 7-C-1 commit that touches orchestrator (non-negotiable) |

## 4. Success criteria

1. Fast unit tests green on Mac.
2. `test_pytorch_tiny_hf_integration.py` green on Mac CPU.
3. MLX slow regression bucket (all 6) stays green.
4. `scripts/spark_smoke_test.py` produces coherent tokens on DGX Spark.
5. `tests/fixtures/pytorch_tier1_tokens.json` replaced with Spark-generated data (committed).
6. `test_pytorch_tier1.py` green on DGX Spark.
7. README Phase 7-C-1 status paragraph.
8. Memory `project_gossip_moe.md` Phase 7-C-1 COMPLETE entry.

## 5. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Our `self_attn` kwargs mismatch real HF signature (e.g. `past_key_value` vs `past_key_values` plurality) | Task 1 research step reads HF source verbatim; Task 5 tiny-HF integration test catches any residual mismatch before Spark deploy |
| Outer-residual fix breaks MLX path | Task 4 runs MLX slow regression before committing the orchestrator change; if it breaks, fall back to Case C (`isinstance`-guarded) |
| `Gemma4TextConfig` minimum-viable field set is undocumented; test setup may need many kwargs | Task 5 implementer reads HF config source (`modeling_gemma4.py::configuration_gemma4.py`) and enumerates required fields; worst case, copy from HF's own test fixtures |
| DGX Spark not yet on Tailscale | Task 6 can use direct SSH; Tailscale enrollment is a separate operational action the user is aware of |
| `model.model.rotary_emb` attribute name differs from HF version | Task 1 research confirms attribute path; if different, `make_masks` adapts |
| `cache.get_seq_length()` API signature varies across transformers versions | Pin `transformers>=5.5.0` (Phase 7-B); if 5.5 doesn't expose `get_seq_length`, use `cache.seen_tokens` or the equivalent public attr |

## 6. Non-goals

- Cross-backend correctness harness (MLX vs PyTorch `allclose` + top-1) — **Phase 7-C-2**.
- Heterogeneous gossip cluster — **Phase 7-C-3**.
- Phase 6-B provenance on PyTorch path — deferred with 7-C-3.
- Removing `lm` param threading in orchestrator — **Phase 7-C-4** cleanup.
- Removing `_MLX_COMPUTE_LOCK` alias — **Phase 7-C-4** cleanup.
- 4-bit quantization on PyTorch.
- Performance optimizations (torch.compile, flash-attn variant, CUDA graphs).

## 7. Decision log

- **D1 — Sub-phase decomposition.** 7-C is too big for one spec; split into 7-C-1 (real HF integration, this spec), 7-C-2, 7-C-3, 7-C-4. Each is its own brainstorm.
- **D2 — `(cos, sin)` via existing `masks` tuple slot.** Keeps Backend protocol unchanged; avoids cascading signature changes into MLX + orchestrator + tests. Accepted per Q3-A.
- **D3 — `pt_moe.*` replicates HF forward exactly.** Alternative of "short-circuit single-node to atomic path" was rejected because 7-C-3 multi-node needs the split-layer path anyway. Better to do it right once in 7-C-1.
- **D4 — Tiny-HF integration test on Mac.** Adds real-HF integration coverage without requiring Spark access for every commit. `Gemma4TextConfig(num_experts=4, hidden_size=64, num_hidden_layers=2)` is small enough for CPU.
- **D5 — Spark fixture as regression bar.** Phase 7-B shipped a placeholder; 7-C-1 Task 6 generates the real one. Top-1 agreement on 10 positions × 3 prompts is the permanent regression guarantee.
- **D6 — Outer residual case-analysis deferred to Task 4.** The MLX path's current behavior is the source of truth; the fix is whatever makes both paths correct post-change.
- **D7 — `@pytest.mark.slow` for tiny-HF integration.** Keeps CI fast loop under 5s; developers run slow tier before commits.

## 8. Task decomposition

1. Research dump: subagent reads HF `modeling_gemma4.py::Gemma4TextDecoderLayer.forward`, `Gemma4TextAttention.forward`, `Gemma4TextConfig` fields. Dumps findings into the plan itself. Unblocks Tasks 2 + 3 + 5.
2. `pytorch_engine.run_layer_atomic` + `make_masks` rework per §2.1. Updated synthetic tests. Commit.
3. `pt_moe.*` HF-correct forward per §2.2. Updated synthetic layer. Commit.
4. `ExpertOrchestrator.run_split_layer` outer residual per §2.3. MLX slow regression verified. Commit.
5. Tiny-HF integration test per §2.4. New file, Mac CPU, slow-marked. Commit.
6. DGX Spark integration: generate fixture, replace placeholder, verify `test_pytorch_tier1.py` on Spark, README + memory update. Commit.

Each task follows TDD + subagent-driven development with two-stage review (spec + quality) per the Phase 7-A/7-B workflow.

## 9. Open questions

None at spec time. User-confirmed answers:
- Scope: Phase 7-C-1 only (Q1-A).
- Success bar: unit tests + Spark fixture + Tier-1 regression (Q2-B).
- Position-state threading: repurpose `masks` tuple (Q3-A).
- Design sections 1-4 all approved.

The spec has no placeholders and no unresolved design choices.
