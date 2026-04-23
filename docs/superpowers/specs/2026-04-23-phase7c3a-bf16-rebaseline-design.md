# Phase 7-C-3a: Bf16 Canonical Rebaseline — Design

**Status:** Draft, awaiting user review.
**Date:** 2026-04-23
**Phase predecessor:** 7-C-2 (Cross-Backend Correctness Harness, commits `432d680` through `1396ef8`).
**Phase successor:** 7-C-3b (heterogeneous gossip cluster — MLX node + PyTorch nodes in one pipeline; gets its own brainstorm).

## 1. Goal

Replace the 4-bit MLX model (`mlx-community/gemma-4-26b-a4b-it-4bit`) as the canonical Mac path with bf16-MLX, derived once from `google/gemma-4-26B-A4B-it` (the same HF source the PyTorch path uses on DGX Spark). All Phase 1–7-C-2 tests continue to pass on the new baseline.

This is a **precursor phase, not a feature phase**. Nothing observable to users changes except the underlying model precision and the corresponding fixture rebaseline. The motivation is downstream: Phase 7-C-3b requires the cluster's "same model" invariant, and the cleanest way to satisfy that across MLX and PyTorch backends is to standardize on bf16 source weights everywhere.

## 2. Scope and non-goals

**In scope:**
- One-time conversion HF bf16 → MLX bf16 (output path supplied by the user, not baked into code).
- Removal of all hardcoded model-ID string literals from code; `config/shards.yaml::model_id` becomes the single source of truth.
- `ShardMap` carries `model_id`; `node.py` and the entry-point scripts read from there.
- The repo's default `config/shards.yaml` is updated to reference the bf16 model. (Users who want the legacy 4-bit MLX path put that string in their own `shards.yaml`.)
- All committed fixtures regenerated against bf16: Phase 1 oracle (`artifacts/ref/`), MLX top-K (`tests/fixtures/mlx_tier1_tokens.json`), PyTorch top-K (`tests/fixtures/pytorch_tier1_tokens.json`, regenerated on Spark).
- Cross-backend agreement floors in `tests/test_cross_backend_correctness.py` updated to reflect the post-rebaseline reality.
- Memory smoke test confirming single-process bf16 load fits comfortably on M5.

**Explicitly out of scope (deferred to 7-C-3b or later):**
- Heterogeneous gossip cluster routing — MLX node + PyTorch nodes in one pipeline.
- Cluster-wide precision-contract gossip and admission control.
- Cross-backend expert migration (9-tensor MLX-quant ↔ 2-tensor PyTorch-bf16 bridge).
- Removing the legacy 4-bit MLX path entirely.
- Phase 6-B provenance verification on the PyTorch path (already largely backend-agnostic; explicit verification deferred to 7-C-3b).
- 7-C-4 cleanup items (`_MLX_COMPUTE_LOCK` alias retirement; `lm` param threading removal; `mlx.core` import gating in `node.py`).

## 3. Architecture

### 3.1 Model artifact

**Canonical source:** `google/gemma-4-26B-A4B-it` — the gated HF bf16 distribution. ~54 GB on disk. Already authed and cached on Spark from Phase 7-C-1.

**MLX local conversion:** via `mlx_lm.convert`. The conversion script (`scripts/convert_mlx_bf16.py`) takes the HF source and the output path as **explicit CLI arguments** — no defaults baked into the script:

```bash
uv run python scripts/convert_mlx_bf16.py \
    --hf-source google/gemma-4-26B-A4B-it \
    --output-dir <path-the-user-specifies>
```

The output path is a deployment decision, not a code constant. The user picks where to store the converted weights (typical choice: `~/.cache/mlx-models/...`, but the repo doesn't enforce that). Whatever path is chosen gets recorded in `config/shards.yaml` (see §3.2). The script's job is to make the conversion reproducible, not to pick the storage location.

**Output layout:** `mlx_lm.convert` produces a directory containing `config.json`, `model.safetensors*`, `tokenizer.*`, etc., laid out the same way `mlx-community/*` repos are. `mlx_engine.load_model(hf_id)` accepts any string `mlx_lm.load()` accepts — HF id or local directory path — so the existing load path keeps working unchanged.

**Disk footprint:** ~54 GB. One-time cost. Conversion runtime estimated at 15–30 min based on disk I/O dominating.

**HuggingFace weights variant:** the multimodal `google/gemma-4-26B-A4B-it` includes vision tower weights we don't use. Conversion preserves the `Gemma4Model` (multimodal wrapper) topology so `pytorch_engine._text_model()` keeps working unchanged. MLX-side, the same `model.language_model.model.layers` access pattern from the 4-bit path continues to work — the architecture wrapper is preserved by `mlx_lm.convert`.

### 3.2 Code surface changes — eliminating hardcoded model IDs

**Current state (verified 2026-04-23):** model IDs are hardcoded as string literals at 7 call sites across the code:

- `src/model_shard/node.py:186` — `Node.__init__` default-backend branch (MLX → 4-bit literal)
- `src/model_shard/node.py:188` — same branch (PyTorch → bf16 HF literal)
- `scripts/generate_tier1_comparison_fixture.py:46` — MLX dispatch (4-bit literal)
- `scripts/generate_tier1_comparison_fixture.py` (PyTorch dispatch) — bf16 HF literal
- `scripts/run_reference.py:46` — `--model` argparse default (4-bit literal)
- `scripts/run_node.py:92` — `--model` argparse default (4-bit literal)
- `scripts/run_client.py:39` — `--model` argparse default (4-bit literal)

**Target state:** zero hardcoded model strings in code. Single source of truth = `config/shards.yaml`.

#### 3.2.1 `config/shards.yaml` carries the model ID

`shards.yaml` already carries the cluster topology contract. It is the right place for the model identity contract too. Add a top-level `model_id` field:

```yaml
model_id: "/Users/lukechang/.cache/mlx-models/gemma-4-26b-a4b-it-bf16"
shards:
  - shard_id: head
    layer_range: [0, 10]
    address: 127.0.0.1:9001
  - shard_id: mid
    layer_range: [10, 20]
    address: 127.0.0.1:9002
  - shard_id: tail
    layer_range: [20, 30]
    address: 127.0.0.1:9003
```

The string is whatever `mlx_lm.load()` / `transformers.AutoModel.from_pretrained()` accepts — an HF id (`google/gemma-4-26B-A4B-it`) or a local directory path. Each deployment writes the value appropriate for its environment:

- M5 Mac MLX: local converted bf16 directory path (e.g., `~/.cache/mlx-models/gemma-4-26b-a4b-it-bf16`)
- DGX Spark PyTorch: HF id `google/gemma-4-26B-A4B-it` (HF cache resolves it)
- Ubuntu 3090 PyTorch: same HF id

In 7-C-3a (single backend per cluster) every shard in a given `shards.yaml` uses the same model_id. In 7-C-3b, when a heterogeneous cluster is built, the cluster's "same model" invariant is enforced by gossiping the model_id and rejecting joins that disagree — but that admission logic is 7-C-3b's problem, not 7-C-3a's.

#### 3.2.2 `ShardMap` exposes the model_id

`src/model_shard/shard_map.py::ShardMap` gains a `model_id: str` field, populated by the YAML loader.

#### 3.2.3 Call sites consume from ShardMap or CLI args (no defaults)

- **`node.py`:** the default-backend branch at line 184 reads `hf_id = self._shard_map.model_id` instead of the literal string.
- **`scripts/run_node.py`, `run_client.py`, `run_reference.py`:** the `--model` argparse argument loses its default and becomes either required or — preferably — gets resolved from the loaded `shards.yaml` automatically (since these scripts already load the YAML for topology). The `--model` arg stays as an explicit override option but has no default literal.
- **`scripts/generate_tier1_comparison_fixture.py`:** takes `--model` as a required CLI argument; both backend dispatch branches read from it. No string literals in either branch.

The `mlx_engine.load_model(hf_id: str)` and `pytorch_engine.load_model(hf_id: str, ...)` signatures stay unchanged. The hardcoded strings get *removed*; nothing replaces them in code, because the YAML is now the source of truth.

**No `model_ids.py` helper module.** A helper that hides hardcoded strings is still hardcoding. Either the value lives in user-editable configuration (yes), or the code requires the user to supply it (yes). The only acceptable location for a string literal is the YAML — which is configuration, not code.

**No protobuf / wire changes.** `tensor_to_bytes` already encodes dtype in the activation envelope; activations cross the wire as bf16 regardless of underlying weight precision.

**Legacy 4-bit MLX is not deleted, but is no longer privileged.** A user who wants the 4-bit path for memory-constrained Mac dev puts `model_id: "mlx-community/gemma-4-26b-a4b-it-4bit"` in their own `shards.yaml`. The repo's checked-in `config/shards.yaml` defaults to bf16 (the canonical cluster model). No env-var toggle, no code-level fallback.

### 3.3 Fixture rebaseline

The committed reference data falls into three buckets:

1. **`artifacts/ref/` — Phase 1 oracle.** Tier 1 generated tokens + Tier 2 per-layer hidden states for 5 canonical prompts. Regenerated by `scripts/run_reference.py`. Drives `tests/test_tier1_tokens.py` and `tests/test_tier2_hidden.py`. Bf16 versions become the new ground truth.

2. **`tests/fixtures/mlx_tier1_tokens.json` — 7-C-2 MLX top-K fixture.** Regenerated on M5 via `MODEL_SHARD_BACKEND=mlx scripts/generate_tier1_comparison_fixture.py`. The unified generator script already exists from 7-C-2; no script changes required.

3. **`tests/fixtures/pytorch_tier1_tokens.json` — 7-C-2 PyTorch top-K fixture.** Regenerated on Spark via `MODEL_SHARD_BACKEND=pytorch scripts/generate_tier1_comparison_fixture.py`. The PyTorch path itself does not change — this regeneration is purely so the comparison test floors are recomputed on a baseline where MLX and PyTorch consume the *same* source weights.

4. **`tests/fixtures/cross_backend_comparison.md` — auto-regenerated by the test on each run.** No manual step.

### 3.4 Test floor adjustments

`tests/test_cross_backend_correctness.py` has two assertion floors today:

- `≥ 1 of 3 prompts position-0 top-1 agrees` → expected to become `3 of 3` (or all positions, if we want to tighten further).
- `avg top-5 overlap ≥ 0.5 across 30 (prompt, position) pairs` → expected to climb dramatically; if it saturates at ~5.0, set the floor at ~4.0 to leave headroom for residual MLX vs PyTorch implementation drift.

Concrete floor values are determined empirically post-rebaseline; the design commitment is just "tighten meaningfully, don't leave the conservative 4-bit-era floors in place."

### 3.5 Memory smoke test

New fast test (`tests/test_bf16_memory_smoke.py`, < 1 s if guarded; runs only under `-m slow` since it needs a real model load): assert that single-process bf16 load completes and `psutil.Process().memory_info().rss < 80 * 1024**3`. Provides a fast regression bar that prevents accidental memory blowup if the load path mutates weights in-place.

## 4. Verification

All of these must pass for 7-C-3a to ship:

| # | Test bucket | Where | Notes |
|---|---|---|---|
| 1 | Phase 1 Tier 1 (`tests/test_tier1_tokens.py`) | M5 | Exact-match generated tokens vs new bf16 oracle |
| 2 | Phase 1 Tier 2 (`tests/test_tier2_hidden.py`) | M5 | Per-layer hidden state allclose; tolerance may be tightened post-rebaseline |
| 3 | Phase 3 (`tests/test_moe_split_equivalence.py`) | M5 | Atomic vs split layer 15, bit-exact within bf16 |
| 4 | Phase 5a (`tests/test_partial_load_*.py`) | M5 | Per-expert bit-exact + 3-shard mod-3 split |
| 5 | Phase 5b (`tests/test_migration_*.py`, `test_partial_load_tier1_migration.py`) | M5 | Bit-exact migration; ownership convergence |
| 6 | Phase 6-A (`tests/test_expert_retry_*.py`) | M5 | Bit-exact retry with replica preservation |
| 7 | Phase 6-B (`tests/test_provenance_*.py`) | M5 | Tier 1 with provenance enabled (still bit-exact); rejection path |
| 8 | Phase 6-C (`tests/test_eviction_*.py`, `test_partial_load_detach.py`) | M5 | Eviction E2E and attach/detach roundtrip |
| 9 | Phase 7-A backend Tier 1 | M5 | MLXBackend default unchanged |
| 10 | Phase 7-B PyTorch Tier 1 (`tests/test_pytorch_tier1.py`) | Spark | Fixture regenerates; PyTorch path itself invariant |
| 11 | Phase 7-C-2 cross-backend (`tests/test_cross_backend_correctness.py`) | Either | New floors reflect post-rebaseline agreement |
| 12 | Memory smoke test (`tests/test_bf16_memory_smoke.py`) | M5 | Single-process bf16 < 80 GB resident |
| 13 | Fast suite (`uv run pytest`) | M5 | All non-slow tests still green |
| 14 | Lint + type | M5 | `uv run ruff check src tests scripts && uv run mypy src tests scripts` |

## 5. Risks and mitigations

| ID | Risk | Mitigation |
|---|---|---|
| R1 | A Phase 5b/6-A/6-C bit-exact test passed under 4-bit but fails under bf16 (e.g., a test compared bf16-MLX hidden state to a 4-bit-derived expected value). | All bit-exact tests are within-MLX bf16-vs-bf16. If any fails, root-cause it — that's signal, not noise. |
| R2 | Tier 2 hidden-state allclose tolerance was tuned to 4-bit precision. | Tighten the tolerance once after rebaseline; document the new value in the test. Don't loosen — bf16 should be the new floor. |
| R3 | 3-process Mac tests OOM under bf16 because mmap-page-sharing breaks with partial loading. | Run the memory smoke test (verification #12) early in the task list, before the slow buckets. Catch resource regressions before they manifest as confusing pytest failures. |
| R4 | Conversion takes 15–30 min and needs ~54 GB of free disk. | One-time cost; cache the converted model in a stable path; commit `scripts/convert_mlx_bf16.py` for reproducibility. Document disk requirement in README. |
| R5 | HF auth on M5 — `google/gemma-4-26B-A4B-it` is gated. | User already authed for 7-C-1 on Spark; same `huggingface-cli login` flow on M5. Document in script. |
| R6 | 7-C-2 fixture top-K saturation (overlap = 5.0 on every position) makes the agreement floor uninformative. | Saturation is itself the validation that the rebaseline worked. Set the floor at ~4.0 to leave headroom for residual MLX vs PyTorch kernel-rounding drift. |
| R7 | The `mlx_lm.convert` CLI surface differs from current docs (e.g., parameter renames between mlx_lm versions). | First task in the implementation plan is to verify the conversion script actually runs end-to-end on M5 — fail fast on tooling issues before rippling fixture changes. |
| R8 | A residual hardcoded model-ID literal hides somewhere the initial grep missed (e.g., a test fixture, a docstring with a runnable example, a comment). | Grep for both literals (`mlx-community/gemma-4-26b-a4b-it-4bit` and `google/gemma-4-26B-A4B-it`) before starting Task 3 (the call-site refactor); fix every match in the same task. After the refactor, a final grep should return zero hits in `src/` and only intentional references in docs. |
| R9 | `shards.yaml` is now load-bearing for model identity, not just topology. A typo or wrong path silently routes to the wrong model. | `ShardMap` validates that `model_id` is non-empty at YAML-load time. The model load itself fails fast and loudly if the path/HF id is invalid — no silent fallback to a default. |

## 6. Rough task breakdown

The full plan is produced by the writing-plans skill. This is a preview to validate scope:

1. **Conversion tooling.** `scripts/convert_mlx_bf16.py` (CLI args for `--hf-source` and `--output-dir`, no defaults); one-time conversion executed by user against a chosen output path.
2. **`shards.yaml` schema + `ShardMap` change.** Add `model_id` top-level field; `ShardMap` exposes it; YAML loader validates non-empty.
3. **Eliminate hardcoded model literals.** `node.py` reads `hf_id` from `ShardMap.model_id`; `run_node.py`/`run_client.py`/`run_reference.py`/`generate_tier1_comparison_fixture.py` lose their default literals (resolve from loaded shards.yaml or fail loudly). Final grep returns zero hits in `src/`.
4. **Repo `config/shards.yaml` updated** to reference the bf16 path produced in Task 1.
5. **Memory smoke test** (verification #12). Runs first among bf16-loading tests to catch resource issues before fixture work.
6. **Phase 1 oracle regeneration.** Run `scripts/run_reference.py` against bf16; commit new `artifacts/ref/`.
7. **Phase 1 verification.** Tier 1 + Tier 2 green on bf16 oracle. Adjust Tier 2 tolerance if needed.
8. **Phase 3–6 slow buckets verification.** Run sequentially; fix any 4-bit-tolerant assertions revealed.
9. **MLX top-K fixture regeneration.** Run `generate_tier1_comparison_fixture.py` with backend=mlx; commit new fixture.
10. **PyTorch top-K fixture regeneration on Spark.** Same script with backend=pytorch; commit new fixture from Spark (Spark needs its own shards.yaml entry pointing at `google/gemma-4-26B-A4B-it`).
11. **Cross-backend agreement floor update.** Run test, observe new numbers, tighten floors.
12. **README + memory update.** Document new canonical model, the no-defaults policy, conversion procedure, where the legacy 4-bit string lives if a user wants it.

Estimated 12 tasks, similar to prior phases.

## 7. Open questions

None at design time. All scope decisions resolved during brainstorming:
- Routing-only scope chosen (Option A) for 7-C-3 overall.
- Bf16 everywhere chosen (Option 1) over precision tiers and shared-quant formats.
- Three-node Mac+Spark+3090 topology chosen for 7-C-3b demo.
- Verification bar chosen (Option D) — Tier 1 reference plus boundary allclose.
- Phase split into 7-C-3a (this spec) + 7-C-3b (separate brainstorm post-7-C-3a).
