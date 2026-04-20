# model_shard

Gossip-based distributed MoE inference. Phase 1 prototype — see [plan](../../.claude/plans/fluffy-mapping-flurry.md).

## Quickstart

```bash
uv sync --extra dev
uv run python -m grpc_tools.protoc -I proto --python_out=src/model_shard/_pb proto/wire.proto
uv run pytest
```

## Phase 2 status: Gossip Discovery — complete

Each node now runs a SWIM-style membership protocol over UDP (port `tcp_port + 1000`).
The head admits `BeginRequest`s only when every required shard is `ALIVE`; in-flight
requests fail with `Error{SHARD_UNAVAILABLE, is_final=true}` if a peer transitions
out of `ALIVE` mid-decode. Set `ENABLE_GOSSIP=false` to bypass and reproduce Phase 1
behavior. See `docs/superpowers/specs/2026-04-16-phase2-gossip-discovery-design.md`.

## Phase 3 status: Expert-Level Sharding (single layer) — complete

Layer 15's 128 routed experts are distributed round-robin across the three nodes via
the new `moe_experts` field in `config/shards.yaml`. The node hosting the layer's
attention block (`layer_10-20`) runs the router and fans out post-attention activations
to peer nodes via `ExpertRequest` over the existing TCP envelope transport; peer
responses are aggregated in top-k slot order for bit-strict Tier 1 reproduction.
In-flight peer failure surfaces as `ExpertRpcFailure` in the orchestrator and becomes
`Error{SHARD_UNAVAILABLE}` to the client; the Phase 2 membership observer aborts
pending RPCs immediately when a peer leaves `ALIVE`. Set `ENABLE_EXPERT_SHARD=false`
(default) to bypass and reproduce Phase 2 behavior. See
`docs/superpowers/specs/2026-04-16-phase3-expert-sharding-design.md`.

## Phase 5a status: Partial Expert Weight Loading — complete

A node can now load only the routed experts listed in its shard's `moe_experts`
YAML instead of the full 128-expert stack per layer. Opt-in via
`ENABLE_PARTIAL_LOAD=true`. Resident memory per shard drops from ~14 GB to
chassis (~4.5 GB) + `k/128 × 9 GB` for routed experts, which is the unlock for
eventual 24 GB-VRAM deployments. Correctness is proven by
`tests/test_partial_load_bit_exact_per_expert.py` (per-expert bit-exact vs
full load) and `tests/test_partial_load_split_equivalence.py` (three mod-3
sliced shards compose bit-exact to atomic layer 15). `run_selected_experts`
handles the global→local expert-id translation; stock mlx-vlm `Experts` /
`SwitchLinear` modules are untouched. See
`docs/superpowers/specs/2026-04-17-phase5a-partial-expert-loading-design.md`.

## Phase 5b status: Dynamic Expert Migration — complete

A node can now request expert weights from any peer over TCP and slot them
bit-exactly into its compact stacked tensor at runtime. Opt-in via
`ENABLE_DYNAMIC_MIGRATION=true` (requires `ENABLE_PARTIAL_LOAD=true`). Scope:
(A) per-(layer, expert) heat tracking as an EMA, (B) target-pull migration RPC,
and (D) decode-loop hang fix; (C) policy threshold is a simple stub. Each node
tracks expert activation heat; a background scanner periodically identifies
experts that exceed `MIGRATION_HEAT_THRESHOLD` and issues pull requests to peers
that hold the weights. Heat reports and ownership `ADD` deltas piggyback on
existing SWIM `Ping`/`Ack`/`PingReq`/`PingReqAck` messages; `ExpertOrchestrator`
routing resolves live owners via a `live_owners_provider` callback that unions
the bootstrap `ShardSpec` with gossip-observed `ADD` deltas. Correctness is
proven by `tests/test_migration_bit_exact_per_expert.py` (slice→attach bit-exact
on real Gemma weights) and `tests/test_migration_over_tcp.py` (same proof across
a real TCP round-trip). The decode-loop hang fix uses observer-triggered
queue-poison to unblock the head immediately when any peer leaves `ALIVE`
mid-decode; verified by `tests/test_decode_hang_fix_e2e.py`. Tier 1 regression:
`tests/test_partial_load_tier1_migration.py` runs all 5 canonical prompts with
both flags ON and confirms token-id bit-exact to the Phase 1 reference. Known
carryover from 5a §7.5: sort-path FP noise limits bit-exactness to B*Seq ≤ 7 or
prompts ≤ 8 tokens — documented in the spec. See
`docs/superpowers/specs/2026-04-17-phase5b-dynamic-migration-design.md`.

## Phase 6-A status: Expert-Peer Retry — complete

When a peer fails mid-fan-out, the node's local `ExpertOrchestrator` now retries to
an alternate replica rather than surfacing `ExpertRpcFailure` immediately. The retry
loop lives in `ExpertOrchestrator._phase_b_with_retry` inside `run_split_layer` Phase B.
Each invocation maintains a per-call excluded-peer set; on failure the failed peer is
added to that set and `live_owners_provider` is re-queried with exclusions applied,
so subsequent attempts land on a different replica. Partial outputs from peers that
already completed are preserved across the retry; only the failed slot is re-dispatched.
Decentralization is fully preserved: the retry decision is local to the node performing
the fan-out — no central coordinator is consulted.

Gate: `ENABLE_EXPERT_RETRY=true` (default). Env knobs: `EXPERT_RETRY_MAX_ATTEMPTS`
(default 3) and `EXPERT_RETRY_BACKOFF_MS` (default "100,500", comma-separated list of
per-attempt delays in milliseconds). Setting `ENABLE_EXPERT_RETRY=false` reverts to
Phase 3 behavior (immediate failure on any peer error).

Correctness proof: `tests/test_expert_retry_bit_exact.py` asserts `mx.array_equal`
between a no-failure run and a one-shot-failure-plus-retry run on real Gemma weights
across all 6 experts. This relies on Phase 5b's property that any valid replica of
expert E produces identical output to any other. E2E coverage:
`tests/test_expert_retry_e2e.py` kills a replica-holding shard mid-generation and
verifies the client exits cleanly — either via retry-carries-through or via the Phase 5b
Task 18 queue-poison fallback. In the canonical 3-node config every shard is
simultaneously a pipeline peer and an expert host, so killing any shard breaks the
activation pipeline; the E2E therefore exercises the no-hang fallback path. The pure
replica-preserving retry path is proven by the unit and bit-exact tests.

Non-goals: pipeline-peer failure (requires redundant-layer-range design in shards.yaml —
separate sub-project), head-peer failure (client-side story), Byzantine (Phase 6-B).
Phase 6 decomposes into three independent sub-projects: 6-A (retry, this), 6-B
(provenance verification), and 6-C (eviction + REMOVE OwnershipDelta). See
`docs/superpowers/specs/2026-04-17-phase6a-expert-retry-design.md`.

## Phase 6-C status: Expert Eviction — complete

A node can now evict migration-added experts under capacity pressure via
`OwnershipDelta{REMOVE}` gossip with last-writer-wins convergence on
`ts_unix_ms`. Gate: `ENABLE_EVICTION=true` (default on). Knob:
`MIGRATION_EVICT_COOLDOWN_SECONDS=30`.

`detach_expert` is the inverse of Phase 5b's `attach_expert`: it shrinks
the compact stacked tensor via a complementary-index `mx.take` under
`_MLX_COMPUTE_LOCK`, so tensor mutation is serialized with any in-flight
`ExpertRequest`. `MigrationScanner._maybe_evict_one` runs after the pull
pass under the same single-in-flight lock, fires only at capacity, and
selects the coldest-heat migration-added expert as the eviction victim.

Safety invariants: bootstrap-held experts are never evicted; a last-replica
local check refuses eviction if no other live owner is known; the 30-second
attach cooldown prevents attach/evict oscillation; compute-lock
serialization prevents tensor mutation under an in-flight compute.

Phase 5b's `_ownership_seen: set` (ADD-only) was promoted to
`_ownership_view_internal: dict[(shard, L, E), (action, ts_unix_ms)]` with
last-writer-wins convergence. The `owners_of()` contract is preserved — it
still returns the set of ADD shard_ids visible to callers.

Significant carry-forward bug fix discovered during Task 7 E2E: `Node.owners_of`
was disconnected from `MembershipRunner` gossip — only bootstrap and
self-announcements were visible; peer-announced deltas never reached the
node's ownership view. Fixed with an observer pattern:
`MembershipRunner.register_ownership_observer` fires callbacks after each
LWW acceptance, and `Node._on_gossip_ownership_delta` applies them via its
own LWW. This corrects a latent Phase 5b/6-A/6-B bug in which multi-node
ownership gossip was effectively invisible to `ExpertOrchestrator`
live-routing, Phase 6-A retry exclusion, and Phase 6-B provenance
authorization. A companion latent fix: `_handle_expert_request` now
consults `_live_experts` (runtime) rather than `self._shard.moe_experts`
(bootstrap), so migration-attached experts are served correctly and evicted
experts correctly return `ERR_WRONG_SHARD`.

Correctness proofs: `tests/test_partial_load_detach.py` (attach→detach
roundtrip byte-identical); `tests/test_ownership_view_convergence.py`
(ADD/REMOVE LWW convergence); `tests/test_eviction_e2e.py` (3-node cluster
attach+evict cycle with gossip convergence); and
`tests/test_eviction_race_with_expert_request.py` (post-eviction
`ExpertRequest` returns `ERR_WRONG_SHARD`). Non-goals: quorum last-replica,
two-phase tentative eviction, memory-pressure probing — Phase 7+. Phase 6
trilogy complete: 6-A retry, 6-B provenance, 6-C eviction all shipped. See
`docs/superpowers/specs/2026-04-18-phase6c-eviction-design.md`.

## Phase 7-A status: Backend Protocol + MLXBackend — complete

A `Backend` protocol plus an `MLXBackend` wrapper now sit between `Node` /
`ExpertOrchestrator` and every tensor-level operation, with zero behavioral
change on default MLX deployments. The stateful backend class owns the
`LoadedModel` internally; consumers receive opaque `Activation`, `Cache`,
`Mask`, and `TopK` handles and either pass them straight back into other
backend calls or serialize via `tensor_to_bytes`. `Node.__init__(backend=None)`
defaults to `MLXBackend()`; legacy `loaded_model=` callers remain supported via
`MLXBackend.from_loaded_model`. `ExpertOrchestrator` gains a `backend` field
and routes every compute call through `self.backend.X()`; a temporary
`backend=None` fallback preserves Phase 1–6 construction patterns and will be
removed in Phase 7-B. `mlx_engine.py` gained a public `run_layer_atomic`
helper plus a `mx_to_wire_dtype` alias so backends do not depend on private
names. Correctness is preserved: Tier 1 is bit-exact to the Phase 1 reference
under the default `MLXBackend`, and every Phase 1–6 E2E test (migration,
retry, provenance, eviction) passes unchanged. The point of Phase 7-A is the
seam: Phase 7-B will add a `PyTorchBackend` for CUDA / DGX Spark, and Phase
7-C a heterogeneous cluster with an `allclose` + top-1 correctness bar across
platforms. See
`docs/superpowers/specs/2026-04-19-phase7a-backend-protocol-design.md`.

## Phase 7-B status: PyTorchBackend + DGX Spark — complete

A `PyTorchBackend` now implements the full `Backend` protocol over HF
`transformers` `Gemma4ForCausalLM` loaded in bfloat16 on DGX Spark
(GB10 Grace Blackwell, SM_121, 128 GB unified LPDDR5X). The module
layout mirrors the MLX side: `pytorch_engine.py` holds the forward-pass
primitives (load, embed, cache, `run_layer_atomic`, finalize, wire
serialization), `pt_moe.py` holds the split-layer MoE helpers
(attention + route, shared expert, per-expert compute, aggregate), and
`pt_partial_load.py` holds the slice/attach/detach operations against
HF's stacked expert tensors. `backends/pytorch_backend.py` is a thin
delegation wrapper that owns the HF model instance. All 20 protocol
methods are parity with `MLXBackend`, including `slice_expert`,
`attach_expert`, and `detach_expert`, so Phase 5a/5b/6-C features
(partial load, dynamic migration, eviction) are available on Spark.
Backend selection is driven by the `MODEL_SHARD_BACKEND=pytorch|mlx`
env var, falling back to auto-detect (MLX on Apple Silicon, PyTorch
elsewhere). The Phase 7-A temporary shims are gone:
`ExpertOrchestrator.backend=None` fallback and `Node._lm` property both
deleted. Correctness bar: `tests/test_pytorch_tier1.py` asserts top-1
agreement against a fixture generated once on Spark and committed.
Cross-backend parity (MLX vs PyTorch) is deferred to Phase 7-C, along
with 4-bit quantization on PyTorch, heterogeneous clustering, and perf
tuning. See
`docs/superpowers/specs/2026-04-19-phase7b-pytorch-backend-design.md`.

## Phase 7-C-1 status: Real HF Gemma 4 forward integration — complete

Phase 7-C-1 closes the Phase 7-B synthetic-test gap by wiring
`PyTorchBackend` against HF `transformers`' real `Gemma4TextDecoderLayer.forward`
with the correct signatures, rather than the stub-shaped call sites that
Phase 7-B's unit tests exercised. Tasks 1-5 (landed Mac-side) handled
the architectural refactor from a reading of HF source: the cache kwarg
is `past_key_values` (plural) and the decoder returns a plain tensor
rather than a tuple; the router returns a 3-tuple `(probs, weights,
index)` from which only `weights` and `index` are consumed; the dense
MLP always runs alongside the MoE branch, and the two combine via
per-branch `post_feedforward_layernorm_{1,2}` summed, then an outer
`post_feedforward_layernorm` + residual + per-layer `layer_scalar`
multiply; `layer_type` lives on `self_attn`, not the decoder layer;
real configs wrap `Gemma4TextConfig` so nested access goes through
`config.get_text_config()`; `shared_kv_states={}` is required on
attention even when `num_kv_shared_layers=0`; and rotary embeddings are
per-layer-type, not per-layer-index. Implementation: `make_masks`
returns `(rotary_dict, attn_mask_dict)` keyed by layer_type through the
existing `masks` tuple slot; `run_layer_atomic` and
`pt_moe.run_attention_and_route` call the HF layer / attention with full
kwargs; the orchestrator applies outer layernorm + residual +
`layer_scalar` after the per-position aggregate loop via a backend-aware
layer accessor so the MLX path is unaffected. Task 6b (landed on DGX
Spark) caught four more real-HF divergences that synthetic tests had
masked: `model.model` is a `Gemma4Model` multimodal wrapper with the
text model nested as `.language_model` (added a `_text_model()` helper
used across `pytorch_engine`, `pt_moe`, `pt_partial_load`, and the
orchestrator); `num_layers` must read via `len(_text_model(...).layers)`
rather than `config.num_hidden_layers` (the multimodal config doesn't
expose it); `final_logit_softcapping = 30.0` on the real 26B model and
must be applied after `lm_head`; and the fixture generator was rewritten
to greedy-decode through `PyTorchBackend` itself rather than HF's
`model.generate()`, so Tier-1 is an *internal* regression (our forward
path doesn't drift) rather than a cross-framework equivalence check.
Testing: 16 synthetic units in `test_pytorch_engine.py`, 8 in
`test_pt_moe_unit.py`, 3 slow CPU integration tests in
`test_pytorch_tiny_hf_integration.py` that instantiate a tiny real
`Gemma4ForCausalLM` from config and verify end-to-end plumbing, and
`tests/test_pytorch_tier1.py` now a permanent Spark-side regression
against the committed fixture. MLX slow regression bucket (all 6 files)
stays green. Cross-framework parity (MLX ↔ PyTorch ↔ HF) is explicitly
deferred to Phase 7-C-2. See
`docs/superpowers/specs/2026-04-19-phase7c1-real-hf-integration-design.md`.

## Phase 6-B status: Provenance Verification — complete

Every forward pass now carries a hash-chained DAG of `ProvenanceEntry` records that
mirrors Gemma's computation graph. Each entry is a BLAKE2b-256 digest over
`(parents || node_id || op_descriptor || output_bytes)`. In the canonical 3-node ×
1-split-layer × top-8 config this produces 40 entries per forward pass: one per pipeline
hop, one per expert invocation, and one final aggregation entry. Every node validates
inbound chains at receive-time and rejects any entry whose hash does not match the
declared parents; an invalid chain returns `Error{ERR_INVALID_PROVENANCE}` to the
client immediately.

The goal is topology and authorization enforcement — reject any path that doesn't
match the model's true computation graph — not Byzantine-insider detection. Phase 5b's
`owners_of` is the authorization oracle: a node is only a valid parent for a given
(layer, expert) pair if it appears in `owners_of(L, E)`. Phase 6-A's retries validate
naturally because the retry target is always drawn from `owners_of(L, E)`, so the
chain stays structurally correct across retry hops.

Gate: `ENABLE_PROVENANCE=true` (default off).

Correctness proofs: `tests/test_provenance_tier1.py` asserts bit-exact Tier 1 token
output against the Phase 1 reference with provenance enabled, proving provenance is
pure bookkeeping with zero compute effect. `tests/test_provenance_determinism.py`
confirms that identical inputs produce identical hashes. `tests/test_provenance_rejection.py`
corrupts one byte of a chain entry and verifies the next hop rejects with
`ERR_INVALID_PROVENANCE` and the client receives the error cleanly with no hang.

A gap filled during Task 10: the tail-to-head-to-client error-propagation path for
mid-pipeline rejections was absent. Task 10 added `_handle_upstream_error` on the
head node to forward errors to the client and poison the decode queue; this is a
broader improvement beyond provenance that previously would have caused mid-pipeline
errors to hang the decode loop silently.

Non-goals: cryptographic signatures, hash re-verification by sample re-run (Phase
6-B.4 follow-up), KV-cache integrity, cross-token chain linking. See
`docs/superpowers/specs/2026-04-17-phase6b-provenance-verification-design.md`.

## Phase 4 status: Load-Aware Routing — complete

Nodes now gossip a compact queue-depth EMA to each other via `LoadReport` piggybacked
on existing SWIM `Ping`/`Ack` messages. When `moe_experts` in `config/shards.yaml`
lists an expert on multiple shards, `ExpertOrchestrator` routes each top-k dispatch
to the less-loaded candidate via power-of-two-choices. The default config overlaps
experts 0, 1, and 2 across two shards each for a live multi-candidate scenario;
routing correctness is verified by `tests/test_routing_correctness.py`, and gossip
delivery is verified end-to-end by `tests/test_expert_rpc_load_shift.py` via a new
`/loads` debug endpoint (served alongside `/membership` at `tcp_port + 2000`). No
new env var — the behavior auto-activates when `moe_experts` has overlapping entries.
See `docs/superpowers/specs/2026-04-16-phase4-load-aware-routing-design.md`.
