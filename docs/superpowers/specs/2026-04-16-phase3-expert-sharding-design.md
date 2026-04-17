# Phase 3 — Expert-Level Sharding (Single-Layer Prototype)

**Status:** draft, 2026-04-16
**Scope:** Prove the expert-level fan-out / fan-in pattern on *one* MoE layer (layer 15) while keeping Phases 1–2 untouched for the other 29 layers. A green Phase 3 is the minimum demonstration that individual experts in Gemma 4 26B A4B can be independently addressed and distributed, without committing to gossiped placement or load-aware routing.

## 1. Background & Decisions

### 1.1 Model surface
Gemma 4 26B A4B has 30 transformer layers, each with a MoE block: 1 shared expert (always active, 3× size), 128 routed experts, top-8 routing. Today every layer runs atomically: `layer(h, mask, cache) = attention(h) → router → 8 experts + shared → aggregate`. Phase 3 breaks open one such layer.

### 1.2 Decisions (recorded here for downstream skills)
- **D1 Minimum viable slice.** Prototype on a single layer (§1 "Target layer: 15"). Generalization to all layers is out of scope — mechanical once this is green.
- **D2 Expert placement.** Round-robin by id: node `i` of 3 nodes owns experts `{e | e % 3 == i}`. Hardcoded for layer 15; replaces a gossiped shard map in Phase 4+.
- **D3 Shared expert.** Replicated on all 3 nodes (weights already loaded because nodes load the full model). Aggregation always runs locally, no fan-out.
- **D4 Aggregation site.** The node that owns the attention block for a given layer also owns the router and the aggregator. For layer 15 that is the mid node (layers 10-20 territory).
- **D5 Correctness bar.** Strict Tier 1 reproduction — tokens emitted by the distributed pipeline must match the Phase 1 reference byte-for-byte. Achieved by reproducing mlx-vlm's two-branch MoE op order exactly (see §8): pair each top-k weight with its slot's expert output (top-k slot order, not id-sorted), sum across the top-k axis, apply `post_feedforward_layernorm_2`, then add the separately-computed dense branch `h1 = post_feedforward_layernorm_1(mlp(pre_feedforward_layernorm(h)))`.
- **D6 Failure semantics.** Hard-fail, consistent with Phase 2. If an `ExpertRequest` RPC fails (TCP broken, timeout, observer fires on peer leaving ALIVE), the aggregator emits `Error{SHARD_UNAVAILABLE, is_final=true}` to the client. No retry, no 7/8 degradation.
- **D7 Transport.** Reuse Phase 1's length-prefixed TCP framing and protobuf `Envelope` oneof. Add two new oneof cases (`ExpertRequest`, `ExpertResponse`). No new transport.

### 1.3 Non-goals (explicit)
- Expert replication / heat tracking → Phase 4
- Multiple expert-sharded layers → trivial generalization once this is green
- Gossiped expert placement → Phase 4
- Dynamic migration → Phase 5
- Pipelining expert compute across layers → Phase 4+
- Graceful degradation with < top-k experts available → not planned

## 2. Topology

```
head(layers 0-10; experts {0,3,6,…,126} of L15) ─┐
                                                  ├─ fan-in to mid
mid (layers 10-20; experts {1,4,7,…,127} of L15; ─┤   (aggregator)
     attention+router+aggregator for L15)         │
                                                  │
tail(layers 20-30; experts {2,5,8,…,125} of L15) ─┘
```

Each node additionally holds the shared expert for L15 (D3). Phase 1's linear activation pipeline (head → mid → tail → head for sampled tokens) is unchanged; the expert fan-out is a lateral detour at layer 15 only.

## 3. Components

### 3.1 `src/model_shard/moe.py` (new)

Pure functions, no network. Mirrors mlx-vlm's MoE op order exactly.

| Function | Signature | Purpose |
|---|---|---|
| `run_attention_and_route` | `(lm, h, layer_idx, cache, masks) -> (post_attn_h, top_k_ids, top_k_weights)` | Pre-expert half of a single layer. |
| `run_shared_expert` | `(lm, h, layer_idx) -> mx.array` | Always-local: returns the dense-branch output `h1 = post_feedforward_layernorm_1(mlp(pre_feedforward_layernorm(h)))`. Despite the name, this is `layer.mlp` (3× intermediate), not a separate shared-expert module — see §8. |
| `run_selected_experts` | `(lm, h, layer_idx, expert_ids: list[int]) -> dict[int, mx.array]` | Runs only experts hosted on this node. Returns per-expert `SwitchGLU`-style outputs before weight-application and before `post_feedforward_layernorm_2` — the weight pairing and post-norm happen in `aggregate_experts`. |
| `aggregate_experts` | `(expert_outputs: dict[int, mx.array], top_k_ids: list[int], top_k_weights: mx.array, shared_out: mx.array) -> mx.array` | `shared_out + post_feedforward_layernorm_2(Σ_j top_k_weights[...,j,:] * expert_outputs[top_k_ids[j]])`. Slot-order iteration (`j` from 0 to k-1), not id-sorted. Bit-exact to the atomic layer call when the op sequence matches mlx-vlm §8. |

### 3.2 `src/model_shard/mlx_engine.py` modification

`run_layers` becomes aware of expert-split layers via a new `split_layers: set[int]` argument (default empty). For `i in split_layers`, the inner loop calls into a collaborator (`ExpertOrchestrator`, §3.3) instead of `layer(...)`. All other layers run atomically exactly as today.

### 3.3 `src/model_shard/expert_orchestrator.py` (new)

Owns the per-layer fan-out. Constructed with a shard-map view of who hosts which experts and a peer-RPC callable.

```python
class ExpertOrchestrator:
    def run_split_layer(
        self,
        h: mx.array,
        layer_idx: int,
        cache_slot: Any,
        masks: MaskPair,
    ) -> mx.array:
        post_attn_h, top_k_ids, top_k_weights = run_attention_and_route(...)
        by_node = group_expert_ids_by_owner(top_k_ids)           # {node_id: [expert_id,...]}
        local_ids = by_node.pop(self_node_id, [])
        # Fan out in parallel.
        local_futures = submit_local(post_attn_h, layer_idx, local_ids)
        remote_futures = [
            submit_remote(peer, post_attn_h, layer_idx, ids)
            for peer, ids in by_node.items()
        ]
        shared_out = run_shared_expert(...)
        outputs = gather(local_futures, remote_futures)          # dict[int, mx.array]
        return aggregate_experts(outputs, top_k_ids, top_k_weights, shared_out)
```

Gather uses a per-request condition variable fed by the node's existing inbound-message loop (same pattern as head's SampledToken queue from Phase 1). RPC timeout = Phase 2's SUSPECT deadline so peer death and RPC timeout line up.

### 3.4 `src/model_shard/node.py` additions

- New inbound handler for `ExpertRequest`: decode tensor, call `run_selected_experts`, encode response. Runs on the existing inbound-connection thread pool — no new threads.
- Head and tail must accept `ExpertRequest` for layer 15 even though they don't run layer 15's attention. This is new: a node now serves *two* kinds of RPCs — activation forwarding (Phase 1) and expert compute (Phase 3).
- Mid gains an `ExpertOrchestrator` instance. Head and tail do not.

### 3.5 `config/shards.yaml` extension

```yaml
shards:
  layer_0-10:
    host: 127.0.0.1
    port: 9001
    start_layer: 0
    end_layer: 10
    moe_experts:
      15: [0, 3, 6, 9, 12, ..., 126]   # experts on this node for layer 15
  # ...
```

`moe_experts` is optional. Absent → node holds no split-layer experts (layer still runs atomically where it was pinned). Present → node serves `ExpertRequest{layer_idx in moe_experts}` RPCs.

## 4. Wire Protocol

Add two `Envelope.oneof` cases to `proto/wire.proto`:

```proto
message ExpertRequest {
  uint32 protocol_version = 1;
  string request_id       = 2;
  uint32 layer_idx        = 3;
  repeated uint32 expert_ids = 4;
  // tensor post_attn_h carried out-of-band (shape + dtype in oneof header)
  TensorDescriptor h_spec       = 5;
}

message ExpertResponse {
  uint32 protocol_version = 1;
  string request_id       = 2;
  uint32 layer_idx        = 3;
  repeated uint32 expert_ids = 4;  // same order as request; response tensor is stacked
  TensorDescriptor outputs_spec = 5;
}
```

Tensor payload uses the existing out-of-band length-prefixed frame. `ExpertResponse` stacks the per-expert outputs on a new dim: shape `[B, L, len(expert_ids), hidden]`. The aggregator unstacks.

`Error{SHARD_UNAVAILABLE}` is reused unchanged.

## 5. Data Flow (one token through layer 15)

1. Activation `h` arrives at mid from head (layer 14 → 15 boundary), as in Phase 2.
2. Mid's `run_layers` sees `i=15 ∈ split_layers`, calls `ExpertOrchestrator.run_split_layer(h, 15, cache[15], masks)`.
3. Mid runs attention + router locally, producing `post_attn_h`, `top_k_ids` (8), `top_k_weights`.
4. Partition by owner: e.g. `{head: [3,6], mid: [4,7], tail: [2,5,8]}` (three of the top-8 may coincide on one node; batch them).
5. Mid sends one `ExpertRequest` to head (for its 2 experts), one to tail (for its 3). Runs its own 2 experts in parallel on the MLX device.
6. Mid runs the dense-branch (so-called "shared expert") path locally: `h1 = post_feedforward_layernorm_1(mlp(pre_feedforward_layernorm(post_attn_h)))`.
7. Mid gathers per-expert outputs. Iterates top-k in slot order (j = 0..7), multiplies `expert_outputs[top_k_ids[j]]` by `top_k_weights[..., j:j+1]`, sums across j, applies `post_feedforward_layernorm_2`, adds `h1`. This order must match mlx-vlm's `DecoderLayer` op order bit-for-bit (§6 split-equivalence test is the proof).
8. Result continues to layer 16 atomically.

### 5.1 Concurrency
- Local experts: MLX graph, one `mx.eval` at the aggregation point.
- Remote experts: one request per owning peer, fan-out parallel, gather blocks on all responses (request-id-keyed condition variable).
- Attention + router and shared expert can run while remote RPCs are in flight. Aggregation is the join point.

## 6. Testing Strategy

### 6.1 Fast tests (no model)
- `test_moe_group_expert_ids_by_owner` — partition helper.
- `test_moe_aggregate_order_is_id_sorted` — fixture tensors, deterministic result.
- `test_expert_request_envelope_roundtrip` — protobuf + transport layer.
- `test_shard_map_parses_moe_experts_key` — YAML extension.

### 6.2 Slow tests (load model)
- **Split equivalence (load-bearing):** `test_moe_split_equivalent_to_atomic_layer15` — run layer 15 atomically, run it via the split functions (`run_attention_and_route` → `run_selected_experts` for all 128 → `run_shared_expert` → `aggregate_experts`), assert bit-equality. Pure math, no network. If this fails, no other test can succeed.
- **End-to-end Tier 1:** `test_tier1_distributed_with_expert_split_layer15` — 5 canonical prompts, 3-node pipeline with layer 15 split across nodes, tokens must match Phase 1 reference exactly.
- **End-to-end Tier 2:** `test_tier2_hidden_with_expert_split_layer15` — hidden state at layer boundaries must be within the Phase 1 tolerance.
- **Failure:** `test_expert_rpc_failure_emits_shard_unavailable` — kill head mid-decode while a layer 15 request is pending an `ExpertRequest` RPC; client must receive `Error{SHARD_UNAVAILABLE, is_final=true}` within the Phase 2 SUSPECT window.

### 6.3 Regression
- All Phase 1 slow tests continue to pass unchanged (split_layers=∅ by default).
- All Phase 2 E2E tests continue to pass.

## 7. Migration / Rollback

Like Phase 2, Phase 3 is opt-in. A new env var `ENABLE_EXPERT_SHARD=false` (default) reverts to atomic layer-15 compute. `ENABLE_EXPERT_SHARD=true` plus non-empty `moe_experts` entries in `shards.yaml` activates the split path.

## 8. MoE Forward — Resolved (2026-04-16)

mlx-vlm's `Experts` class (in `.venv/lib/python3.13/site-packages/mlx_vlm/models/gemma4/language.py`, lines 103-130) implements the MoE forward as **sparse**. It delegates to `mlx_lm.models.switch_layers.SwitchGLU` (lines 160-199 of `switch_layers.py`), which calls `mx.gather_mm` / `mx.gather_qmm` with `rhs_indices=top_k_indices` against a stacked weight tensor of shape `(num_experts, output_dims, input_dims)`. Only the 8 selected experts' rows are gathered and multiplied per token — all 128 experts' weights are resident, but only the top-k slice is computed. When `indices.size >= 64`, the path additionally sorts tokens by expert id (`_gather_sort`) for coalesced access and unsorts after `down_proj` (`_scatter_unsort`).

The atomic MoE op sequence inside one layer (Gemma4 `DecoderLayer.__call__`, language.py lines 300-352) is actually a two-branch block — **not** a classic "shared expert + gated sum" with a single pre-norm. Ordered:

1. `h1 = post_feedforward_layernorm_1( mlp( pre_feedforward_layernorm(h) ) )` — dense MLP branch. In the 26B config `intermediate_size=2112` and `moe_intermediate_size=704`, i.e. `self.mlp` is exactly 3× the routed-expert intermediate. This is the "shared expert (3× size)" referenced in §1.1.
2. `top_k_indices, top_k_weights = router(h)` where the router already L1-renormalizes `top_k_weights` and multiplies by `per_expert_scale[top_k_indices]`.
3. `h2 = post_feedforward_layernorm_2( experts( pre_feedforward_layernorm_2(h), top_k_indices, top_k_weights ) )`. Inside `Experts.__call__` the per-token gated sum is `(expert_out * top_k_weights[..., None]).sum(axis=-2)` — i.e. the weighted sum across the top-k axis happens **inside** the experts module.
4. `h = h1 + h2`; then `h = residual + post_feedforward_layernorm(h)`; then optional per-layer-input gating and `* layer_scalar`.

So the two branches are summed **as equal peers** after each has gone through its own post-norm. There is no "shared × scale + gated sum" compositing; it's `norm1(mlp(norm0(h))) + norm2(experts(norm3(h), …))`.

Phase 3 implications:
- `run_selected_experts` will **run only the selected-and-locally-hosted experts** (the intersection of `top_k_indices` with this node's owned expert set), using a `gather_mm`-style per-token call. Each node passes that intersection as the `indices` argument to a `SwitchGLU`-equivalent over its local expert weight slab. No node ever computes all 128.
- `aggregate_experts` op order: the gated sum across top-k is bit-equivalent to mlx-vlm's only if we reproduce `(expert_out_sorted_by_top_k_position * top_k_weights[..., None]).sum(axis=-2)` **before** applying `post_feedforward_layernorm_2`, then add the separately-computed `post_feedforward_layernorm_1(mlp(...))` dense-branch output. Translation: `aggregate_experts` must return `h1 + post_feedforward_layernorm_2( gated_sum_over_top_k(expert_outputs) )`, where the gated sum iterates over the top-k slot order (not ascending expert-id order). D5's "sort by expert-id before the gated sum" in §1.2 is **wrong** and must be revisited in Task 2 — addition is commutative but `post_feedforward_layernorm_2` is not, and more importantly the weights are paired to top-k positions, so we must preserve the top-k slot→expert-id mapping rather than resort by id. The `aggregate_experts` signature in §3.1 already takes `top_k_ids` and `top_k_weights` separately, which is sufficient; only the comment/docstring about "sort by expert-id" is stale.
- Output dtype: **bf16 throughout**. The model is loaded in bfloat16 (`"dtype": "bfloat16"` in both the top-level and `text_config` of `config.json`); `mx.gather_mm` / `mx.gather_qmm` emit in the activation dtype; no fp32 accumulation is used by mlx-vlm. Wire frames for `ExpertRequest`/`ExpertResponse` tensors should carry bf16 payloads.

Caveat: the "shared expert" terminology in §1.1 and §1.2-D3 refers to `self.mlp` (the dense MLP branch), not a 129th expert module. D3 ("shared expert replicated on all 3 nodes") still holds because `self.mlp`'s weights are part of every node's full-model load; no change to placement strategy. Task 2 should update `run_shared_expert` to literally invoke `layer.mlp` wrapped in the two norms, and revise D5's stale "sort by expert-id" note.

## 9. Acceptance Criteria (for Phase 3 complete)

1. `uv run pytest` — all fast tests pass.
2. `uv run pytest -m slow` — all slow tests pass, including the four new Phase 3 tests.
3. `ENABLE_GOSSIP=true ENABLE_EXPERT_SHARD=true uv run pytest -m slow` — all slow tests pass with both Phase 2 and Phase 3 features active.
4. Manual 3-node subprocess smoke: 5 canonical prompts generate tokens identical to the Phase 1 reference.
5. `ruff check`, `mypy` — clean.
6. README updated with Phase 3 status paragraph.

## 10. References

- Phase 1 plan & architecture: `memory/phase1_architecture.md`
- Phase 2 spec: `docs/superpowers/specs/2026-04-16-phase2-gossip-discovery-design.md`
- Gemma 4 26B A4B architecture facts: `memory/gemma4_26b_architecture.md`
- Gossip MoE full spec: `/Users/lukechang/Downloads/gossip-moe-inference-spec.md` §3.3, §6.4
- mlx-vlm MoE source: `.venv/lib/python3.13/site-packages/mlx_vlm/models/gemma4/language.py` (`Experts`, `Router`, `DecoderLayer`) and `.venv/lib/python3.13/site-packages/mlx_lm/models/switch_layers.py` (`SwitchGLU`, `SwitchLinear.__call__` using `mx.gather_mm` / `mx.gather_qmm`) — read in Task 1, §8 resolved.
