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
- **D5 Correctness bar.** Strict Tier 1 reproduction — tokens emitted by the distributed pipeline must match the Phase 1 reference byte-for-byte. Achieved by sorting expert outputs by expert-id before the gated sum, mirroring mlx-vlm's internal MoE op order.
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
| `run_shared_expert` | `(lm, h, layer_idx) -> mx.array` | Always-local path. |
| `run_selected_experts` | `(lm, h, layer_idx, expert_ids: list[int]) -> dict[int, mx.array]` | Runs only experts hosted on this node. |
| `aggregate_experts` | `(expert_outputs: dict[int, mx.array], top_k_ids: list[int], top_k_weights: mx.array, shared_out: mx.array) -> mx.array` | Deterministic sum: sort by expert-id, gated sum, add shared. Bit-exact to the atomic layer call. |

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
  TensorSpec h_spec       = 5;
}

message ExpertResponse {
  uint32 protocol_version = 1;
  string request_id       = 2;
  uint32 layer_idx        = 3;
  repeated uint32 expert_ids = 4;  // same order as request; response tensor is stacked
  TensorSpec outputs_spec = 5;
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
6. Mid runs shared expert locally.
7. Mid gathers outputs. Sorts by expert-id. Computes `sum(w[i] * out[sorted_ids[i]] for i in 0..7) + shared_out`. This order must match mlx-vlm's `MoEBlock` op order bit-for-bit (§6 split-equivalence test is the proof).
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

## 8. Open Technical Question (to resolve during Task 1)

The mlx-vlm MoE forward may compute all 128 experts' gate/up/down on the activation tensor and mask by top-k, rather than the sparse "run only 8" pattern assumed above. If so, **Case X (masked-all):** `run_selected_experts` must run all experts on this node's share and let aggregation cancel the unused ones; bit-equivalence is preserved and the only cost is compute. **Case Y (sparse):** mlx-vlm truly runs only the selected experts; our prototype mirrors that directly and the split-equivalence test needs care around which 8-of-128 are present.

Task 1 of the implementation plan will read mlx-vlm's MoE source and resolve this. The design supports either case because the wire protocol carries explicit `expert_ids` in both directions and `aggregate_experts` is already id-indexed.

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
- mlx-vlm MoE source: to be read in Task 1 (resolves §8)
