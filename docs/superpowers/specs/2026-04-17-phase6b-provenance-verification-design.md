# Phase 6-B — Provenance-Chain Verification (topology enforcement)

**Status:** draft, 2026-04-17
**Scope:** Every forward pass of every token carries a hash-chained DAG of `ProvenanceEntry`s recording which node performed which operation, matching Gemma 4's known computation graph. Every node validates inbound chains at receive-time and rejects any chain that doesn't match the model's topology × the cluster's authorized ownership. Second sub-project of Phase 6 (Fault Tolerance & Verification).

## 1. Background & Decisions

### 1.1 Why now
Phase 1 defined `ProvenanceEntry` with an unused `hash: bytes = b""` field and a comment "for Phase 6 verification." Phase 5b added `live_owners_provider` — each node's runtime view of who owns which expert. Phase 6-A added retry using that view. Phase 6-B is the dormant-code-becomes-active moment: every hop now records its work as a provenance entry; every node validates inbound chains against the model's graph and the cluster's current authorized ownership.

The end goal per user framing: "allow inputs to pass hierarchically through layers that may be split across nodes ... enforce that the only valid paths are ones that follow the true map of the model architecture. Everything else should be rejected."

Phase 6-B is **topology / authorization enforcement**, not Byzantine-insider detection. A malicious insider with valid authorization could still forge valid-looking chains — that's Phase 7+ territory with cryptographic signatures.

### 1.2 Decomposition of Phase 6-B
- **6-B.1 Chain mechanism (in scope).** Hash protocol, `ProvenanceEntry` structure, wire fields, population at each op.
- **6-B.2 Topology validator (in scope).** Given a chain + the receiver's view of ShardMap + live ownership, is every entry's `node_id` authorized for its claimed op, and does the DAG shape match the model's graph?
- **6-B.3 Enforcement (in scope).** Receive-time rejection via `Error{ERR_INVALID_PROVENANCE}`.
- **6-B.4 Hash re-verification (deferred).** A replica holder sample-re-runs a past op and compares hashes to catch a node that lies about what it computed. Standalone follow-up.

This spec covers 6-B.1 + 6-B.2 + 6-B.3 — they're tightly coupled; without 6-B.3 the first two are audit logs nobody acts on.

### 1.3 Decisions

- **D1. Scope.** Topology / authorization enforcement. Out: Byzantine-insider forgery, KV-cache integrity, cross-token chain linking, hash re-verification (6-B.4 follow-up).

- **D2. Enforcement site.** Every node validates inbound provenance at receive-time, before any MLX compute. Matches the "only valid paths" framing and the no-central-coordinator invariant.

- **D3. Chain granularity: hybrid DAG matching the computation graph.** One entry per atomic MLX operation. For the canonical Gemma 4 26B A4B A4B config with 30 layers and 1 split layer at L=15 (top-k=8): 40 entries per token's forward pass (29 non-split OP_LAYER_ATOMIC + 11 split-layer ops + OP_EMBED + OP_FINALIZE).

- **D4. Op taxonomy** (see §2.1 for DAG diagram):
  - `OP_EMBED` — `embed_tokens` on the head.
  - `OP_LAYER_ATOMIC layer=L` — a full non-split decoder layer on one node.
  - `OP_ATTENTION_ROUTE layer=L` — split-layer attention + router producing `post_attn` and `top_k_ids/weights`.
  - `OP_SHARED_EXPERT layer=L` — dense-MLP branch; runs on same node as `OP_ATTENTION_ROUTE`.
  - `OP_EXPERT layer=L expert_id=E` — one expert's compute, possibly on a different node (replica-aware).
  - `OP_AGGREGATE layer=L` — routed-branch aggregation + `post_ffn_ln_2` + sum with shared; on same node as `OP_ATTENTION_ROUTE`.
  - `OP_FINALIZE` — `norm + LM head + softcap` on the tail.

- **D5. Hash algorithm.** BLAKE2b-256 via `hashlib.blake2b(..., digest_size=32)`. Cryptographic, stdlib, ~3× SHA-256 throughput.

- **D6. Hash content.**

  ```
  hash_i = BLAKE2b-256(
      concat(parent_hashes_i) ||
      utf8(node_id) ||
      op_descriptor_bytes ||
      output_tensor_bytes
  )
  ```

  `op_descriptor_bytes` = `(op_type:uint8 || layer_idx:uint32 || expert_id:uint32)` little-endian, unused fields zero.

  Input bytes elided: parent hashes transitively commit to the input (for linear ops, prev_hash commits to prev output = our input; for OP_AGGREGATE, all expert/shared hashes commit to all inputs).

- **D7. `node_id`.** The stable `shard.shard_id` from YAML (e.g., `"layer_0-10"`). Matches what `ShardMap` authorizes. Stable across process restarts.

- **D8. Validation rules** (every inbound message at every node):
  1. **Structure root/tail.** Chain begins with OP_EMBED. If the chain contains any OP_FINALIZE entry, it must be the last entry (OP_FINALIZE is always terminal). For mid-pipeline messages (`Activation` / `ExpertRequest`), the chain ends at the op that produced the tensor currently being forwarded and contains no OP_FINALIZE.
  2. **Layer completeness.** For every L in `[0, total_layers)` reached by the chain so far: exactly one "layer-L-completed" entry exists — either OP_LAYER_ATOMIC (non-split) or OP_AGGREGATE (split).
  3. **Split-layer DAG shape.** For each split L reached: exactly one OP_ATTENTION_ROUTE, one OP_SHARED_EXPERT, ≥1 OP_EXPERT with distinct `expert_id`s and `parent_hashes` referencing the OP_ATTENTION_ROUTE entry, and one OP_AGGREGATE whose `parent_hashes` reference shared + every expert.
  4. **Hash tail check.** Re-compute the tail entry's hash from its declared fields + the incoming tensor bytes; assert it equals the recorded `hash`. Older entries are trusted via transitive chain coverage (they were validated at their own receive-time hops).
  5. **Authorization** (per op_type):
     - OP_EMBED: `node_id` has `start_layer == 0`.
     - OP_LAYER_ATOMIC layer=L: `L ∈ [shard.start_layer, shard.end_layer)` AND L is not a split layer on that shard's view.
     - OP_ATTENTION_ROUTE / OP_SHARED_EXPERT / OP_AGGREGATE layer=L: `L ∈ [shard.start_layer, shard.end_layer)` AND L is a split layer.
     - OP_EXPERT layer=L expert_id=E: `node_id ∈ receiver.owners_of(L, E)` (Phase 5b live view).
     - OP_FINALIZE: `node_id` has `end_layer == total_layers`.

  Any rule violated → `Error{ERR_INVALID_PROVENANCE, is_final=true}`, close the connection. Same control flow as `ERR_SHARD_UNAVAILABLE`.

- **D9. Retry and migration compatibility.** 6-A retries and 5b migrations validate naturally: live `owners_of` is the source of truth, so a new replica or a retry-target peer passes authorization as soon as gossip propagates its ownership. During convergence windows a transient chain may reject; 6-A's retry loop can reroute around the temporarily-unauthorized peer.

- **D10. Gate.** `ENABLE_PROVENANCE=false` default. Opt-in until confidence in performance + compat is established. When off: no entries produced, no wire fields populated, no validation — behavior identical to pre-6-B.

- **D11. Correctness bar.**
  - **Bit-exact chain determinism.** Two independent runs with the same prompt + same shard config produce byte-identical provenance chains.
  - **No-op-ness on compute.** With `ENABLE_PROVENANCE=true`, Tier 1 tokens match the Phase 1 reference bit-exactly. Provenance is pure bookkeeping.
  - **Fault-injection sanity.** Corrupt one entry's hash byte → next hop rejects with `ERR_INVALID_PROVENANCE`.

- **D12. Non-goals (explicit).**
  - No signatures / key-management / PKI.
  - No hash re-verification (6-B.4 deferred).
  - No KV-cache integrity.
  - No provenance crossing token boundaries within a request (each token's forward pass gets its own fresh chain).
  - No replacement of the compute path — validation is strictly additive.
  - No new gossip surface.

## 2. Components

### 2.1 Computation graph / DAG diagram

```
OP_EMBED (head)
  │
  ├─ OP_LAYER_ATOMIC layer=0 (head)
  ├─ ... layer=1..9 ...
  ├─ OP_LAYER_ATOMIC layer=10 (mid)   — (activation crosses wire here)
  ├─ ... layer=11..14 ...
  │
  ├─ OP_ATTENTION_ROUTE layer=15 (mid)
  │   ├──────────── OP_SHARED_EXPERT layer=15 (mid)
  │   ├─ OP_EXPERT layer=15 expert_id=e₁ (owner_1)  — (post_attn crosses wire)
  │   ├─ OP_EXPERT layer=15 expert_id=e₂ (owner_2)
  │   │   ...
  │   └─ OP_EXPERT layer=15 expert_id=e_k (owner_k)
  │                                               (expert outputs cross wire back)
  ├─ OP_AGGREGATE layer=15 (mid)  — parents: shared + all experts
  │
  ├─ OP_LAYER_ATOMIC layer=16..19 (mid)
  ├─ OP_LAYER_ATOMIC layer=20 (tail)   — (activation crosses wire here)
  ├─ ... layer=21..29 ...
  │
  └─ OP_FINALIZE (tail)
        │
        ▼
    sampled token
```

40 entries total for the canonical config.

### 2.2 `src/model_shard/request.py`

`ProvenanceEntry` gains `parent_hashes: tuple[bytes, ...]` and a structured op descriptor. Current shape:

```python
@dataclass(frozen=True)
class ProvenanceEntry:
    shard_id: str            # unchanged (alias for node_id in 6-B terminology)
    node_id: str             # unchanged (retained for pre-6-B compat)
    timestamp: float
    hash: bytes = b""
```

New shape:

```python
class OpType(IntEnum):
    OP_UNSPECIFIED     = 0
    OP_EMBED           = 1
    OP_LAYER_ATOMIC    = 2
    OP_ATTENTION_ROUTE = 3
    OP_EXPERT          = 4
    OP_AGGREGATE       = 5
    OP_FINALIZE        = 6
    OP_SHARED_EXPERT   = 7


@dataclass(frozen=True)
class OpDescriptor:
    op_type: OpType
    layer_idx: int = 0
    expert_id: int = 0

    def pack(self) -> bytes:
        return struct.pack("<BII", int(self.op_type), self.layer_idx, self.expert_id)


@dataclass(frozen=True)
class ProvenanceEntry:
    shard_id: str            # unchanged
    node_id: str             # unchanged; equals shard_id in Phase 6-B
    timestamp: float
    hash: bytes = b""
    parent_hashes: tuple[bytes, ...] = ()
    op: OpDescriptor | None = None
```

`Request.append_provenance` gains `*, op: OpDescriptor, parent_hashes: tuple[bytes, ...]` kwargs (defaults preserve Phase 1 test behavior).

### 2.3 `src/model_shard/provenance.py` (new)

```python
"""Phase 6-B provenance hashing + validation helpers.

Pure (no MLX imports beyond bytes serialization via mlx_engine). Called
from node.py / expert_orchestrator.py at each op to produce ProvenanceEntry
instances, and at every inbound message to validate a chain."""


def compute_hash(
    parent_hashes: tuple[bytes, ...],
    node_id: str,
    op: OpDescriptor,
    output_bytes: bytes,
) -> bytes:
    """BLAKE2b-256 over the concat of parents + id + packed op + tensor bytes."""

def build_entry(
    *,
    node_id: str,
    op: OpDescriptor,
    output_tensor: mx.array,
    parent_entries: Iterable[ProvenanceEntry],
) -> ProvenanceEntry:
    """Construct a ProvenanceEntry from its inputs and the output tensor."""

class ProvenanceError(ValueError):
    """Raised by validate_chain on any D8 rule violation. The node handler
    converts this into Error{ERR_INVALID_PROVENANCE, is_final=true}."""


def validate_chain(
    chain: list[ProvenanceEntry],
    *,
    shard_map: ShardMap,
    total_layers: int,
    live_owners_of: Callable[[int, int], set[str]],
    split_layers_for_shard: Callable[[str], set[int]],
    tail_tensor_bytes: bytes | None,
) -> None:
    """Enforce D8 rules 1-5. Raises ProvenanceError on the first violation.
    If tail_tensor_bytes is provided, rule 4 (hash tail check) is run; for
    internal peer-to-peer chain snapshots where the receiver doesn't yet
    have the matching tensor, pass None to skip rule 4."""
```

### 2.4 Wire protocol — `proto/wire.proto`

New messages:

```proto
enum OpType {
  OP_TYPE_UNSPECIFIED     = 0;
  OP_EMBED                = 1;
  OP_LAYER_ATOMIC         = 2;
  OP_ATTENTION_ROUTE      = 3;
  OP_EXPERT               = 4;
  OP_AGGREGATE            = 5;
  OP_FINALIZE             = 6;
  OP_SHARED_EXPERT        = 7;
}

message OpDescriptorPb {
  OpType op_type     = 1;
  uint32 layer_idx   = 2;
  uint32 expert_id   = 3;
}

message ProvenanceEntryPb {
  bytes hash                   = 1;
  repeated bytes parent_hashes = 2;
  string node_id               = 3;
  OpDescriptorPb op            = 4;
  double timestamp             = 5;
}
```

New fields on existing payloads (next unused tags):

- `Activation`: `repeated ProvenanceEntryPb provenance = <next>;`
- `ExpertRequest`: `repeated ProvenanceEntryPb provenance = <next>;`
- `ExpertResponse`: `repeated ProvenanceEntryPb provenance = <next>;`

New error code: `ERR_INVALID_PROVENANCE = 6` in the `ErrorCode` enum.

Concrete tag numbers are assigned at plan-time based on current `wire.proto` state.

### 2.5 Integration points in compute code

Instrumentation lives in `src/model_shard/node.py` and `src/model_shard/expert_orchestrator.py`, gated by a new `ENABLE_PROVENANCE=true` env var.

- **Embed (head):** after `embed_tokens`, append OP_EMBED entry with parent_hashes=(); tensor = embedded hidden state.
- **Non-split layer execution (`run_layers`):** after each atomic layer call, append OP_LAYER_ATOMIC entry; parent = previous layer's entry.
- **Split-layer (`ExpertOrchestrator.run_split_layer`):** after Phase A, append OP_ATTENTION_ROUTE (parent = prev layer) and OP_SHARED_EXPERT (parent = OP_ATTENTION_ROUTE). Each expert computation (local or remote) produces OP_EXPERT; parent = OP_ATTENTION_ROUTE. After Phase C aggregation, append OP_AGGREGATE; parents = (OP_SHARED_EXPERT, OP_EXPERT×k).
  - **Remote OP_EXPERT entries come over the wire on `ExpertResponse.provenance`.** The orchestrator merges them into the per-request chain it carries.
- **Finalize (tail):** after `finalize`, append OP_FINALIZE entry; parent = last OP_LAYER_ATOMIC.

### 2.6 Chain carriage & validation sites

- **`Activation` outbound:** sender attaches current chain prefix (all entries for ops up to and including the one producing the tensor being forwarded).
- **`Activation` inbound:** receiver validates chain (rules 1-5 with tail_tensor_bytes = incoming bytes), then appends its own entries as it runs ops. If validation fails → `Error{ERR_INVALID_PROVENANCE, is_final=true}` back upstream.
- **`ExpertRequest` outbound:** fan-out attaches the chain up to OP_ATTENTION_ROUTE; expert peer validates (rules 1-5, tail = post_attn bytes).
- **`ExpertResponse`:** expert peer attaches the single OP_EXPERT entry(s) it just produced; orchestrator receives and merges into its running chain.
- **Client sampled-token delivery:** the head attaches the full chain to each outbound `SampledToken` (new `repeated ProvenanceEntryPb provenance` on that message too). Client receives but does not validate in 6-B (client-side validation is out of scope). Enables offline inspection + future 6-B.4.

Wire budget per hop: ~40 × 100 bytes ≈ 4 KB. Negligible vs. activation tensor payloads (5.6-40 KB per hop).

## 3. Integration with prior phases

| Prior phase | Interaction |
|---|---|
| Phase 1 | `Request.provenance` and `ProvenanceEntry` finally populated at runtime |
| Phase 2 | No change to SWIM; validation uses the membership view only indirectly through `owners_of` |
| Phase 3 | `ExpertOrchestrator.run_split_layer` gains the per-op instrumentation path |
| Phase 4 | `live_owners_provider` infrastructure reused for authorization check |
| Phase 5a | `held_ids_per_layer` has no direct tie — provenance looks at live ownership, not local holdings |
| Phase 5b | `owners_of` is the authorization oracle; migrations that add replicas naturally un-reject previously-unauthorized nodes |
| Phase 6-A | Retry reroutes preserve chain validity because the alternate peer is `owners_of(L, E)`-authorized |

## 4. Memory & Performance

- Per-hop hashing: ~40 × BLAKE2b-256 over ≤80 KB activation + ~24 bytes metadata. At ~1 GB/s BLAKE2b throughput, ~80 µs per entry × 40 = ~3 ms per forward pass. Not a bottleneck on decode (tokens/sec dominated by MLX compute).
- Wire overhead: ~4 KB per message. Trivial.
- Memory: ~4 KB per in-flight request-token chain, held in `Request.provenance`. Cleared at request end.

## 5. Testing

### 5.1 Fast unit tests (no model load)

`tests/test_provenance_hash.py`:
- `test_compute_hash_deterministic` — same inputs → same digest.
- `test_compute_hash_includes_parent` — changing any parent changes the output.
- `test_compute_hash_includes_node_id` / `op_descriptor` / `output_bytes` — each component is load-bearing (mutate one, hash changes).

`tests/test_provenance_validate.py`:
- `test_validate_accepts_wellformed_chain` — construct a synthetic 40-entry chain for the canonical config; passes.
- `test_validate_rejects_missing_embed` — chain with no OP_EMBED.
- `test_validate_rejects_missing_finalize_on_terminal` — terminal chain without OP_FINALIZE.
- `test_validate_rejects_skipped_layer` — chain with layer 12 missing.
- `test_validate_rejects_reordered_layer` — chain with layer 12 after layer 13.
- `test_validate_rejects_unauthorized_node_for_layer_atomic` — entry claims layer L on a shard whose range doesn't include L.
- `test_validate_rejects_unauthorized_expert_owner` — OP_EXPERT node_id not in `live_owners_of(L, E)`.
- `test_validate_rejects_missing_shared_expert` — split layer DAG missing OP_SHARED_EXPERT.
- `test_validate_rejects_aggregate_missing_parents` — OP_AGGREGATE parent_hashes don't include every OP_EXPERT.
- `test_validate_rejects_tampered_tail_hash` — recompute disagrees with recorded.

### 5.2 Fast correctness

`tests/test_provenance_integration_unit.py`:
- `test_split_layer_produces_wellformed_chain` — drive `ExpertOrchestrator._phase_b_with_retry` with fake experts; inspect the resulting chain, pass it through the validator, assert no raise.

### 5.3 Slow (model-loading)

`tests/test_provenance_tier1.py`:
- `test_tier1_bit_exact_with_provenance_enabled` — 3-node cluster, `ENABLE_PROVENANCE=true`, Phase 1 canonical prompts, tokens match reference bit-exactly (provenance is pure bookkeeping).
- `test_chain_is_deterministic_across_runs` — two runs of the same prompt produce byte-identical chains.
- `test_corrupted_chain_gets_rejected` — intercept an `Activation` on the wire (monkeypatch `_forward_activation` on the mid node to corrupt one byte of one hash), verify downstream receiver emits `ERR_INVALID_PROVENANCE` and the client gets a clean error (no hang).

### 5.4 Regression

- All Phase 1-6A fast + slow tests pass with `ENABLE_PROVENANCE=false` (default) — strictly unchanged behavior.
- `ENABLE_PROVENANCE=true` Tier 1 E2E passes on the Phase 4 shard config.

## 6. Risks & Mitigations

- **R1 — Transient convergence rejection.** A node receives a chain with an OP_EXPERT from a newly-migrated replica whose ownership hasn't propagated to the node yet. The node rejects with `ERR_INVALID_PROVENANCE`. Mitigation: 6-A retry filters the offender out of `live_owners_provider` and reroutes; a few hundred ms later gossip has converged. Acceptable.

- **R2 — Chain growth in multi-layer-split configs.** If future phases split more layers (e.g., every decoder layer), chain length grows to ~200+ entries per forward pass. Wire size ~20 KB per hop. Still <10% of activation size. Revisit if splits >5 layers become routine.

- **R3 — Hashing cost at large prompt lengths.** Prefill with 1024-token prompts → per-hop activation ≈ 1024 × 2816 × 2 bytes = 5.6 MB. 40 entries × 5.6 MB × 1 GB/s ≈ 220 ms per forward pass. Significant. Mitigation: for Phase 6-B scope, Tier 1 uses short prompts (≤8 tokens per 5a §7.5) so this doesn't hit. Production long-prompt deployments may need streaming/incremental BLAKE2b or Merkle-tree chunking.

- **R4 — Missing `_SHARED_EXPERT` in older decoder variants.** If a future model has no shared-expert branch at some split layers, the DAG shape check would falsely reject. Mitigation: `OpDescriptor.op_type` discrimination covers it — a layer without `OP_SHARED_EXPERT` in its set simply doesn't emit or expect that entry. The D8.3 rule's "exactly one OP_SHARED_EXPERT" is Gemma-specific; make it "at most one" to stay forward-compat.

- **R5 — Chain-as-audit replay attack.** A previously-valid chain from request A could in principle be replayed inside request B's `Activation` (wrong request_id, same hashes). Mitigation: the chain is scoped to a single forward pass — `request_id` + per-token position are implicit in the tensor bytes each entry hashes, so replaying in a different request would produce different downstream-hash expectations and reject at the first new layer. Verify empirically in a fast test.

- **R6 — (resolved in D8 rule 1).** Rule 1 no longer needs a terminal/non-terminal discriminator — it checks only two things: chain starts with OP_EMBED, and if OP_FINALIZE appears it must be last. Mid-pipeline messages simply don't carry OP_FINALIZE entries; the validator never needs to know whether it's the terminal hop.

## 7. References

- Phase 6-A spec: `docs/superpowers/specs/2026-04-17-phase6a-expert-retry-design.md` (§D5 live-owners; §D7 typed failures)
- Phase 5b spec: `docs/superpowers/specs/2026-04-17-phase5b-dynamic-migration-design.md` (§D9 live_owners_provider, §D10 OwnershipDelta)
- Phase 1 groundwork: `src/model_shard/request.py:14-37` — `ProvenanceEntry` + `Request.append_provenance`
- Spec §10 from the project's original gossip-moe spec: provenance chain motivation (reject invalid paths)
