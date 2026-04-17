# Gossip-Based Distributed MoE Inference

## Project Specification — Draft v0.1

**Date:** April 2026
**Status:** Brainstorm / Pre-prototype

---

## 1. Vision

A decentralized inference engine that distributes Mixture-of-Experts model execution across heterogeneous hardware using gossip-protocol coordination. No central scheduler. No master node. Nodes self-organize, discover available model shards, route requests along valid computation paths, and adapt dynamically as the network changes.

The system treats model shards like content in a CDN — popular experts get replicated to where they're needed, rarely-used experts persist on fewer nodes, and the network learns its own topology and load characteristics through gossip propagation.

### Target Model: Gemma 4 26B A4B (MoE)

The first prototype targets Google's Gemma 4 26B A4B, an open-weight Mixture-of-Experts model released April 2025 under the Apache 2.0 license. Its architecture is particularly well-suited to this system:

- **26B total parameters, ~3.8B active per token** — large enough that distributing is useful, sparse enough that per-token compute is modest
- **128 fine-grained experts per MoE layer, top-8 routing + 1 shared expert** — the high expert count creates a natural granularity for distribution, and the shared expert provides a stable "always-on" baseline
- **Shared expert is 3× the size of regular experts** — contains general knowledge that's always activated; a natural candidate for replication across all nodes
- **Hybrid attention: 5:1 local/global pattern** — alternating local sliding-window (1024 tokens) and global full-attention layers with unified K/V and Proportional RoPE (p-RoPE)
- **256K token context window**
- **Per-Layer Embeddings (PLE)** — each decoder layer has its own small embedding lookup, adding a per-layer conditioning signal
- **~52 GB at bf16, ~14–16 GB at 4-bit quantization** — too large for a single consumer GPU at full precision, but easily distributable across 2–4 nodes; fits a single node when quantized

### Why Gemma 4 26B A4B Is Ideal for This Architecture

The 128-expert design with top-8 routing creates a natural distribution surface: for any given token, only 8 of 128 experts activate, meaning ~94% of expert parameters are idle per forward pass. In a distributed setting, this means a node only needs to hold the experts that are likely to be routed to it — not the entire expert pool. The gossip protocol can track which experts are where and how frequently they're activated, enabling demand-driven placement.

The shared expert (always active, 3× size) serves as a natural anchor — it should be replicated everywhere since it's needed for every token. The 120 routed experts are the dynamic, distributable component.

---

## 2. Architecture Overview

### 2.1 Conceptual Layers

```
┌─────────────────────────────────────────────────┐
│           Model Sharding Blueprint              │
│   (Declarative description of computation DAG)  │
├─────────────────────────────────────────────────┤
│           DAG Router & Request Manager          │
│   (Path selection, load-aware routing, batching)│
├─────────────────────────────────────────────────┤
│           Gossip Protocol                       │
│   Cold plane: membership, shard maps, health    │
│   Hot plane: load signals, expert activation    │
├─────────────────────────────────────────────────┤
│           Transport Layer                       │
│   (Pluggable: TCP/QUIC for WAN, RDMA for LAN)  │
└─────────────────────────────────────────────────┘
```

### 2.2 Core Abstractions

**Shard**: A named, self-contained unit of model computation. For Gemma 4, natural shard boundaries include:

- Embedding layer + PLE lookup tables
- Attention blocks (grouped by local/global type)
- Individual MoE experts (the 128 routed experts)
- Shared expert (always-on, replicated)
- Router/gating network per MoE layer
- LM head / output projection

**Node**: A compute participant. Stores one or more shards, maintains a gossip-informed view of the network, and processes requests for its assigned computation stages.

**Request**: A uniquely identified inference job (a single forward pass for one or more tokens) that traverses the computation DAG. Carries its own provenance chain.

**Provenance Chain**: An append-only record attached to each request, logging which node performed which computation step, enabling verification at the terminal node.

---

## 3. Model Sharding Blueprint

### 3.1 Computation DAG for Gemma 4 26B A4B

Each decoder layer in Gemma 4 follows this structure:

```
Input Activations + PLE Signal
        │
        ▼
┌─────────────────┐
│   Attention      │  (local sliding-window OR global full-attention)
│   + RMSNorm      │  (unified K/V on global layers, p-RoPE)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Router/Gate    │  → selects top-8 of 128 experts
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│Shared  │ │Expert  │ × 8 (selected)
│Expert  │ │  FFNs  │
│(3× sz) │ │        │
└────┬───┘ └────┬───┘
     │          │
     └────┬─────┘
          ▼
    Weighted Sum → Output Activations
```

### 3.2 Sharding Strategy

The blueprint defines shards as metadata objects:

```
Shard {
    id:               string        // e.g., "layer_12.expert_47"
    stage_type:       enum          // EMBEDDING | ATTENTION | ROUTER | EXPERT | SHARED_EXPERT | LM_HEAD
    layer_index:      int           // which transformer layer
    expert_index:     int?          // which expert (null for non-expert stages)
    input_tensors:    TensorSpec[]  // shape, dtype of expected inputs
    output_tensors:   TensorSpec[]  // shape, dtype of outputs
    memory_estimate:  bytes         // VRAM/RAM needed to hold this shard
    compute_estimate: flops         // estimated FLOPs per forward pass
    dependencies:     ShardId[]     // which shards must complete before this one
    is_conditional:   bool          // true for routed experts (only activated when selected)
}
```

For Gemma 4 26B A4B, this produces roughly:

| Component | Count | Nature | Distribution Strategy |
|---|---|---|---|
| Embedding + PLE | 1 | Always active | Replicate to entry nodes |
| Attention blocks | ~30–35 | Always active, sequential | Pipeline across nodes |
| Router/gate networks | ~30–35 | Always active, lightweight | Co-locate with attention |
| Shared experts | ~30–35 | Always active, 3× size | Replicate broadly |
| Routed experts | ~30–35 layers × 128 = 3,840–4,480 | Conditionally active (top-8) | Distribute by demand |
| LM head | 1 | Always active | Place on exit node(s) |

The routed experts are the primary distribution target — they represent the vast majority of parameters but are individually small and only conditionally needed.

### 3.3 Granularity Levels

The system should support multiple sharding granularities:

- **Coarse** (prototype): Shard by layer groups (e.g., layers 0–9 on Node A, 10–19 on Node B). Simple pipeline parallelism. All experts for those layers live on the same node.
- **Medium**: Shard attention blocks separately from expert pools. Attention stays pipelined, experts get distributed across nodes.
- **Fine** (target): Individual experts are independently addressable. The router runs on one node, determines top-8, and dispatches to whichever nodes hold those specific experts. This is where the gossip-based coordination really pays off.

---

## 4. Node Structure

### 4.1 Node State

```
Node {
    node_id:          UUID
    capabilities: {
        compute_type:   enum[]      // GPU_CUDA, GPU_ROCM, CPU, APPLE_METAL
        vram_total:     bytes
        vram_available: bytes
        ram_total:      bytes
        network_bandwidth: bps      // measured, not theoretical
    }
    shards_held:      Map<ShardId, ShardState>
    shard_map:        Map<ShardId, Set<NodeId>>   // gossip-learned: who holds what
    node_health:      Map<NodeId, HealthRecord>   // gossip-learned: who's alive
    request_queue:    PriorityQueue<Request>
    kv_cache:         Map<RequestId, KVCacheState> // ephemeral, per-request
    load_metrics: {
        queue_depth:      int
        avg_latency_ms:   float     // exponential moving average
        throughput_tps:    float     // tokens per second
        expert_hit_counts: Map<ExpertId, int>  // local activation frequency
    }
}
```

### 4.2 Shard Lifecycle on a Node

1. **Acquisition**: Node downloads shard weights (from shared storage, peer, or model hub)
2. **Loading**: Weights loaded to compute device (GPU VRAM, CPU RAM, etc.)
3. **Advertisement**: Node gossips that it now holds this shard
4. **Serving**: Node accepts and processes requests for this shard
5. **Eviction**: If memory pressure or low demand, node can unload and gossip removal

---

## 5. Gossip Protocol

### 5.1 Dual-Plane Design

**Cold Plane** (epidemic/SWIM-style, every 1–5 seconds):

- Node membership: join, leave, suspected-dead
- Shard map updates: who holds which shards
- Capability advertisements: hardware specs, available memory
- Protocol version and blueprint hash (ensure everyone agrees on the model structure)

**Hot Plane** (piggybacked on request traffic + short-interval heartbeats):

- Queue depth and recent latency per shard
- Expert activation frequency histograms (which experts are hot right now)
- Batch formation signals: "I have N requests buffered for shard X, forwarding in T ms"

### 5.2 Gossip Message Structure

```
GossipMessage {
    sender:         NodeId
    timestamp:      Timestamp
    sequence:       uint64          // monotonic per sender, for dedup
    plane:          COLD | HOT
    entries: [
        {
            type:   MEMBERSHIP | SHARD_UPDATE | LOAD_REPORT | EXPERT_HEAT
            key:    string
            value:  bytes
            ttl:    Duration        // how long to cache before expiring
        }
    ]
}
```

### 5.3 Convergence Properties

- Cold plane: full convergence within O(log N) gossip rounds for N nodes (standard epidemic guarantee)
- Hot plane: approximate convergence is sufficient — stale load data by a few hundred milliseconds is acceptable for routing decisions
- Expert heat maps: aggregated over sliding windows (e.g., last 60 seconds), used for replication/migration decisions on longer timescales

### 5.4 Expert Heat Tracking

Each node maintains a local histogram of expert activations it has observed or serviced. These are gossiped periodically and aggregated network-wide to identify:

- **Hot experts**: Consistently activated at high frequency → candidates for replication to more nodes
- **Cold experts**: Rarely activated → can be held by fewer nodes, or evicted from memory-constrained nodes
- **Bursty experts**: Activation spikes correlated with certain prompt types → inform prefetching strategies

---

## 6. Request Processing

### 6.1 Request Lifecycle

```
1. INGRESS      Client submits prompt to any node (entry point)
2. TOKENIZE     Entry node tokenizes, creates Request with unique ID
3. EMBED        Embedding + PLE lookup (entry node or dedicated embed node)
4. LAYER LOOP   For each transformer layer:
   a. ATTENTION   Forward to node holding this layer's attention shard
   b. ROUTE       Run gating network → get top-8 expert indices
   c. DISPATCH    Forward activation to nodes holding selected experts
                  (parallel fan-out to up to 8 nodes + shared expert node)
   d. AGGREGATE   Collect expert outputs, compute weighted sum
   e. RESIDUAL    Add residual connection, apply norm
5. LM HEAD      Final projection → logits
6. SAMPLE       Token selection (top-p, temperature, etc.)
7. APPEND       Add new token to KV cache, repeat from step 4 for next token
8. COMPLETE     Return generated sequence to client
```

### 6.2 Request Object

```
Request {
    request_id:     UUID
    sequence_id:    UUID            // groups multi-token generation
    token_position: int             // which token in the sequence
    current_stage:  ShardId         // where in the DAG we are
    activation:     Tensor          // the current hidden state
    provenance: [
        {
            shard_id:   ShardId
            node_id:    NodeId
            timestamp:  Timestamp
            hash:       bytes       // hash of (input + output + shard weights hash)
        }
    ]
    routing_hints:  Map<ShardId, NodeId>  // preferred next-hops from gossip
    priority:       int
    deadline:       Timestamp?      // optional latency SLO
    kv_cache_refs:  Map<LayerIdx, NodeId> // where KV cache segments live
}
```

### 6.3 Sequence Validation

Before processing, each node verifies:

1. The request's `current_stage` matches a shard this node holds
2. The provenance chain shows valid prior stages per the DAG
3. The activation tensor dimensions match expected input specs
4. (Optional) Spot-check: hash of prior stage output matches provenance record

Invalid requests are dropped and the failure is gossiped as a warning.

### 6.4 Expert Dispatch (Fan-Out / Fan-In)

The MoE dispatch step is the most network-intensive part. For each token at each MoE layer:

1. Router produces top-8 expert indices + gating weights
2. Node looks up gossip-informed shard map for those 8 experts
3. Activation tensor is sent to each expert's node (fan-out)
4. Expert nodes compute their FFN and return results (fan-in)
5. Aggregator computes gating-weighted sum of 8 expert outputs + shared expert output

**Optimization: co-location batching.** If multiple selected experts happen to be on the same node, send one message with the list of expert indices rather than 8 separate messages.

**Optimization: pipelining.** Don't wait for all 8 experts to return before starting the next layer's attention. If experts have predictable latency, overlap expert computation with the next layer's attention prep.

---

## 7. Load-Aware Routing

### 7.1 Path Selection

When multiple nodes hold the same shard, the DAG router picks based on:

1. **Queue depth** (from hot gossip): lower is better
2. **Network proximity**: measured RTT, propagated via cold gossip
3. **Batch compatibility**: prefer a node that's already batching requests at a similar sequence position
4. **Historical latency**: exponential moving average of completion times

### 7.2 Power-of-Two-Choices

For expert dispatch, rather than globally optimizing across all candidate nodes, use the "power of two choices" algorithm:

1. From the set of nodes holding the needed expert, randomly sample two
2. Query their most recent gossiped load
3. Send to the less loaded one

This achieves near-optimal load balancing with O(1) decision overhead and avoids the thundering-herd problem of global optimization.

### 7.3 Anti-Oscillation

To prevent load-balancing oscillation:

- Load reports use exponential moving averages (α ≈ 0.3) rather than instantaneous values
- Nodes add uniform random jitter (±10%) to reported queue depths
- Routing decisions are "sticky" for the duration of a sequence — once a request starts using a node for a given shard, it prefers to continue using that node (preserves KV cache locality)

---

## 8. Fault Tolerance

### 8.1 Failure Detection

- Cold gossip plane detects unresponsive nodes within 2–3 gossip rounds (3–15 seconds)
- Hot plane detects failures faster via request timeouts (configurable, default 500ms for LAN, 2s for WAN)
- SWIM-style protocol: suspicion → indirect probe via third-party node → confirmed dead

### 8.2 Recovery Strategies

**Request-level**: If a node fails mid-computation:

- Requesting node retries with an alternative node holding the same shard (from shard map)
- If no alternative exists, request fails with an error to the client
- KV cache for that request on the failed node is lost; the sequence must either restart from the last checkpoint or recompute affected layers

**Shard-level**: If a node holding a unique shard fails:

- Gossip propagates the loss; other nodes can acquire the shard from storage/peers
- During recovery window, requests needing that shard queue or fail gracefully

**Network partitions**: Nodes on each side of a partition continue operating independently with their available shards. Requests that need shards on the other side of the partition will fail. Partition healing is detected via gossip convergence.

### 8.3 Redundancy Targets

- **Shared expert**: Replicated to every node (always needed)
- **Hot routed experts**: Replicated to ≥2 nodes (gossip-driven)
- **Cold routed experts**: Minimum 1 node, with lazy replication if that node looks unhealthy
- **Attention shards**: ≥2 copies for pipeline-critical stages
- **Embedding + LM head**: Replicated to all entry/exit nodes

---

## 9. KV Cache Management

### 9.1 The Core Challenge

KV cache is per-request, per-layer, and lives on whatever node processed that layer's attention. For autoregressive generation, the cache must persist across token steps. This creates affinity — a request wants to return to the same node for the same layer on every token.

### 9.2 Strategies

**Sticky routing** (default): Once a request is assigned to a node for a given attention layer, all subsequent tokens for that sequence go to the same node. The gossip routing layer records this as a soft preference.

**Lazy replication**: During idle cycles, nodes gossip KV cache metadata (request ID, layer, size). Backup nodes can request a copy for fault tolerance. Not real-time — this is a best-effort backup.

**Checkpoint-and-resume**: For long sequences, periodically snapshot KV cache state to shared storage. If the original node fails, another node loads the snapshot and continues. Adds latency but prevents full recomputation.

### 9.3 Memory Budget

Per the Gemma 4 architecture analysis, the KV cache at full 256K context consumes approximately 5.2 GiB per sequence in bf16. For shorter contexts (e.g., 4K tokens), this drops to roughly 80 MB — very manageable. The sliding-window layers (5:1 ratio) only cache 1024 tokens regardless of sequence length, which significantly reduces memory pressure.

---

## 10. Dynamic Expert Migration

### 10.1 Trigger Conditions

The system periodically evaluates whether to migrate or replicate experts based on gossip-aggregated signals:

- **Replicate** when: an expert's activation frequency exceeds a threshold AND the node(s) holding it report high queue depth
- **Evict** when: an expert hasn't been activated in N minutes AND the node is under memory pressure
- **Migrate** when: an expert is consistently activated by requests originating from a topologically distant part of the network

### 10.2 Migration Protocol

1. Target node announces intent to acquire shard via gossip
2. Source node (or shared storage) streams weights to target
3. Target node loads weights to compute device
4. Target node gossips shard availability
5. Router begins including target node in routing decisions
6. (Optional) Source node evicts shard if no longer needed

### 10.3 Convergence Behavior

Over time, the system should converge toward a state where:

- Hot experts are replicated near the nodes generating requests that activate them
- Cold experts are consolidated on fewer, cheaper nodes
- The network's expert placement mirrors the actual activation distribution of the workload
- This is analogous to how a CDN converges on caching popular content at edge nodes

---

## 11. Final Aggregation & Verification

### 11.1 Provenance Verification

When a request completes the full DAG, the exit node (LM head) performs:

1. **Chain completeness**: Every required stage in the DAG is represented in the provenance
2. **Ordering**: Stages appear in a valid topological order
3. **Hash verification** (optional, configurable): Spot-check selected stage hashes against known-good values from trusted nodes
4. **Freshness**: No stage timestamp is unreasonably old (suggesting a stuck/replayed request)

### 11.2 Byzantine Tolerance (Optional, Future)

For deployments on untrusted hardware:

- Multiple nodes compute the same stage independently
- Results are compared; majority vote wins
- Disagreeing nodes are flagged and their trust score decremented via gossip
- Not needed for lab deployments; important for wide-area heterogeneous networks

---

## 12. Deployment Scenarios

### 12.1 Lab Cluster (2–4 GPUs, 10GbE / InfiniBand)

- **Sharding**: Coarse or medium granularity (pipeline by layer groups)
- **Transport**: RDMA or TCP with zero-copy
- **Gossip**: Fast convergence, low overhead (small N)
- **Primary value**: Run Gemma 4 26B at full bf16 precision across multiple GPUs that individually lack the VRAM; dynamic resharding without cluster restart
- **Expert dispatch latency**: Sub-millisecond over InfiniBand
- **Batch size**: High; nodes accumulate and batch-forward

### 12.2 Heterogeneous / Wide-Area (Consumer GPUs, Mixed Hardware)

- **Sharding**: Fine granularity (individual experts distributed)
- **Transport**: QUIC or TCP over internet/campus network
- **Gossip**: Slower convergence acceptable; load-aware routing critical
- **Primary value**: Pool disparate hardware to serve a model no single machine can hold; CDN-like expert placement
- **Expert dispatch latency**: 1–50ms depending on network topology
- **Quantization**: Mixed — some nodes run 4-bit, others 8-bit; the system tracks precision per shard

---

## 13. Prototype Roadmap

### Phase 1: Static Pipeline (Weeks 1–3)

- Manually partition Gemma 4 26B A4B across 2–3 nodes by layer groups
- Implement basic request forwarding along a fixed linear DAG
- Validate correctness: output matches single-node inference
- Transport: simple TCP with protobuf serialization
- No gossip yet — hardcoded shard map

### Phase 2: Gossip Discovery (Weeks 4–6)

- Implement cold-plane gossip (SWIM-style membership + shard map)
- Nodes discover each other and build the shard map dynamically
- Add/remove nodes without restarting the cluster
- Request routing uses gossip-learned shard map instead of hardcoded config

### Phase 3: Expert-Level Sharding (Weeks 7–10)

- Break MoE layers into individually addressable expert shards
- Implement fan-out/fan-in for expert dispatch
- Router node runs gating network, dispatches to expert nodes
- Shared expert replicated to all nodes

### Phase 4: Load-Aware Routing (Weeks 11–13)

- Implement hot-plane gossip (load metrics, expert heat)
- Power-of-two-choices routing for expert dispatch
- Batching: nodes accumulate requests and batch-forward
- Measure throughput vs. single-node baseline

### Phase 5: Dynamic Expert Migration (Weeks 14–16)

- Implement expert replication/eviction based on heat maps
- Migration protocol: stream weights, gossip availability
- Demonstrate convergence: expert placement adapts to workload changes

### Phase 6: Fault Tolerance & Verification (Weeks 17–20)

- Failure detection via gossip
- Request retry on alternative nodes
- KV cache checkpointing for long sequences
- Provenance chain verification

---

## 14. Technology Stack (Proposed)

| Component | Candidate | Rationale |
|---|---|---|
| Language | Rust | Performance-critical networking and memory management; good async ecosystem (tokio) |
| ML Runtime | candle (Rust) or PyTorch via FFI | candle for pure-Rust inference; PyTorch for compatibility with existing model loading |
| Serialization | FlatBuffers or Cap'n Proto | Zero-copy deserialization for activation tensors |
| Transport | QUIC (quinn crate) / TCP | QUIC for WAN (multiplexing, 0-RTT); TCP for LAN |
| Gossip | Custom (SWIM-based) | Tailored to dual-plane design; existing Rust SWIM libraries as starting point |
| Model Weights | Safetensors | Standard format, memory-mapped loading, compatible with HuggingFace ecosystem |

---

## 15. Key Metrics to Track

- **Tokens per second** (end-to-end, per-node)
- **Time-to-first-token** (latency from prompt to first generated token)
- **Expert dispatch latency** (fan-out to fan-in time per MoE layer)
- **Gossip convergence time** (time for a shard update to reach all nodes)
- **Expert utilization distribution** (are experts evenly loaded?)
- **Overhead ratio** (total network bytes transferred vs. useful computation bytes)
- **Fault recovery time** (time from node failure to request rerouting)

---

## 16. Open Questions

1. **Attention shard distribution**: Should attention blocks be pipelined (each layer on a different node) or replicated (all layers on every node, only experts distributed)? The latter simplifies KV cache management dramatically.

2. **Batching across sequences**: Can the system batch requests from different users through the same expert simultaneously? This is standard in centralized inference but harder in a decentralized setting where requests arrive at different nodes.

3. **Quantization heterogeneity**: If Node A runs expert 47 at 4-bit and Node B runs it at 8-bit, are the outputs compatible enough to mix in the same forward pass? Probably not — the system may need to enforce uniform quantization per shard.

4. **Speculative decoding integration**: Could a small draft model run locally on each node while the full MoE model runs distributed? The verification step would need to traverse the full DAG, but the speculation could happen locally.

5. **Multi-model support**: Can the same gossip network serve multiple models simultaneously? Nodes could hold shards from different models, with the DAG router selecting the right computation graph per request.

6. **Economic incentives**: For wide-area deployments on contributed hardware, how do you incentivize node operators to participate? Token-based compensation? Reciprocal compute credits?
