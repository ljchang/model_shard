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
