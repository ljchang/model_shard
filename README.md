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
