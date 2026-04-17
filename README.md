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
