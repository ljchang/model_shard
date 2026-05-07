# model_shard

**Decentralized, gossip-coordinated inference for Mixture-of-Experts models across heterogeneous hardware.** A "CDN for experts" — nodes self-organize via gossip, route activations through a peer-to-peer pipeline, and migrate expert weights between machines based on observed load. No central scheduler.

The reference target is **Gemma 4 26B-A4B-it** (30 layers, 128 routed experts, top-8 routing) running across a Mac (MLX) and one or more CUDA boxes (PyTorch).

> **Status:** Research prototype. Single-machine 3-process pipeline is bit-exact to a single-process reference on all canonical prompts. Heterogeneous (Mac + DGX Spark + RTX 3090) cluster boots, gossips, and serves end-to-end inference; full multi-prompt smoke is blocked on an upstream PyTorch + Grace Blackwell + CUDA 13 kernel pathology, not on anything in this repo. See [Roadmap](#roadmap).

---

## What it does

Conceptually, the project answers one question: **can a Mixture-of-Experts model be served by a swarm of heterogeneous machines that coordinate themselves, with no central scheduler, and remain correct under failure?**

The components that make that work today:

- **Pipeline sharding.** The model's transformer layers are partitioned across nodes. Activations flow peer-to-peer over length-prefixed TCP; the head dials the mid, mid dials the tail, the tail samples and returns the token to the head, which streams to the client.
- **Expert sharding.** Within an MoE layer, the 128 routed experts are distributed across nodes. The node hosting a layer's attention block runs the router and fans out post-attention activations to peer nodes via `ExpertRequest` RPCs, then aggregates the top-k outputs.
- **Partial expert loading.** A node only loads the experts it owns, dropping resident memory from ~14 GB per shard to ~4.5 GB chassis + `k/128 × 9 GB` of routed experts. This is what makes 24 GB-VRAM deployments (3090s) viable.
- **Gossip membership (SWIM).** Each node runs a SWIM-style membership protocol over UDP. The head admits new requests only when every required shard is `ALIVE`; in-flight requests fail cleanly if a peer transitions out of `ALIVE` mid-decode.
- **Dynamic expert migration.** Nodes track per-expert activation heat as an EMA, gossip top-N heat reports piggybacked on SWIM messages, and pull hot expert weights from peers over TCP, slotting them bit-exactly into a compact stacked tensor at runtime. Eviction (the inverse) runs under capacity pressure with last-writer-wins ownership convergence.
- **Load-aware routing.** When an expert is replicated across multiple nodes, each top-k dispatch goes to the less-loaded candidate via power-of-two-choices using gossiped queue-depth EMAs.
- **Fault tolerance.** Local retry to alternate replicas on peer failure; receive-time validation of a hash-chained provenance DAG that mirrors the model's true computation graph (rejects topology / authorization errors).
- **Pluggable backends.** A `Backend` protocol abstracts every tensor-level operation. `MLXBackend` (Apple Silicon, bf16 or 4-bit) and `PyTorchBackend` (CUDA, bf16) implement the same 20-method surface, including `slice_expert` / `attach_expert` / `detach_expert` so partial-load + migration + eviction work on either side. The same wire protocol drives both.

---

## Quickstart

### Install

```bash
# Apple Silicon (MLX backend, default)
uv sync --extra dev

# Linux / CUDA (PyTorch backend)
uv sync --extra dev --extra pytorch

# Regenerate protobuf bindings (only needed if proto/wire.proto changed)
uv run python -m grpc_tools.protoc -I proto --python_out=src/model_shard/_pb proto/wire.proto
```

### Single-machine 3-process demo

This runs the full distributed pipeline on a single Mac with three node processes communicating over localhost TCP. Useful for development and as a correctness reference.

The default config (`config/shards.yaml`) expects a local bf16 conversion of `google/gemma-4-26B-A4B-it`. Convert it once (~48 GB to disk):

```bash
uv run python scripts/convert_mlx_bf16.py \
  --hf-source google/gemma-4-26B-A4B-it \
  --output-dir ~/.cache/mlx-models/gemma-4-26b-a4b-it-bf16
```

Then in three terminals:

```bash
uv run python scripts/run_node.py --config config/shards.yaml --shard layer_0-10
uv run python scripts/run_node.py --config config/shards.yaml --shard layer_10-20
uv run python scripts/run_node.py --config config/shards.yaml --shard layer_20-30
```

In a fourth terminal, drive the cluster:

```bash
uv run python scripts/run_client.py \
  --config config/shards.yaml \
  --prompt-set tests/prompts.json \
  --out-dir artifacts/run \
  --max-new-tokens 16
```

For comparison, capture single-process oracle output:

```bash
uv run python scripts/run_reference.py \
  --prompt-set tests/prompts.json \
  --out-dir artifacts/ref \
  --max-new-tokens 16
```

### Heterogeneous cluster (Mac + CUDA)

See `config/shards.heterogeneous.example.yaml` for a 3-machine template (Mac MLX head + DGX Spark PyTorch mid + RTX 3090 PyTorch tail) over Tailscale. All nodes must gossip the same `model_id` for cluster admission. On the CUDA boxes, set `MODEL_SHARD_BACKEND=pytorch` (or rely on auto-detect — MLX on Apple Silicon, PyTorch elsewhere).

---

## Configuration

### Backend selection

| Variable | Values | Default |
|---|---|---|
| `MODEL_SHARD_BACKEND` | `mlx`, `pytorch` | auto-detect (MLX on Apple Silicon, PyTorch elsewhere) |
| `MLX_MODEL_BF16_LOCAL_PATH` | filesystem path | `~/.cache/mlx-models/gemma-4-26b-a4b-it-bf16` |

### Feature flags

Each major capability ships behind an env flag so a regression can be bisected against a single variable. Defaults are conservative — `ENABLE_GOSSIP`, `ENABLE_EXPERT_RETRY`, and `ENABLE_EVICTION` are on; everything else is opt-in.

| Variable | Default | Effect |
|---|---|---|
| `ENABLE_GOSSIP` | `true` | SWIM membership over UDP (`tcp_port + 1000`); head admits requests only when all shards are `ALIVE`. |
| `ENABLE_EXPERT_SHARD` | `false` | Distribute MoE experts across nodes via `ExpertRequest` RPCs. |
| `ENABLE_PARTIAL_LOAD` | `false` | Load only the experts listed in this shard's `moe_experts` (precondition for migration). |
| `ENABLE_DYNAMIC_MIGRATION` | `false` | Heat-driven pull-migration of expert weights between nodes. Requires `ENABLE_PARTIAL_LOAD=true`. |
| `ENABLE_EXPERT_RETRY` | `true` | On peer failure, retry to an alternate replica before surfacing the error. |
| `ENABLE_EVICTION` | `true` | Evict cold migration-added experts under capacity pressure with LWW gossip. |
| `ENABLE_PROVENANCE` | `false` | Receive-time validation of a hash-chained provenance DAG; rejects topology / authorization errors with `ERR_INVALID_PROVENANCE`. |

Tuning knobs (selected): `MIGRATION_HEAT_THRESHOLD`, `MIGRATION_SCAN_INTERVAL_SECONDS`, `MIGRATION_EVICT_COOLDOWN_SECONDS`, `EXPERT_RETRY_MAX_ATTEMPTS`, `EXPERT_RETRY_BACKOFF_MS`. See `src/model_shard/node.py` for the full list and defaults.

### Debug endpoints

Each node exposes two HTTP endpoints alongside its TCP port:

- `http://<host>:<tcp_port + 2000>/membership` — SWIM membership view.
- `http://<host>:<tcp_port + 2000>/loads` — gossiped queue-depth EMAs.

---

## Architecture

```
                            ┌───────────────────────────────────────┐
                            │           Gossip plane (UDP)          │
                            │   SWIM membership + heat reports +    │
                            │   ownership ADD/REMOVE deltas (LWW)   │
                            └───────────────────────────────────────┘
                                   │           │           │
Client → head (layers 0-9) ─activations→ mid (layers 10-19) ─activations→ tail (layers 20-29)
   ▲                              │                │                              │
   │                              ▼                ▼                              │
   │                      ExpertRequest        ExpertRequest                      │
   │                      (fan out top-8       (fan out top-8                     │
   │                       to expert hosts)     to expert hosts)                  │
   │                                                                              │
   └─────────────────────── SampledToken (tail → head → client) ──────────────────┘
```

Each node is one OS process. Inbound connections are handled one thread per peer; outbound peer connections are persistent. The head's client-handler thread drives the decode loop by `queue.get()`-ing `SampledToken`s from the tail.

### Wire protocol

`proto/wire.proto` is the source of truth. The `Envelope` is a oneof of:

| Message | Direction | Purpose |
|---|---|---|
| `BeginRequest` | client → head | Start a generation request (prompt + max tokens). |
| `Activation` | node → downstream | Forward hidden state along the pipeline. |
| `ExpertRequest` / `ExpertResponse` | router → expert host | Fan-out / aggregate within an MoE layer. |
| `SampledToken` | tail → head, head → client | Newly sampled token (with `is_final`). |
| `EndRequest` | tail → upstream | Cleanup KV-cache slots along the chain. |
| `Error` | any → upstream | Structured error (admission, peer-down, provenance). |

Tensors are serialized out-of-band: `[msg_len:4][msg][tensor_len:4][tensor]`.

### Source layout

```
src/model_shard/
├── shard.py, shard_map.py         Static topology (YAML-backed)
├── request.py                     Request + ProvenanceEntry chain
├── transport.py, envelope.py      Length-prefixed TCP framing
├── node.py                        Decentralized node, decode loop, env-flag gates
├── client.py                      Thin client (BeginRequest + token stream)
├── expert_orchestrator.py         Phase-A/B/C MoE fan-out + retry + provenance
├── moe.py, partial_load.py        Pure MLX MoE helpers
├── pt_moe.py, pt_partial_load.py  PyTorch counterparts (HF-correct forward)
├── mlx_engine.py, pytorch_engine.py    Per-backend forward primitives
├── backends/                      Backend protocol + MLXBackend + PyTorchBackend
├── membership/                    SWIM gossip (UDP, observer pattern)
├── heat.py, migration.py          Per-expert EMA + pull-migration scanner
├── load.py                        Gossiped queue-depth + power-of-two-choices
├── provenance.py                  BLAKE2b-256 hash-chained DAG
├── reference.py                   Single-process oracle (correctness baseline)
└── _pb/                           Generated protobuf bindings
```

---

## Tests

```bash
uv run pytest                                      # fast suite (~380 tests, no model load)
uv run pytest -m slow                              # loads the bf16 model (~52 GB)
uv run pytest -m slow tests/test_tier1_tokens.py   # bit-exact token regression
uv run pytest -m slow tests/test_tier2_hidden.py   # per-layer hidden-state agreement
uv run ruff check src tests scripts
uv run mypy src/model_shard
```

Two correctness tiers anchor the project:

- **Tier 1 — token bit-exactness.** The 3-shard distributed pipeline produces token-identical output to the single-process reference on all 5 canonical prompts in `tests/prompts.json`.
- **Tier 2 — per-layer hidden-state tolerance.** Per-layer hidden states agree with the reference within `< 1e-3`.

Cross-backend agreement (MLX bf16 vs PyTorch bf16, against the same source weights) is asserted via `tests/test_cross_backend_correctness.py` using top-K softmax-weighted token sets per decode position. Floors are calibrated as regression guards, not tightness bars: at landing, 3/3 position-0 top-1 matches and 3.07 average top-5 overlap (vocab=262K → random top-5 overlap ≈ 0).

---

## Roadmap

The project is structured as a six-phase research program. The first six phases are complete; Phase 7 (heterogeneous deployment) is in progress.

| Phase | Status | What it adds |
|---|---|---|
| 1. Static pipeline | done | 3-shard pipeline; Tier 1/2 bit-exact baseline. |
| 2. Gossip discovery | done | SWIM membership; admission control; clean in-flight failure on peer death. |
| 3. Expert-level sharding | done | Fan out a single MoE layer's 128 experts across nodes via `ExpertRequest`. |
| 4. Load-aware routing | done | Power-of-two-choices over multi-candidate experts using gossiped load EMAs. |
| 5a. Partial expert loading | done | Load only owned experts; precondition for migration and 24 GB-VRAM deployments. |
| 5b. Dynamic expert migration | done | Heat-driven pull-migration of expert weights between peers, bit-exact on receive. |
| 6-A. Expert-peer retry | done | Local retry to alternate replicas on peer failure (no central coordinator). |
| 6-B. Provenance verification | done | Hash-chained DAG validated at every receive hop; rejects topology / authorization errors. |
| 6-C. Expert eviction | done | LWW REMOVE deltas with safety invariants (bootstrap-protected, last-replica refusal, attach cooldown). |
| 7-A. Backend protocol | done | `Backend` abstraction with `MLXBackend`; zero behavioral change on default deployments. |
| 7-B. PyTorch backend | done | Full 20-method protocol parity on CUDA / DGX Spark, including slice/attach/detach. |
| 7-C-1. Real HF Gemma 4 forward | done | `PyTorchBackend` validated against real HF `Gemma4ForCausalLM` (not stubs). |
| 7-C-2. Cross-backend correctness | done | Top-K agreement floors between MLX and PyTorch backends. |
| 7-C-3a. Bf16 canonical rebaseline | done | Single source weights for both backends; agreement jumped from 1/3 to 3/3 top-1. |
| 7-C-3b. Heterogeneous cluster smoke | **partial** | 3-machine cluster boots and serves; full smoke blocked on upstream `grouped_mm` pathology on Grace Blackwell + CUDA 13. |
| 7-C-4. Tech-debt cleanup | done | Backend owns outer decoder ops + batched aggregate; unused threading removed. |

Per-phase design rationale lives in `docs/superpowers/specs/<date>-phase<N>-*-design.md`; per-phase implementation plans (with task breakdowns) live in `docs/superpowers/plans/<date>-phase<N>-*.md`.

### What's next

Carry-forwards under consideration for a future phase:

- File a PyTorch upstream issue for the `grouped_mm` shape-resolution pathology on GB10 + CUDA 13 + bf16 (the blocker for 7-C-3b's clean smoke).
- Pipeline-peer redundancy (deferred from 6-A case 2 — requires redundant layer ranges in `shards.yaml`).
- Signed `ProvenanceEntry` + sample-rerun re-verification (Byzantine-insider detection beyond 6-B's topology bar).
- Cross-node ownership-exclusion gossip (so a peer that fails for one node is excluded for all routers, not just locally — 6-A R5).
- `slice_expert` format bridge for cross-backend migration (MLX returns the 9-tensor canonical layout; PyTorch returns HF's 2-tensor fused layout).

---

## Hardware notes

The development matrix is asymmetric:

- **M5 Mac, 128 GB unified.** Primary dev machine. Can hold the full bf16 model (~52 GB) plus 2–3 partitioned node processes simultaneously. Used for Tier 1/2 bit-exactness tests, single-machine 3-process demos, and the MLX side of cross-backend agreement.
- **NVIDIA DGX Spark (GB10, 128 GB unified LPDDR5X).** PyTorch + CUDA 13 reference. Holds full bf16 model. Used for Tier 1 PyTorch fixtures and as the mid shard in heterogeneous smoke.
- **RTX 3090 (24 GB VRAM).** PyTorch + CUDA 12.8 (forward-compat against driver 545). Used for partial-load deployments — the 24 GB constraint is the design point for `ENABLE_PARTIAL_LOAD` + migration.
- **Tailscale mesh** connects the three machines; SWIM gossip and activation TCP both go over IPv4.

---

## Further reading

- **Spec:** `/Users/lukechang/Downloads/gossip-moe-inference-spec.md` (v0.1, April 2026) — the original 20-week research-program spec.
- **Per-phase design specs:** `docs/superpowers/specs/`
- **Per-phase implementation plans:** `docs/superpowers/plans/`
- **Reference architecture facts** (Gemma 4 26B): see Phase 1 plan for the layer-shape derivation.
