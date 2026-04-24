# Heterogeneous 3-node deployment runbook (Phase 7-C-3b)

This runbook walks through deploying a 3-machine inference cluster with
mixed MLX and PyTorch backends:

- **Mac M5** — MLX bf16 head, layers 0-9, full load
- **DGX Spark** — PyTorch bf16 mid, layers 10-19, full load
- **Ubuntu 3090 (24 GB VRAM)** — PyTorch bf16 tail, layers 20-29, partial load

All three machines connect over Tailscale and serve the same source weights
(`google/gemma-4-26B-A4B-it`).

## Prerequisites

- All 3 machines on the same Tailscale tailnet. Verify with:
  ```bash
  tailscale status
  tailscale ping <each-other-host>
  ```

- HuggingFace authentication for `google/gemma-4-26B-A4B-it` on Spark and
  3090 (Mac uses the local conversion):
  ```bash
  huggingface-cli login
  huggingface-cli whoami  # confirm
  ```

- On Mac: MLX bf16 conversion already produced (Phase 7-C-3a Task 7).
  Verify the cache exists:
  ```bash
  ls ~/.cache/mlx-models/gemma-4-26b-a4b-it-bf16/
  ```

- On Spark and 3090: clone or rsync the model_shard repo and `uv sync
  --extra dev --extra pytorch`.

- On 3090: confirm at least 22 GB free VRAM (`nvidia-smi`).

## Configuration

1. Copy `config/shards.heterogeneous.example.yaml` on each machine.
2. Replace `<mac-tailscale-hostname>`, `<spark-tailscale-hostname>`, and
   `<3090-tailscale-hostname>` with the actual hostnames or IPs.
3. The `model_id` string MUST be identical on all three machines —
   admission control will reject mismatched peers.

Save as `~/model-shard-shards.yaml` (or anywhere; pass via `--config`).

## Pre-flight smoke checks

On each machine:
```bash
# Confirm config parses and model_id is the canonical HF id.
uv run python -c "
from pathlib import Path
from model_shard.shard_map import ShardMap
sm = ShardMap.from_yaml(Path('~/model-shard-shards.yaml').expanduser())
print('model_id:', sm.model_id)
print('shards:', sm.all_shards())
"
```

Expected on all 3 machines: `model_id: google/gemma-4-26B-A4B-it`.

## Start the cluster

Order of startup doesn't matter — SWIM gossip is order-independent. But
starting tail first lets you watch its memory usage as the others
connect.

### On the 3090 (tail):

```bash
cd ~/Github/model_shard
ENABLE_PARTIAL_LOAD=true \
MODEL_SHARD_BACKEND=pytorch \
uv run python scripts/run_node.py \
    --config ~/model-shard-shards.yaml \
    --shard tail
```

In another terminal on the 3090, watch VRAM:
```bash
watch -n 1 nvidia-smi
```

The tail should load partial weights and stabilize at <22 GB VRAM. If
it OOMs, reduce `moe_experts` per layer in the shards.yaml.

### On Spark (mid):

```bash
cd ~/Github/model_shard
MODEL_SHARD_BACKEND=pytorch \
uv run python scripts/run_node.py \
    --config ~/model-shard-shards.yaml \
    --shard mid
```

### On Mac (head):

```bash
cd ~/Github/model_shard
MODEL_SHARD_BACKEND=mlx \
uv run python scripts/run_node.py \
    --config ~/model-shard-shards.yaml \
    --shard head
```

## Smoke verification

In a 4th terminal on Mac, run a single-prompt client against the head:

```bash
cd ~/Github/model_shard
MODEL_SHARD_BACKEND=mlx \
uv run python scripts/run_client.py \
    --config ~/model-shard-shards.yaml \
    --prompt-set tests/prompts.json \
    --out-dir /tmp/heterogeneous-out \
    --max-new-tokens 16
```

Compare the generated tokens for prompt 0 against the bf16 oracle:

```bash
uv run python -c "
import json
ref = json.load(open('artifacts/ref/manifest.json'))
got = json.load(open('/tmp/heterogeneous-out/results.json'))
ref_ids = ref['prompts'][0]['generated_tokens'][:16]
got_ids = got['prompts'][0]['generated_tokens'][:16]
print('reference:', ref_ids)
print('cluster:  ', got_ids)
print('match:', ref_ids == got_ids)
"
```

Expected: `match: True`. If False, see "Common failure modes" below.

## Common failure modes

### `RuntimeError: rejecting peer ... with model_id mismatch`

A node has a different `model_id` in its `shards.yaml`. Verify all 3
configs have `model_id: "google/gemma-4-26B-A4B-it"` exactly.

### Tail OOMs on the 3090

Reduce `moe_experts` per layer in the tail's `shards.yaml`. The example
config holds ~42 experts per layer × 10 MoE layers; if that doesn't fit,
try ~28 experts per layer.

### Cluster never stabilizes (peers stuck SUSPECT)

Tailscale connectivity issue. Run `tailscale ping <peer>` from each
machine to confirm bidirectional reachability. If only one direction
works, check Tailscale firewall rules.

### Token sequence mismatches the oracle

If position-0 differs, the wire format isn't aligned across backends —
re-run `tests/test_cross_backend_wire_roundtrip.py` on Mac to verify.

If positions 1+ drift while position-0 matches, that's accumulating
floating-point divergence between backends, which is expected to a
small degree. Tier 1 tolerance is exact-match, so any drift fails the
test. The 7-C-2 cross-backend agreement bar already measures this; if
it's >3.07 top-5 overlap (the post-7-C-3a baseline), something
regressed.

### Head can't reach mid or tail

Verify the `host` fields in `shards.yaml` are reachable Tailscale
hostnames. Try `nslookup <host>` and `ping <host>` from the Mac.
