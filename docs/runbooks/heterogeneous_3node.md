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

Order of startup doesn't matter — SWIM gossip is order-independent. We
recommend tail first so you can confirm partial-load OOM doesn't happen
before committing to launching the others.

### On the 3090 (tail):

```bash
cd ~/Github/model_shard
ENABLE_PARTIAL_LOAD=true \
MODEL_SHARD_BACKEND=pytorch \
uv run python scripts/run_node.py \
    --config "$HOME/model-shard-shards.yaml" \
    --shard tail
```

In another terminal on the 3090, watch VRAM:
```bash
watch -n 1 nvidia-smi
```

The tail should load partial weights and stabilize at <22 GB VRAM. VRAM
peaks at startup (during model load) and stays flat afterward — peer
joins do not add to it. If you OOM, see "Tail OOMs on the 3090" below.

### On Spark (mid):

```bash
cd ~/Github/model_shard
MODEL_SHARD_BACKEND=pytorch \
uv run python scripts/run_node.py \
    --config "$HOME/model-shard-shards.yaml" \
    --shard mid
```

### On Mac (head):

```bash
cd ~/Github/model_shard
# If your MLX bf16 conversion is NOT at the conventional path
# (~/.cache/mlx-models/gemma-4-26b-a4b-it-bf16/), uncomment and set:
# export MLX_MODEL_BF16_LOCAL_PATH=/your/path

MODEL_SHARD_BACKEND=mlx \
uv run python scripts/run_node.py \
    --config "$HOME/model-shard-shards.yaml" \
    --shard head
```

## Smoke verification

In a 4th terminal on Mac, run the client against the head with all 5
canonical prompts in `tests/prompts.json`. The `--warmup` flag pre-runs
each prompt once with `max_new_tokens=1` to pay the per-shape kernel JIT
cost on the Spark GB10 (`_grouped_mm` on Blackwell + CUDA 13 compiles a
fresh kernel per prompt-shape; once cached, subsequent runs are fast).
Without `--warmup` the second prompt's 11-token prefill takes ~3 minutes
on first touch.

```bash
cd ~/Github/model_shard
MODEL_SHARD_BACKEND=mlx \
uv run python scripts/run_client.py \
    --config "$HOME/model-shard-shards.yaml" \
    --prompt-set tests/prompts.json \
    --out-dir /tmp/heterogeneous-out \
    --max-new-tokens 16 \
    --warmup
```

Expect five `[warmup i/5]` lines (each 5–60s on first run, depending on
prompt length and CUDA kernel cache state) followed by five `[i/5]` lines
that complete in ~1–2s each. The final line writes
`/tmp/heterogeneous-out/manifest.json`.

Sanity-check the manifest: all 5 prompts present, each with 16 generated
tokens, and the cluster output is "model-like" rather than empty or
garbage:

```bash
uv run python -c "
import json
m = json.load(open('/tmp/heterogeneous-out/manifest.json'))
ok = True
for p in m['prompts']:
    n = len(p['generated_tokens'])
    print(f\"prompt {p['id']}: {n} tokens, text={p['generated_text']!r}\")
    if n != 16:
        ok = False
print('ok:', ok)
"
```

Expected: `ok: True` and five lines each printing 16 tokens. The
generated text on greedy decode of the base (non-instruct) model can
loop/repeat — that is real model behaviour, not a cluster bug. Cross-
backend token equality vs the bf16 oracle is **not** a smoke gate (bf16
last-bit drift between MLX and PyTorch causes top-1 swaps when the top
two logits are nearly tied; Phase 7-C-2's calibrated top-K agreement
floors are the correctness bar).

### Done when

- All 3 nodes report ALIVE in `curl http://127.0.0.1:<head-tcp-port + 2000>/membership`
- Tail VRAM stays under 22 GB (`nvidia-smi` on 3090)
- Smoke client writes `manifest.json` with all 5 prompts × 16 tokens (`ok: True`)

## Common failure modes

### `WARNING: rejecting peer ... with model_id mismatch`

A node has a different `model_id` in its `shards.yaml`. Verify all 3
configs have `model_id: "google/gemma-4-26B-A4B-it"` exactly. To
observe the current cluster view from any node:
`curl http://127.0.0.1:<that-node-tcp-port + 2000>/membership`.

### Tail OOMs on the 3090

First, confirm partial-load actually took effect — the tail's startup
logs should mention loading only the assigned experts (not the full
model). If they don't, `ENABLE_PARTIAL_LOAD=true` didn't propagate
into the subprocess (check shell quoting / env-var export).

If partial-load is active and you still OOM, reduce `moe_experts` per
layer in the tail's `shards.yaml`. The example config holds ~42 experts
per layer × 10 MoE layers; try ~28 experts per layer.

### Cluster never stabilizes (peers stuck SUSPECT)

Query each node's debug endpoint to see SWIM membership state:
`curl http://127.0.0.1:<that-node-tcp-port + 2000>/membership`.
Expect every peer to be `ALIVE`. If anyone is `SUSPECT`, that's a
Tailscale connectivity issue — run `tailscale ping <peer>` from each
machine to confirm bidirectional reachability. If only one direction
works, check Tailscale firewall rules.

### Token sequence mismatches the oracle

Any divergence (whether at position 0 or later) fails the exact-match
Tier 1 contract. Diagnostic order:

1. Re-run `tests/test_cross_backend_wire_roundtrip.py` on Mac to confirm
   MLX↔PyTorch wire format alignment. If it fails, that's the root cause.
2. Re-run a single-backend Tier 1 (`uv run pytest -m slow tests/test_tier1_tokens.py`)
   on Mac to confirm the bf16 oracle is reproducible single-process.
3. If both pass but the heterogeneous cluster diverges, the bug is in
   the cross-backend pipeline — likely in how the activation tensor
   crosses the MLX→PyTorch boundary at the head→mid hop. Inspect
   subprocess logs on the mid node for any tensor reshape or dtype
   mismatch.

### Head can't reach mid or tail

Verify the `host` fields in `shards.yaml` are reachable Tailscale
hostnames. Try `nslookup <host>` and `ping <host>` from the Mac.
