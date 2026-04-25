"""Drive a distributed inference against a running 3-node pipeline.

Usage:
    uv run python scripts/run_client.py \\
        --config config/shards.yaml \\
        --prompt-set tests/prompts.json \\
        --out-dir artifacts/dist \\
        --max-new-tokens 64

This is a *client* — it knows how to reach the head node and stream tokens
back. It has no pipeline logic. Nodes coordinate with each other directly.

The --warmup flag pre-runs each prompt once with max_new_tokens=1, discarding
outputs, before the timed pass. On CUDA backends with per-shape kernel JIT
(e.g. HF transformers' _grouped_mm on Grace Blackwell + CUDA 13) this pays
the prefill JIT cost up front so the timed pass measures steady-state
behaviour rather than first-touch compile latency.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from mlx_vlm import load as mlx_vlm_load

from model_shard.client import Client
from model_shard.shard_map import ShardMap


def _find_head(shard_map: ShardMap) -> str:
    for sid in shard_map.all_shards():
        if shard_map.lookup(sid).start_layer == 0:
            return sid
    raise ValueError("no head shard (start_layer=0) in config")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompt-set", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--warmup",
        action="store_true",
        help=(
            "Run each prompt once with max_new_tokens=1 before the timed pass, "
            "discarding outputs. Use on CUDA hosts that JIT per prompt shape."
        ),
    )
    args = parser.parse_args()

    shard_map = ShardMap.from_yaml(args.config)
    model_id = args.model or shard_map.model_id
    if not model_id:
        parser.error(
            "no model id available: pass --model or set model_id in shards.yaml"
        )
    head_spec = shard_map.lookup(_find_head(shard_map))

    # Tokenizer only — no weights exercised here, all compute is on the Nodes.
    _model, processor = mlx_vlm_load(model_id)
    tokenizer = processor.tokenizer

    prompts = list(json.loads(args.prompt_set.read_text())["prompts"])
    args.out_dir.mkdir(parents=True, exist_ok=True)

    client = Client(head_address=head_spec.address)

    encoded_prompts = [
        list(tokenizer.encode(text, add_special_tokens=False)) for text in prompts
    ]

    if args.warmup:
        print(f"Warmup pass: {len(prompts)} prompts x 1 token each", flush=True)
        for i, prompt_tokens in enumerate(encoded_prompts):
            t0 = time.time()
            client.generate(prompt_tokens, 1)
            print(
                f"[warmup {i + 1}/{len(prompts)}] len={len(prompt_tokens)} "
                f"took {time.time() - t0:.1f}s",
                flush=True,
            )

    records: list[dict[str, object]] = []
    for i, (text, prompt_tokens) in enumerate(
        zip(prompts, encoded_prompts, strict=True)
    ):
        print(f"[{i + 1}/{len(prompts)}] {text!r}", flush=True)
        generated = client.generate(prompt_tokens, args.max_new_tokens)
        records.append(
            {
                "id": i,
                "text": text,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated,
                "generated_text": tokenizer.decode(generated, skip_special_tokens=True),
            }
        )

    manifest = {
        "model": model_id,
        "config": str(args.config),
        "max_new_tokens": args.max_new_tokens,
        "captured_at": datetime.now(UTC).isoformat(),
        "prompts": records,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(prompts)} prompts to {args.out_dir}/manifest.json", flush=True)


if __name__ == "__main__":
    main()
