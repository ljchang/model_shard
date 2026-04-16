"""Drive a distributed inference against a running 3-node pipeline.

Usage:
    uv run python scripts/run_orchestrator.py \\
        --config config/shards.yaml \\
        --prompt-set tests/prompts.json \\
        --out-dir artifacts/dist \\
        --max-new-tokens 64

Loads only the tokenizer (via mlx-vlm), does NOT load model weights — all
compute happens on the Node processes.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from mlx_vlm import load as mlx_vlm_load

from model_shard.orchestrator import Orchestrator
from model_shard.shard_map import ShardMap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompt-set", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default="mlx-community/gemma-4-26b-a4b-it-4bit")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=2816)
    args = parser.parse_args()

    shard_map = ShardMap.from_yaml(args.config)
    total_layers = max(shard_map.lookup(sid).end_layer for sid in shard_map.all_shards())

    # We only need the tokenizer here — but mlx_vlm.load couples it to the model.
    # It's fast because weights are mmap'd and we don't exercise them.
    _model, processor = mlx_vlm_load(args.model)
    tokenizer = processor.tokenizer

    prompts = list(json.loads(args.prompt_set.read_text())["prompts"])
    args.out_dir.mkdir(parents=True, exist_ok=True)

    orch = Orchestrator(
        shard_map=shard_map, total_layers=total_layers, hidden_size=args.hidden_size
    )

    records: list[dict[str, object]] = []
    for i, text in enumerate(prompts):
        print(f"[{i + 1}/{len(prompts)}] {text!r}", flush=True)
        prompt_tokens = list(tokenizer.encode(text, add_special_tokens=False))
        generated = orch.generate_greedy(prompt_tokens, args.max_new_tokens)
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
        "model": args.model,
        "config": str(args.config),
        "total_layers": total_layers,
        "max_new_tokens": args.max_new_tokens,
        "captured_at": datetime.now(UTC).isoformat(),
        "prompts": records,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(prompts)} prompts to {args.out_dir}/manifest.json", flush=True)


if __name__ == "__main__":
    main()
