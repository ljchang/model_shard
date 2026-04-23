"""Capture reference oracle outputs for a prompt set.

Usage:
    uv run python scripts/run_reference.py \\
        --model <HF-id-or-local-path> \\
        --prompt-set tests/prompts.json \\
        --out-dir artifacts/ref \\
        --max-new-tokens 64

Writes:
    <out-dir>/manifest.json              — prompts, token ids, generated tokens
    <out-dir>/hidden_states_<i>.npz      — per-prompt layer-boundary snapshots

Acceptance tests later compare distributed-pipeline outputs against these
artifacts.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx
import numpy as np

from model_shard.reference import ReferenceModel


def _to_numpy_fp32(arr: mx.array) -> np.ndarray:
    """Cast MLX array to float32 before numpy handoff.

    MLX's bf16 doesn't round-trip cleanly through numpy's buffer protocol —
    it reports itemsize=2 but numpy sees it as uint8. Casting to fp32 keeps
    the comparison tolerance budget we have to spare at 1e-3.
    """
    return np.array(arr.astype(mx.float32))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-set", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model id or local path (no default — be explicit).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--skip-hidden-states",
        action="store_true",
        help="Capture only generated tokens (faster; skips Tier 2 oracle data)",
    )
    args = parser.parse_args()

    prompt_set = json.loads(args.prompt_set.read_text())
    prompts: list[str] = list(prompt_set["prompts"])

    print(f"Loading {args.model} ...", flush=True)
    ref = ReferenceModel.load(args.model)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "model": args.model,
        "num_layers": ref.num_layers,
        "max_new_tokens": args.max_new_tokens,
        "captured_at": datetime.now(UTC).isoformat(),
        "captured_hidden_states": not args.skip_hidden_states,
        "prompts": [],
    }
    prompt_records: list[dict[str, object]] = []

    for i, text in enumerate(prompts):
        print(f"[{i + 1}/{len(prompts)}] {text!r}", flush=True)
        prompt_tokens = ref.tokenize(text)
        generated = ref.generate_greedy(prompt_tokens, args.max_new_tokens)

        record: dict[str, object] = {
            "id": i,
            "text": text,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated,
            "generated_text": ref.detokenize(generated),
        }

        if not args.skip_hidden_states:
            trace = ref.prefill_trace(prompt_tokens)
            arrays: dict[str, np.ndarray] = {
                f"layer_{j}": _to_numpy_fp32(h) for j, h in enumerate(trace.layer_inputs)
            }
            arrays["final_hidden"] = _to_numpy_fp32(trace.final_hidden)
            arrays["logits"] = _to_numpy_fp32(trace.logits)
            hs_filename = f"hidden_states_{i}.npz"
            np.savez_compressed(args.out_dir / hs_filename, **arrays)  # type: ignore[arg-type]
            record["hidden_states_file"] = hs_filename

        prompt_records.append(record)

    manifest["prompts"] = prompt_records
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(prompts)} prompts to {args.out_dir}/manifest.json", flush=True)


if __name__ == "__main__":
    main()
