#!/usr/bin/env python
"""Phase 7-C-3a: convert HuggingFace Gemma 4 26B A4B (bf16) to MLX bf16.

Wraps ``mlx_lm.convert`` with explicit CLI arguments — no defaults baked
in for either source or destination, so the user picks both.

Usage (one-time, ~15-30 min on M5):

    uv run python scripts/convert_mlx_bf16.py \\
        --hf-source google/gemma-4-26B-A4B-it \\
        --output-dir ~/.cache/mlx-models/gemma-4-26b-a4b-it-bf16

Prerequisites:
  * ``huggingface-cli login`` — google/gemma-4-* is gated.
  * ~54 GB free disk at the output path.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-source",
        required=True,
        help="HuggingFace model id (e.g. google/gemma-4-26B-A4B-it).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Local directory to write the MLX bf16 conversion to.",
    )
    args = parser.parse_args()

    from mlx_lm.convert import convert

    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Converting {args.hf_source!r} -> {args.output_dir} (dtype=bfloat16)")
    convert(
        hf_path=args.hf_source,
        mlx_path=str(args.output_dir),
        dtype="bfloat16",
        quantize=False,
    )
    print(f"Done. Wrote MLX bf16 to {args.output_dir}")


if __name__ == "__main__":
    main()
