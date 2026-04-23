#!/usr/bin/env python
"""Phase 7-C-3a: convert HuggingFace Gemma 4 26B A4B (bf16) to MLX bf16.

Wraps ``mlx_vlm.convert`` with explicit CLI arguments — no defaults baked
in for either source or destination, so the user picks both. Uses
``mlx_vlm.convert`` (not ``mlx_lm.convert``) because Gemma 4 is a
multimodal model whose loader (``mlx_vlm.load``) requires the vision
tower weights even when only language inference is exercised.

Usage (one-time, ~15-30 min on M5):

    uv run python scripts/convert_mlx_bf16.py \\
        --hf-source <HF-id> \\
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
        help="HuggingFace model id (gated; requires huggingface-cli login).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Local directory to write the MLX bf16 conversion to.",
    )
    args = parser.parse_args()

    # Gemma 4 is a multimodal VLM — use mlx_vlm.convert (NOT mlx_lm.convert)
    # so the vision tower weights are preserved. Without them, mlx_vlm.load
    # (in model_shard/mlx_engine.py::load_model) raises ValueError on
    # missing vision_tower.* weights even though inference only walks the
    # language model. Reference: mlx-community/gemma-4-26b-a4b-it-4bit was
    # produced via mlx_vlm.convert for the same reason.
    from mlx_vlm import convert

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
