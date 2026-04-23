#!/usr/bin/env python
"""Phase 7-C-2/7-C-3a: unified Tier-1 fixture generator for cross-backend comparison.

Dispatches on ``MODEL_SHARD_BACKEND=mlx|pytorch`` (defaults to pytorch) and
produces ``tests/fixtures/{mlx,pytorch}_tier1_tokens.json`` with top-K per
decode position. Consumed by:

  * ``tests/test_pytorch_tier1.py`` — internal regression (top-1 = ids[0]).
  * ``tests/test_cross_backend_correctness.py`` — cross-backend top-K
    overlap between the two fixtures.

Usage:
    # On Mac (Apple Silicon, MLX bf16):
    MODEL_SHARD_BACKEND=mlx uv run python scripts/generate_tier1_comparison_fixture.py \\
        --model /path/to/local/mlx-bf16-dir

    # On DGX Spark (CUDA, PyTorch bf16):
    MODEL_SHARD_BACKEND=pytorch uv run python scripts/generate_tier1_comparison_fixture.py \\
        --model <HF-id-or-local-path>
"""
from __future__ import annotations

import argparse
import json
import os
import threading
from pathlib import Path
from typing import Any

PROMPTS = [
    "The quick brown fox",
    "In a galaxy far far away",
    "Once upon a time",
]
N_POSITIONS = 10
TOP_K_RECORDED = 5


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Backend-appropriate model id (HF id for PyTorch; HF id or "
             "local MLX bf16 directory for MLX).",
    )
    return parser.parse_args()


def _load_backend(hf_id: str) -> tuple[Any, str, str, str, Any]:
    """Return (backend, hf_id, device, dtype_str, topk_helper).

    topk_helper is the engine-specific ``top_k_ids_and_weights`` function
    so the per-prompt loop below can call either uniformly."""
    name = os.environ.get("MODEL_SHARD_BACKEND", "").lower() or "pytorch"
    if name == "mlx":
        from model_shard import mlx_engine
        from model_shard.backends import MLXBackend
        backend = MLXBackend(mlx_lock=threading.Lock())
        device = "mps"
        dtype_str = "mlx-bf16"
        topk = mlx_engine.top_k_ids_and_weights
    elif name == "pytorch":
        from model_shard import pytorch_engine
        from model_shard.backends import PyTorchBackend
        backend = PyTorchBackend()  # auto-detect cuda/mps/cpu
        device = backend._device
        dtype_str = str(backend._dtype).removeprefix("torch.")
        topk = pytorch_engine.top_k_ids_and_weights
    else:
        raise ValueError(
            f"MODEL_SHARD_BACKEND={name!r} not recognized "
            "(expected 'mlx' or 'pytorch')"
        )
    backend.load(hf_id)
    return backend, hf_id, device, dtype_str, topk


def _greedy_decode_with_topk(
    backend: Any, topk: Any, prompt_ids: list[int], n_positions: int, k: int,
) -> list[dict]:
    """Prefill + greedy decode through the backend. At each position record
    top-K (ids, weights). Returns a list of length n_positions, each entry
    ``{"ids": [...K], "weights": [...K]}``."""
    cache = backend.make_cache()
    h = backend.embed(prompt_ids)
    masks = backend.make_masks(h, cache)
    num_layers = backend.num_layers()
    for i in range(num_layers):
        h = backend.run_layer_atomic(i, h, cache, masks)
    logits = backend.finalize(h)
    out: list[dict] = []
    ids, weights = topk(logits, k=k)
    out.append({"ids": ids, "weights": weights})
    for _ in range(n_positions - 1):
        tok_id = ids[0]
        h = backend.embed([tok_id])
        masks = backend.make_masks(h, cache)
        for i in range(num_layers):
            h = backend.run_layer_atomic(i, h, cache, masks)
        logits = backend.finalize(h)
        ids, weights = topk(logits, k=k)
        out.append({"ids": ids, "weights": weights})
    return out


def main() -> None:
    args = _parse_args()
    backend, hf_id, device, dtype_str, topk = _load_backend(args.model)
    backend_name: str = backend.name

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_id)

    fixture: dict = {
        "model_id": hf_id,
        "backend": backend_name,
        "device": device,
        "dtype": dtype_str,
        "n_positions": N_POSITIONS,
        "top_k_recorded": TOP_K_RECORDED,
        "generator": f"{backend_name} greedy decode + top-{TOP_K_RECORDED} record",
        "prompts": [],
    }

    for prompt in PROMPTS:
        prompt_ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
        top_k_per_position = _greedy_decode_with_topk(
            backend, topk, prompt_ids, N_POSITIONS, TOP_K_RECORDED,
        )
        fixture["prompts"].append({
            "prompt": prompt,
            "prompt_ids": prompt_ids,
            "top_k_per_position": top_k_per_position,
        })

    out_path = (
        Path(__file__).resolve().parent.parent
        / "tests" / "fixtures" / f"{backend_name}_tier1_tokens.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
