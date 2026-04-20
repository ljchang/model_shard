#!/usr/bin/env python
"""Phase 7-B / 7-C-1: one-shot fixture generator for Tier-1 PyTorch tokens.

Run ONCE on DGX Spark (or any CUDA host with ~54 GB VRAM) to produce
``tests/fixtures/pytorch_tier1_tokens.json``. Commit the fixture.
``tests/test_pytorch_tier1.py`` then compares against this fixture so
every subsequent run is an *internal* regression — "PyTorchBackend's
output hasn't drifted from its last-blessed state" — not a cross-
framework equivalence check vs HF's own ``model.generate()``.

Why greedy-decode through PyTorchBackend (not model.generate)?

The distributed engine's prefill + per-layer atomic decode loop is
structurally different from HF's batched ``generate()`` — different
attention backend dispatch order, different ``cache_position`` tracking,
and slight numerical drift accumulates after a few positions. Generating
the fixture via the same code path the test exercises makes this a
faithful self-regression.

Cross-framework equivalence (MLX vs PyTorch vs HF) is explicitly
Phase 7-C-2 scope.

Usage:
    uv run python scripts/generate_pytorch_tier1_fixture.py
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from model_shard.backends import PyTorchBackend

PROMPTS = [
    "The quick brown fox",
    "In a galaxy far far away",
    "Once upon a time",
]
N_POSITIONS = 10


def _greedy_decode_via_backend(
    backend: PyTorchBackend, prompt_ids: list[int], n_positions: int,
) -> list[int]:
    """Prefill + greedy decode through PyTorchBackend. Mirrors exactly the
    per-layer loop in tests/test_pytorch_tier1.py so the fixture is a
    faithful capture of the backend's own output."""
    cache = backend.make_cache()
    h = backend.embed(prompt_ids)
    masks = backend.make_masks(h, cache)
    num_layers = backend.num_layers()
    for i in range(num_layers):
        h = backend.run_layer_atomic(i, h, cache, masks)
    logits = backend.finalize(h)
    token_id = backend.argmax_last(logits)
    out = [token_id]
    for _ in range(n_positions - 1):
        h = backend.embed([token_id])
        masks = backend.make_masks(h, cache)
        for i in range(num_layers):
            h = backend.run_layer_atomic(i, h, cache, masks)
        logits = backend.finalize(h)
        token_id = backend.argmax_last(logits)
        out.append(token_id)
    return out


def main() -> None:
    hf_id = "google/gemma-4-26B-A4B-it"
    backend = PyTorchBackend()
    if backend._device != "cuda":
        print(
            f"WARNING: backend device is {backend._device}, not cuda. "
            "Fixture should ideally be generated on Spark."
        )
    backend.load(hf_id)
    tok = AutoTokenizer.from_pretrained(hf_id)

    fixture: dict = {
        "model_id": hf_id,
        "device": backend._device,
        "dtype": str(backend._dtype).removeprefix("torch."),
        "n_positions": N_POSITIONS,
        "generator": "PyTorchBackend greedy decode (prefill + per-layer atomic)",
        "prompts": [],
    }

    for prompt in PROMPTS:
        prompt_ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
        with torch.no_grad():
            generated = _greedy_decode_via_backend(
                backend, prompt_ids, N_POSITIONS,
            )
        fixture["prompts"].append({
            "prompt": prompt,
            "prompt_ids": prompt_ids,
            "generated_ids": generated,
        })

    out_path = (
        Path(__file__).resolve().parent.parent
        / "tests" / "fixtures" / "pytorch_tier1_tokens.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
