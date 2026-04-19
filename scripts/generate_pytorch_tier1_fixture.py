#!/usr/bin/env python
"""Phase 7-B: one-shot fixture generator for Tier-1 PyTorch tokens.

Run ONCE on DGX Spark (or any CUDA host with ~54 GB VRAM) to produce
``tests/fixtures/pytorch_tier1_tokens.json``. Commit the fixture.
``tests/test_pytorch_tier1.py`` then compares against this fixture so
every subsequent run is a regression test, not a re-generation.

Usage:
    uv run python scripts/generate_pytorch_tier1_fixture.py
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_shard import pytorch_engine

PROMPTS = [
    "The quick brown fox",
    "In a galaxy far far away",
    "Once upon a time",
]
N_POSITIONS = 10


def main() -> None:
    hf_id = "google/gemma-4-26B-A4B-it"
    device = pytorch_engine._default_device()
    if device != "cuda":
        print(f"WARNING: device is {device}, not cuda. Fixture should ideally be generated on Spark.")
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.bfloat16, device_map=device,
    ).eval()

    fixture: dict = {
        "model_id": hf_id,
        "device": device,
        "dtype": "bfloat16",
        "n_positions": N_POSITIONS,
        "prompts": [],
    }

    for prompt in PROMPTS:
        input_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=N_POSITIONS,
                do_sample=False,
                temperature=1.0,
                use_cache=True,
            )
        new_ids = out[0, input_ids.shape[1]:].tolist()
        fixture["prompts"].append({
            "prompt": prompt,
            "prompt_ids": input_ids[0].tolist(),
            "generated_ids": new_ids[:N_POSITIONS],
        })

    out_path = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "pytorch_tier1_tokens.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
