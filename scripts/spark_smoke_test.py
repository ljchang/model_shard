#!/usr/bin/env python
"""Phase 7-B: manual DGX Spark smoke test.

Run from an interactive shell on the Spark host:
    MODEL_SHARD_BACKEND=pytorch uv run python scripts/spark_smoke_test.py \\
        --model <HF-id-or-local-path>

Loads the model, does a 10-token completion, prints timing. Not a
pytest — meant for humans to eyeball sanity after first deploy.
"""
from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoTokenizer

from model_shard.backends import PyTorchBackend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model id or local path.",
    )
    args = parser.parse_args()
    hf_id = args.model

    print(f"Torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"Compute capability: {torch.cuda.get_device_capability(0)}")

    tok = AutoTokenizer.from_pretrained(hf_id)

    t0 = time.time()
    b = PyTorchBackend(device="cuda")
    b.load(hf_id)
    print(f"Load: {time.time() - t0:.1f}s")

    prompt_ids = tok("The quick brown fox", return_tensors="pt").input_ids[0].tolist()
    cache = b.make_cache()
    h = b.embed(prompt_ids)
    masks = b.make_masks(h, cache)
    num_layers = b.num_layers()
    for i in range(num_layers):
        h = b.run_layer_atomic(i, h, cache, masks)
    logits = b.finalize(h)
    tok_id = b.argmax_last(logits)
    print(f"Decoded token 0: id={tok_id} str={tok.decode([tok_id])!r}")


if __name__ == "__main__":
    main()
