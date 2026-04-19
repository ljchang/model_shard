"""Phase 7-B Task 7: PyTorch Tier 1 regression test.

Requires CUDA + ~54 GB VRAM (DGX Spark). Skipped on other hosts. Compares
generated tokens against a fixture pre-generated on Spark and committed
to ``tests/fixtures/pytorch_tier1_tokens.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("requires CUDA", allow_module_level=True)

from transformers import AutoTokenizer  # noqa: E402

from model_shard.backends import PyTorchBackend  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "pytorch_tier1_tokens.json"


def _fixture() -> dict:
    if not FIXTURE.exists():
        pytest.skip(f"fixture missing: {FIXTURE} (run scripts/generate_pytorch_tier1_fixture.py)")
    data = json.loads(FIXTURE.read_text())
    if data.get("_placeholder"):
        pytest.skip("fixture is placeholder; regenerate on Spark via scripts/generate_pytorch_tier1_fixture.py")
    return data


@pytest.fixture(scope="module")
def fixture() -> dict:
    return _fixture()


@pytest.fixture(scope="module")
def backend(fixture: dict) -> PyTorchBackend:
    b = PyTorchBackend(device="cuda")
    b.load(fixture["model_id"])
    return b


@pytest.mark.slow
@pytest.mark.cuda
def test_tier1_tokens_match_fixture_top1(backend, fixture):
    """For each prompt in the fixture, greedy-decode N tokens through the
    backend's forward pass and compare top-1 IDs against the fixture."""
    _ = AutoTokenizer.from_pretrained(fixture["model_id"])
    for case in fixture["prompts"]:
        prompt_ids = case["prompt_ids"]
        expected_ids = case["generated_ids"]
        cache = backend.make_cache()
        h = backend.embed(prompt_ids)
        masks = backend.make_masks(h, cache)
        num_layers = backend.num_layers()
        for i in range(num_layers):
            h = backend.run_layer_atomic(i, h, cache, masks)
        logits = backend.finalize(h)
        token_id = backend.argmax_last(logits)
        got_ids = [token_id]
        for _ in range(fixture["n_positions"] - 1):
            h = backend.embed([token_id])
            masks = backend.make_masks(h, cache)
            for i in range(num_layers):
                h = backend.run_layer_atomic(i, h, cache, masks)
            logits = backend.finalize(h)
            token_id = backend.argmax_last(logits)
            got_ids.append(token_id)
        assert got_ids == expected_ids[:len(got_ids)], (
            f"prompt={case['prompt']!r}: got {got_ids}, expected {expected_ids[:len(got_ids)]}"
        )
