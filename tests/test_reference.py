"""Tests for the ReferenceModel oracle.

The oracle exposes two operations Phase 1 acceptance tests depend on:
  * generate_greedy — for Tier 1 (exact-match generated tokens)
  * prefill_trace  — for Tier 2 (per-layer hidden-state tolerance)
"""

import mlx.core as mx
import pytest

from model_shard.reference import PrefillTrace, ReferenceModel


@pytest.fixture(scope="module")
def reference(loaded_model) -> ReferenceModel:  # type: ignore[no-untyped-def]
    return ReferenceModel(loaded_model)


@pytest.mark.slow
def test_generate_greedy_returns_requested_token_count(reference: ReferenceModel) -> None:
    tokens = reference.generate_greedy(prompt_tokens=[1, 2, 3], max_new_tokens=5)
    assert len(tokens) == 5


@pytest.mark.slow
def test_generate_greedy_is_deterministic(reference: ReferenceModel) -> None:
    """Same prompt must produce identical token sequence on repeated calls."""
    prompt = [105, 2364, 236761]  # arbitrary token ids
    first = reference.generate_greedy(prompt_tokens=prompt, max_new_tokens=8)
    second = reference.generate_greedy(prompt_tokens=prompt, max_new_tokens=8)
    assert first == second


@pytest.mark.slow
def test_prefill_trace_captures_hidden_state_per_layer(reference: ReferenceModel) -> None:
    prompt = [1, 2, 3, 4, 5]
    trace = reference.prefill_trace(prompt)
    assert isinstance(trace, PrefillTrace)
    # One hidden-state snapshot per layer (input to layers 0..N-1).
    assert len(trace.layer_inputs) == reference.num_layers


@pytest.mark.slow
def test_prefill_trace_hidden_state_shapes_consistent(reference: ReferenceModel) -> None:
    prompt = [10, 20, 30, 40]
    trace = reference.prefill_trace(prompt)
    expected = (1, len(prompt), 2816)
    for i, h in enumerate(trace.layer_inputs):
        assert h.shape == expected, f"layer {i}: got {h.shape}, want {expected}"
    assert trace.final_hidden.shape == expected


@pytest.mark.slow
def test_prefill_trace_logits_match_generate_greedy_first_token(
    reference: ReferenceModel,
) -> None:
    """argmax of last-position prefill logits == first token from generate_greedy."""
    prompt = [5, 100, 200]
    trace = reference.prefill_trace(prompt)
    last_logits = trace.logits[0, -1, :]
    prefill_next = int(mx.argmax(last_logits).item())

    first_generated = reference.generate_greedy(prompt_tokens=prompt, max_new_tokens=1)[0]
    assert prefill_next == first_generated


@pytest.mark.slow
def test_tokenize_roundtrip(reference: ReferenceModel) -> None:
    text = "Hello, world."
    ids = reference.tokenize(text)
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    decoded = reference.detokenize(ids)
    assert "Hello" in decoded
