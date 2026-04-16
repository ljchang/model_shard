"""Forward-pass building-block tests for mlx_engine.

Two properties that must hold for Phase 1 to be correct:
  1. Our composed forward pass (embed → run_layers → finalize) produces the
     exact same logits as mlx-vlm's LanguageModel.__call__. If this drifts,
     we've broken the port.
  2. Splitting the layer loop across multiple run_layers() invocations
     (sharing cache + masks) produces the same output as a single-shot
     call. This is the sharding correctness property — if this fails, no
     distributed topology can be correct.
"""

import mlx.core as mx
import pytest

from model_shard._pb import wire_pb2
from model_shard.mlx_engine import (
    bytes_to_tensor,
    embed_tokens,
    finalize,
    make_cache,
    make_masks,
    run_layers,
    tensor_to_bytes,
)


@pytest.mark.slow
def test_load_model_returns_handle_with_num_layers(loaded_model) -> None:  # type: ignore[no-untyped-def]
    assert loaded_model.num_layers == 30
    assert loaded_model.text_model is not None
    assert loaded_model.language_model is not None


@pytest.mark.slow
def test_full_forward_matches_mlx_vlm_language_model(loaded_model) -> None:  # type: ignore[no-untyped-def]
    """Composed (embed + run_layers + finalize) must match LanguageModel forward bit-for-bit."""
    lm = loaded_model
    token_ids = mx.array([[1, 17, 42, 100, 300]])

    # mlx-vlm's reference forward.
    vlm_logits = lm.language_model(token_ids).logits

    # My composed forward.
    h = embed_tokens(lm, token_ids)
    cache = make_cache(lm)
    global_mask, sliding_mask = make_masks(lm, h, cache)
    h = run_layers(lm, h, 0, lm.num_layers, cache, global_mask, sliding_mask)
    my_logits = finalize(lm, h)

    assert my_logits.shape == vlm_logits.shape
    # Same code path, same weights — should be exact, but allow tiny drift
    # for mx.compile fusion differences. In practice we expect zero.
    diff = mx.abs(my_logits - vlm_logits).max().item()
    assert diff < 1e-4, f"max abs diff {diff}"


@pytest.mark.slow
def test_split_layers_equivalent_to_single_shot(loaded_model) -> None:  # type: ignore[no-untyped-def]
    """run_layers([0, 30)) == run_layers([0, 10)) + run_layers([10, 20)) + run_layers([20, 30))."""
    lm = loaded_model
    token_ids = mx.array([[5, 10, 15, 20, 25]])

    # Single shot.
    h1 = embed_tokens(lm, token_ids)
    cache1 = make_cache(lm)
    gm1, sm1 = make_masks(lm, h1, cache1)
    h1 = run_layers(lm, h1, 0, lm.num_layers, cache1, gm1, sm1)
    out_single = lm.text_model.norm(h1)

    # Three shards, shared cache + masks.
    h2 = embed_tokens(lm, token_ids)
    cache2 = make_cache(lm)
    gm2, sm2 = make_masks(lm, h2, cache2)
    h2 = run_layers(lm, h2, 0, 10, cache2, gm2, sm2)
    h2 = run_layers(lm, h2, 10, 20, cache2, gm2, sm2)
    h2 = run_layers(lm, h2, 20, 30, cache2, gm2, sm2)
    out_split = lm.text_model.norm(h2)

    diff = mx.abs(out_single - out_split).max().item()
    assert diff < 1e-4, f"split produced different output: max abs diff {diff}"


@pytest.mark.slow
def test_make_cache_has_one_slot_per_non_shared_layer(loaded_model) -> None:  # type: ignore[no-untyped-def]
    """For 26B (num_kv_shared_layers=0), cache len == num_layers."""
    cache = make_cache(loaded_model)
    assert len(cache) == loaded_model.num_layers


@pytest.mark.slow
def test_embed_produces_expected_shape(loaded_model) -> None:  # type: ignore[no-untyped-def]
    lm = loaded_model
    token_ids = mx.array([[1, 2, 3, 4, 5, 6, 7]])
    h = embed_tokens(lm, token_ids)
    # [batch=1, seq=7, hidden=2816]
    assert h.shape == (1, 7, 2816)


@pytest.mark.slow
def test_finalize_produces_vocab_sized_logits(loaded_model) -> None:  # type: ignore[no-untyped-def]
    lm = loaded_model
    # Fake hidden state (hidden_size=2816, vocab_size=262144).
    h = mx.zeros((1, 3, 2816))
    logits = finalize(lm, h)
    assert logits.shape == (1, 3, 262144)


def test_tensor_bf16_roundtrip_is_lossless() -> None:
    arr = mx.random.normal((4, 3, 8), dtype=mx.bfloat16)
    raw = tensor_to_bytes(arr)
    restored = bytes_to_tensor(raw, shape=list(arr.shape), dtype=wire_pb2.DTYPE_BFLOAT16)
    assert mx.array_equal(arr, restored).item()
    assert restored.dtype == mx.bfloat16


def test_tensor_fp32_roundtrip_is_lossless() -> None:
    arr = mx.random.normal((2, 5), dtype=mx.float32)
    raw = tensor_to_bytes(arr)
    restored = bytes_to_tensor(raw, shape=list(arr.shape), dtype=wire_pb2.DTYPE_FLOAT32)
    assert mx.array_equal(arr, restored).item()
    assert restored.dtype == mx.float32


def test_tensor_bytes_size_matches_element_count() -> None:
    arr = mx.zeros((7, 11), dtype=mx.bfloat16)
    raw = tensor_to_bytes(arr)
    assert len(raw) == 7 * 11 * 2  # 2 bytes per bf16 element


@pytest.mark.slow
def test_decode_step_is_split_equivalent_with_disjoint_caches(loaded_model) -> None:  # type: ignore[no-untyped-def]
    """Prefill + a decode step across 3 disjoint caches matches the shared-cache path.

    Simulates the distributed case: each "shard" keeps its own KV cache containing
    only slots for its layer range. Empty slots at other indices should be inert
    for Phase 1 prompt lengths (< sliding_window).
    """
    lm = loaded_model
    prompt = mx.array([[1, 17, 42, 100, 300]])
    splits = [(0, 10), (10, 20), (20, 30)]

    # --- Shared-cache baseline ---
    shared_cache = make_cache(lm)
    h = embed_tokens(lm, prompt)
    gm, sm = make_masks(lm, h, shared_cache)
    h = run_layers(lm, h, 0, 30, shared_cache, gm, sm)
    baseline_prefill_logits = finalize(lm, h)
    next_tok = int(mx.argmax(baseline_prefill_logits[0, -1, :]).item())

    step_in = mx.array([[next_tok]])
    h = embed_tokens(lm, step_in)
    gm, sm = make_masks(lm, h, shared_cache)
    h = run_layers(lm, h, 0, 30, shared_cache, gm, sm)
    baseline_decode_logits = finalize(lm, h)

    # --- Disjoint per-shard caches ---
    shard_caches = [make_cache(lm) for _ in splits]

    def forward_across_shards(tokens: mx.array) -> mx.array:
        h_local = embed_tokens(lm, tokens)
        for (start, end), cache in zip(splits, shard_caches, strict=True):
            gm_local, sm_local = make_masks(lm, h_local, cache)
            h_local = run_layers(lm, h_local, start, end, cache, gm_local, sm_local)
        return finalize(lm, h_local)

    distributed_prefill_logits = forward_across_shards(prompt)
    next_tok_d = int(mx.argmax(distributed_prefill_logits[0, -1, :]).item())
    assert next_tok_d == next_tok, "prefill next-token disagreement"

    distributed_decode_logits = forward_across_shards(mx.array([[next_tok]]))

    prefill_diff = mx.abs(baseline_prefill_logits - distributed_prefill_logits).max().item()
    decode_diff = mx.abs(baseline_decode_logits - distributed_decode_logits).max().item()
    assert prefill_diff < 1e-4, f"prefill logit drift {prefill_diff}"
    assert decode_diff < 1e-4, f"decode logit drift {decode_diff}"
