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

from model_shard.mlx_engine import (
    embed_tokens,
    finalize,
    make_cache,
    make_masks,
    run_layers,
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
