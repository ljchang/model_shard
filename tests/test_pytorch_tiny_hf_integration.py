"""Phase 7-C-1 Task 5: real-HF integration test on Mac CPU.

Builds a minimal Gemma4ForCausalLM from a hand-rolled Gemma4TextConfig
(random init, not pretrained) and runs it through PyTorchBackend. Catches
real-HF integration bugs (wrong kwargs, signature mismatches, missing
layernorms) that synthetic-test coverage misses. Runs on CPU; takes seconds.

Marked slow because it imports transformers and instantiates a random model.
The verified-working tiny config is from
``docs/superpowers/reference/2026-04-19-hf-gemma4-forward-signatures.md``.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from model_shard.backends import PyTorchBackend  # noqa: E402


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import Gemma4ForCausalLM
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    # Verified-working minimum config from the Task 1 reference doc:
    # Gemma4ForCausalLM(c).model_size ~905k params; forward produces
    # logits of shape (1, 8, 256).
    cfg = Gemma4TextConfig(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_experts=4,
        top_k_experts=2,
        layer_types=["full_attention", "full_attention"],
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        sliding_window=32,
        enable_moe_block=True,
        hidden_size_per_layer_input=0,
    )
    model = Gemma4ForCausalLM(cfg)
    model.eval()
    return model


@pytest.mark.slow
def test_pytorch_backend_forward_on_tiny_hf_model(tiny_model):
    """End-to-end: embed -> make_masks -> run_layer_atomic x N -> finalize
    on a real (tiny, random-init) HF Gemma4ForCausalLM."""
    b = PyTorchBackend.from_loaded_model(tiny_model, device="cpu")
    token_ids = [5, 6, 7, 8]
    cache = b.make_cache()
    h = b.embed(token_ids)
    assert h.shape == (1, 4, 64)
    masks = b.make_masks(h, cache)
    # masks is (rotary_dict, attn_mask_dict_or_None)
    rotary_dict, _ = masks
    assert "full_attention" in rotary_dict
    num_layers = b.num_layers()
    assert num_layers == 2
    for i in range(num_layers):
        h = b.run_layer_atomic(i, h, cache, masks)
        assert h.shape == (1, 4, 64)
        assert torch.isfinite(h).all(), f"layer {i} produced non-finite values"
    logits = b.finalize(h)
    assert logits.shape == (1, 4, 256)
    assert torch.isfinite(logits).all()
    # Greedy-decode 1 token via argmax_last.
    token_id = b.argmax_last(logits)
    assert 0 <= token_id < 256


@pytest.mark.slow
def test_pytorch_backend_two_step_decode_on_tiny_hf_model(tiny_model):
    """Prefill + one decode step, confirming cache grows correctly."""
    b = PyTorchBackend.from_loaded_model(tiny_model, device="cpu")
    cache = b.make_cache()
    prompt_ids = [1, 2, 3]

    # Prefill
    h = b.embed(prompt_ids)
    masks = b.make_masks(h, cache)
    for i in range(b.num_layers()):
        h = b.run_layer_atomic(i, h, cache, masks)
    logits = b.finalize(h)
    token_id = b.argmax_last(logits)
    assert 0 <= token_id < 256
    # After prefill, cache should reflect the prompt length.
    assert cache.get_seq_length() == len(prompt_ids)

    # Decode step 1
    h = b.embed([token_id])
    assert h.shape == (1, 1, 64)
    masks = b.make_masks(h, cache)
    for i in range(b.num_layers()):
        h = b.run_layer_atomic(i, h, cache, masks)
    logits = b.finalize(h)
    next_id = b.argmax_last(logits)
    assert 0 <= next_id < 256
    # Cache should now reflect prefill_len + 1 = 4 tokens.
    assert cache.get_seq_length() == len(prompt_ids) + 1


@pytest.mark.slow
def test_pytorch_backend_moe_layer_via_split_layer_primitives(tiny_model):
    """Exercise the split-layer backend primitives against a real HF MoE
    decoder layer. This smoke-tests run_attention_and_route +
    run_shared_expert + run_selected_experts + aggregate_experts against
    real HF layer internals."""
    b = PyTorchBackend.from_loaded_model(tiny_model, device="cpu")
    token_ids = [1, 2, 3]
    cache = b.make_cache()
    h = b.embed(token_ids)
    masks = b.make_masks(h, cache)

    # Run layer 0's attention+route
    post_attn, top_k = b.run_attention_and_route(
        layer_idx=0, h=h, cache=cache, masks=masks,
    )
    top_k_ids, top_k_weights = top_k
    assert post_attn.shape == (1, 3, 64)
    # Router runs on flat [B*S, H] = [3, H], top_k_* is [3, K=2]
    assert top_k_ids.shape == (3, 2)
    assert top_k_weights.shape == (3, 2)

    # Shared expert (dense MLP)
    shared_out = b.run_shared_expert(layer_idx=0, h=post_attn)
    assert shared_out.shape == (1, 3, 64)
    assert torch.isfinite(shared_out).all()

    # Selected experts — run all unique ids seen in top_k
    unique_ids = sorted({int(e) for e in top_k_ids.reshape(-1).tolist()})
    expert_outputs = b.run_selected_experts(
        layer_idx=0, h=post_attn, expert_ids=unique_ids,
    )
    for eid in unique_ids:
        assert expert_outputs[eid].shape == (1, 3, 64)
        assert torch.isfinite(expert_outputs[eid]).all()
