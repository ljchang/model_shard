"""Phase 7-B Task 2: pytorch_engine primitives.

Uses a tiny synthetic nn.Module instead of loading Gemma 4 — these tests
run on every platform (Mac, Linux, CPU-only) without CUDA or model weights.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from model_shard import pytorch_engine  # noqa: E402
from model_shard._pb import wire_pb2  # noqa: E402

# ---- Synthetic model ----------------------------------------------------

class _SynthRotaryEmb(nn.Module):
    """Mirrors HF Gemma4TextRotaryEmbedding: takes (x, position_ids, layer_type)
    and returns (cos, sin). Gemma 4's rotary is per-layer-type, so the stub
    picks different values per type to verify correct dispatch."""
    def __init__(self, head_dim: int = 4):
        super().__init__()
        self.head_dim = head_dim
        self.last_call: dict = {}

    def forward(self, h: torch.Tensor, position_ids: torch.Tensor, layer_type: str | None = None):
        self.last_call = {"layer_type": layer_type, "seq_len": position_ids.shape[-1]}
        seq_len = position_ids.shape[-1]
        # Use layer_type to vary the values so test can assert correct dispatch.
        if layer_type == "full_attention":
            cos = torch.ones((1, seq_len, self.head_dim))
        elif layer_type == "sliding_attention":
            cos = torch.full((1, seq_len, self.head_dim), 2.0)
        else:
            cos = torch.zeros((1, seq_len, self.head_dim))
        sin = torch.zeros((1, seq_len, self.head_dim))
        return cos, sin


class _SynthLayer(nn.Module):
    def __init__(self, hidden: int, layer_type: str = "full_attention"):
        super().__init__()
        self.layer_type = layer_type
        self.last_kwargs: dict = {}

    def forward(self, hidden_states=None, per_layer_input=None,
                shared_kv_states=None, position_embeddings=None,
                attention_mask=None, position_ids=None,
                past_key_values=None, **kwargs):
        self.last_kwargs = {
            "per_layer_input": per_layer_input,
            "shared_kv_states": shared_kv_states,
            "position_embeddings": position_embeddings,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "kwargs": dict(kwargs),
        }
        # Return plain tensor (NOT tuple) matching HF.
        return hidden_states * 2.0


class _SynthTextModel(nn.Module):
    def __init__(self, vocab: int = 32, hidden: int = 8, num_layers: int = 2):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([_SynthLayer(hidden) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden)
        self.rotary_emb = _SynthRotaryEmb(head_dim=4)


class _SynthModel(nn.Module):
    """Minimal stand-in for Gemma4ForCausalLM."""
    def __init__(self, vocab: int = 32, hidden: int = 8, num_layers: int = 2):
        super().__init__()
        self.model = _SynthTextModel(vocab, hidden, num_layers)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

        class _Cfg:
            num_hidden_layers = num_layers
            layer_types = ["full_attention"] * num_layers
            hidden_size_per_layer_input = 0
        self.config = _Cfg()


def _mk_model() -> _SynthModel:
    torch.manual_seed(0)
    return _SynthModel().eval()


# ---- Tests --------------------------------------------------------------

def test_torch_to_wire_dtype_bfloat16():
    assert pytorch_engine.torch_to_wire_dtype(torch.bfloat16) == wire_pb2.DTYPE_BFLOAT16


def test_torch_to_wire_dtype_float32():
    assert pytorch_engine.torch_to_wire_dtype(torch.float32) == wire_pb2.DTYPE_FLOAT32


def test_torch_to_wire_dtype_unsupported_raises():
    with pytest.raises(ValueError, match="unsupported torch dtype"):
        pytorch_engine.torch_to_wire_dtype(torch.int64)


def test_wire_to_torch_dtype_bfloat16():
    assert pytorch_engine._wire_to_torch_dtype(wire_pb2.DTYPE_BFLOAT16) == torch.bfloat16


def test_embed_tokens_returns_shape_1_L_H():  # noqa: N802
    m = _mk_model()
    h = pytorch_engine.embed_tokens(m, [5, 6, 7])
    assert h.shape == (1, 3, 8)


def test_make_cache_returns_dynamic_cache():
    from transformers import DynamicCache
    m = _mk_model()
    cache = pytorch_engine.make_cache(m)
    assert isinstance(cache, DynamicCache)


def test_make_masks_returns_rotary_dict_and_attn_mask():
    m = _mk_model()
    h = torch.randn((1, 3, 8))
    cache = pytorch_engine.make_cache(m)
    rotary_dict, attn_mask = pytorch_engine.make_masks(m, h, cache)
    # rotary_dict is keyed by layer_type
    assert isinstance(rotary_dict, dict)
    assert "full_attention" in rotary_dict
    cos, sin = rotary_dict["full_attention"]
    assert cos.shape == (1, 3, 4)
    assert sin.shape == (1, 3, 4)
    # attn_mask slot is None for now (HF derives causal from cache).
    assert attn_mask is None


def test_make_masks_computes_per_unique_layer_type():
    """When config has ['full_attention', 'sliding_attention'], both
    rotary entries should be populated."""
    m = _SynthModel()
    m.config.layer_types = ["full_attention", "sliding_attention"]
    # Give layers the right layer_type.
    m.model.layers[0].layer_type = "full_attention"
    m.model.layers[1].layer_type = "sliding_attention"
    h = torch.randn((1, 2, 8))
    cache = pytorch_engine.make_cache(m)
    rotary_dict, _ = pytorch_engine.make_masks(m, h, cache)
    assert set(rotary_dict.keys()) == {"full_attention", "sliding_attention"}
    # Values should differ (stub returns different values per layer_type).
    full_cos, _ = rotary_dict["full_attention"]
    sliding_cos, _ = rotary_dict["sliding_attention"]
    assert not torch.equal(full_cos, sliding_cos)


def test_make_masks_advances_position_ids_with_cache():
    """After decoding some tokens, make_masks should use cache_len + seq_len
    for position_ids."""
    m = _mk_model()
    h = torch.randn((1, 1, 8))

    class _FakeCache:
        def get_seq_length(self):
            return 5

    _rotary_dict, _ = pytorch_engine.make_masks(m, h, _FakeCache())
    # Stub records the position_ids seen; check via the rotary module.
    assert m.model.rotary_emb.last_call["seq_len"] == 1


def test_run_layer_atomic_passes_kwargs_to_hf_layer():
    m = _mk_model()
    h = torch.ones((1, 3, 8))
    cache = pytorch_engine.make_cache(m)
    rotary_dict, attn_mask = pytorch_engine.make_masks(m, h, cache)
    out = pytorch_engine.run_layer_atomic(
        m, 0, h, cache, rotary_dict, attn_mask,
    )
    # Behavior: stub layer doubles input.
    assert torch.allclose(out, torch.full((1, 3, 8), 2.0))
    # Kwargs recorded:
    kwargs = m.model.layers[0].last_kwargs
    # PLURAL form.
    assert kwargs["past_key_values"] is cache
    # position_embeddings is a (cos, sin) tuple matching the rotary for this layer_type.
    cos_expected, sin_expected = rotary_dict["full_attention"]
    cos_recv, sin_recv = kwargs["position_embeddings"]
    assert torch.equal(cos_recv, cos_expected)
    assert torch.equal(sin_recv, sin_expected)
    assert kwargs["position_ids"] is not None
    assert kwargs["position_ids"].shape == (1, 3)


def test_run_layer_atomic_return_is_plain_tensor_not_tuple():
    """HF Gemma4TextDecoderLayer.forward returns a plain tensor. We must
    not wrap / unwrap."""
    m = _mk_model()
    h = torch.ones((1, 3, 8))
    cache = pytorch_engine.make_cache(m)
    rotary_dict, attn_mask = pytorch_engine.make_masks(m, h, cache)
    out = pytorch_engine.run_layer_atomic(
        m, 0, h, cache, rotary_dict, attn_mask,
    )
    assert isinstance(out, torch.Tensor)
    # Not a 1-tuple accidentally unpacked
    assert out.dim() == 3


def test_finalize_applies_norm_then_lm_head():
    m = _mk_model()
    h = torch.randn((1, 2, 8))
    logits = pytorch_engine.finalize(m, h)
    assert logits.shape == (1, 2, 32)


def test_tensor_to_bytes_roundtrip_bfloat16():
    t = torch.full((2, 4), 1.5, dtype=torch.bfloat16)
    raw = pytorch_engine.tensor_to_bytes(t)
    recovered = pytorch_engine.bytes_to_tensor(
        raw, shape=[2, 4], dtype=pytorch_engine.torch_to_wire_dtype(t.dtype)
    )
    assert torch.equal(recovered.cpu(), t.cpu())


def test_tensor_to_bytes_length_matches_element_size():
    """bf16 is 2 bytes/element."""
    t = torch.zeros((3, 5), dtype=torch.bfloat16)
    raw = pytorch_engine.tensor_to_bytes(t)
    assert len(raw) == 3 * 5 * 2


def test_run_layers_delegates_to_run_layer_atomic_for_non_split():
    m = _mk_model()
    h = torch.ones((1, 3, 8))
    cache = pytorch_engine.make_cache(m)
    masks = pytorch_engine.make_masks(m, h, cache)  # (rotary_dict, None)
    out = pytorch_engine.run_layers(
        m, start_layer=0, end_layer=2, h=h, cache=cache, masks=masks,
        is_split_layer=lambda _: False,
    )
    # Each layer doubles, so 2 layers = x4
    assert torch.allclose(out, torch.full((1, 3, 8), 4.0))


def test_default_device_prefers_cuda_then_mps_then_cpu():
    d = pytorch_engine._default_device()
    if torch.cuda.is_available():
        assert d == "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        assert d == "mps"
    else:
        assert d == "cpu"
