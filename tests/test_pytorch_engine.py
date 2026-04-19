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

class _SynthLayer(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.layer_type = "full_attention"

    def forward(self, h, *args, **kwargs):
        return h * 2.0


class _SynthTextModel(nn.Module):
    def __init__(self, vocab: int = 32, hidden: int = 8, num_layers: int = 2):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([_SynthLayer(hidden) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden)


class _SynthModel(nn.Module):
    """Minimal stand-in for Gemma4ForCausalLM."""
    def __init__(self, vocab: int = 32, hidden: int = 8, num_layers: int = 2):
        super().__init__()
        self.model = _SynthTextModel(vocab, hidden, num_layers)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

        class _Cfg:
            num_hidden_layers = num_layers
            layer_types = ["full_attention"] * num_layers
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


def test_run_layer_atomic_doubles_synthetic_layer():
    """_SynthLayer.forward returns h * 2.0."""
    m = _mk_model()
    h = torch.ones((1, 3, 8))
    cache = pytorch_engine.make_cache(m)
    global_mask, sliding_mask = pytorch_engine.make_masks(m, h, cache)
    out = pytorch_engine.run_layer_atomic(m, 0, h, cache, global_mask, sliding_mask)
    assert out.shape == (1, 3, 8)
    assert torch.allclose(out, torch.full((1, 3, 8), 2.0))


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
    """run_layers loops over the shard's layer range calling run_layer_atomic
    on each. No provenance append at this layer of the stack (that's node.py)."""
    m = _mk_model()
    h = torch.ones((1, 3, 8))
    cache = pytorch_engine.make_cache(m)
    masks = pytorch_engine.make_masks(m, h, cache)
    # Layers 0 and 1 both double, so output should be h * 4.0.
    out = pytorch_engine.run_layers(
        m, start_layer=0, end_layer=2, h=h, cache=cache, masks=masks,
        is_split_layer=lambda _: False,
    )
    assert torch.allclose(out, torch.full((1, 3, 8), 4.0))


def test_default_device_prefers_cuda_then_mps_then_cpu():
    d = pytorch_engine._default_device()
    if torch.cuda.is_available():
        assert d == "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        assert d == "mps"
    else:
        assert d == "cpu"
