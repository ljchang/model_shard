"""Phase 7-C-1 Task 3: pt_moe primitives.

Unit tests use synthetic modules that mirror the HF Gemma4 MoE layer
shape — stacked expert tensors, router with norm + proj + scales,
self-attn that accepts HF-shaped kwargs.
No real HF model load.
"""
from __future__ import annotations

from typing import ClassVar

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402, N812
from torch import nn  # noqa: E402

from model_shard import pt_moe  # noqa: E402

# ---- Synthetic modules mirroring HF ------------------------------------

class _SynthSelfAttn(nn.Module):
    """Mirrors Gemma4TextAttention: accepts HF-shaped kwargs, returns
    (attn_output, attn_weights or None) tuple."""
    def __init__(self, hidden: int):
        super().__init__()
        self.proj = nn.Linear(hidden, hidden, bias=False)
        self.last_kwargs: dict = {}

    def forward(
        self, hidden_states=None, position_embeddings=None,
        attention_mask=None, position_ids=None,
        past_key_values=None, shared_kv_states=None,
        **kwargs,
    ):
        self.last_kwargs = {
            "position_embeddings": position_embeddings,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
        }
        return self.proj(hidden_states), None


class _SynthRouter(nn.Module):
    """Mirrors HF Gemma4TextRouter: 3-tuple return (probs, weights, index)
    with internal norm + scale + per_expert_scale."""
    def __init__(self, hidden: int, num_experts: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.scale = nn.Parameter(torch.ones(hidden))
        self.proj = nn.Linear(hidden, num_experts, bias=False)
        self.per_expert_scale = nn.Parameter(torch.ones(num_experts))

    def forward(self, h: torch.Tensor):
        # Input shape: [B*S, H] (flat, per HF)
        x = self.norm(h) * self.scale
        logits = self.proj(x)
        probs = torch.softmax(logits, dim=-1)
        top_w, top_i = torch.topk(probs, self.top_k, dim=-1)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)
        top_w = top_w * self.per_expert_scale[top_i]
        return probs, top_w, top_i


class _SynthExperts(nn.Module):
    def __init__(self, num_experts: int, hidden: int, inter: int):
        super().__init__()
        self.num_experts = num_experts
        self.gate_up_proj = nn.Parameter(torch.randn(num_experts, 2 * inter, hidden))
        self.down_proj = nn.Parameter(torch.randn(num_experts, hidden, inter))


class _SynthSharedMLP(nn.Module):
    """Mimics Gemma4TextMLP: act(gate_proj(x)) * up_proj(x) -> down_proj."""
    def __init__(self, hidden: int, inter: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _SynthDecoderLayer(nn.Module):
    def __init__(
        self, hidden: int = 8, inter: int = 16, num_experts: int = 4, top_k: int = 2,
    ):
        super().__init__()
        self.layer_type = "full_attention"
        self.self_attn = _SynthSelfAttn(hidden)
        self.input_layernorm = nn.LayerNorm(hidden)
        self.post_attention_layernorm = nn.LayerNorm(hidden)
        self.pre_feedforward_layernorm = nn.LayerNorm(hidden)
        self.post_feedforward_layernorm = nn.LayerNorm(hidden)
        self.post_feedforward_layernorm_1 = nn.LayerNorm(hidden)
        self.pre_feedforward_layernorm_2 = nn.LayerNorm(hidden)
        self.post_feedforward_layernorm_2 = nn.LayerNorm(hidden)
        self.mlp = _SynthSharedMLP(hidden, inter)
        self.router = _SynthRouter(hidden, num_experts, top_k)
        self.experts = _SynthExperts(num_experts, hidden, inter)


class _SynthRotaryEmb(nn.Module):
    def __init__(self, head_dim: int = 4):
        super().__init__()
        self.head_dim = head_dim

    def forward(self, x, position_ids, layer_type=None):
        seq_len = position_ids.shape[-1]
        return (torch.ones((1, seq_len, self.head_dim)),
                torch.zeros((1, seq_len, self.head_dim)))


class _SynthTextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_SynthDecoderLayer()])
        self.rotary_emb = _SynthRotaryEmb(head_dim=4)


class _SynthModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _SynthTextModel()

        class _Cfg:
            layer_types: ClassVar[list[str]] = ["full_attention"]
            hidden_size_per_layer_input = 0
        self.config = _Cfg()


def _mk_model() -> _SynthModel:
    torch.manual_seed(42)
    return _SynthModel().eval()


# ---- Tests --------------------------------------------------------------

def test_run_attention_and_route_shapes():
    m = _mk_model()
    h = torch.randn((1, 3, 8))
    cos = torch.ones((1, 3, 4))
    sin = torch.zeros((1, 3, 4))
    masks = ({"full_attention": (cos, sin)}, None)
    post_attn, top_k_ids, top_k_weights = pt_moe.run_attention_and_route(
        m, h, layer_idx=0, cache=None, masks=masks, heat_observer=None,
    )
    assert post_attn.shape == (1, 3, 8)
    # Router is called on flat [B*S, H] = [3, 8], so top_k_* is [3, K=2]
    assert top_k_ids.shape == (3, 2)
    assert top_k_weights.shape == (3, 2)


def test_run_attention_and_route_passes_kwargs_to_self_attn():
    m = _mk_model()
    h = torch.randn((1, 3, 8))
    cos = torch.ones((1, 3, 4))
    sin = torch.zeros((1, 3, 4))
    masks = ({"full_attention": (cos, sin)}, None)

    class _FakeCache:
        def get_seq_length(self):
            return 0

    pt_moe.run_attention_and_route(
        m, h, layer_idx=0, cache=_FakeCache(), masks=masks, heat_observer=None,
    )
    kwargs = m.model.layers[0].self_attn.last_kwargs
    cos_recv, _sin_recv = kwargs["position_embeddings"]
    assert torch.equal(cos_recv, cos)
    assert kwargs["position_ids"].shape == (1, 3)


def test_run_attention_and_route_fires_heat_observer():
    m = _mk_model()
    h = torch.randn((1, 3, 8))
    cos = torch.ones((1, 3, 4))
    sin = torch.zeros((1, 3, 4))
    masks = ({"full_attention": (cos, sin)}, None)
    calls: list[tuple[int, int, float]] = []
    pt_moe.run_attention_and_route(
        m, h, layer_idx=0, cache=None, masks=masks,
        heat_observer=lambda L, E, w: calls.append((L, E, float(w))),  # noqa: N803
    )
    # 3 positions * 2 experts = 6 observations
    assert len(calls) == 6
    assert all(L == 0 for L, _, _ in calls)


def test_run_attention_and_route_returns_ids_then_weights():
    """Our external API returns (post_attn, top_k_ids, top_k_weights) —
    index first, weights second (flipped from HF router's internal order)."""
    m = _mk_model()
    h = torch.randn((1, 2, 8))
    cos = torch.ones((1, 2, 4))
    sin = torch.zeros((1, 2, 4))
    masks = ({"full_attention": (cos, sin)}, None)
    _, top_k_ids, top_k_weights = pt_moe.run_attention_and_route(
        m, h, layer_idx=0, cache=None, masks=masks,
    )
    # ids are long/int; weights are float
    assert top_k_ids.dtype in (torch.long, torch.int64, torch.int32)
    assert top_k_weights.dtype.is_floating_point


def test_run_shared_expert_applies_pre_feedforward_layernorm():
    m = _mk_model()
    h = torch.randn((1, 3, 8))
    out = pt_moe.run_shared_expert(m, h, layer_idx=0)
    assert out.shape == (1, 3, 8)
    layer = m.model.layers[0]
    expected = layer.mlp(layer.pre_feedforward_layernorm(h))
    assert torch.allclose(out, expected)


def test_run_selected_experts_applies_pre_feedforward_layernorm_2():
    m = _mk_model()
    h = torch.randn((1, 3, 8))
    out = pt_moe.run_selected_experts(m, h, layer_idx=0, expert_ids=[1])
    assert out[1].shape == (1, 3, 8)
    layer = m.model.layers[0]
    # Per HF: MoE normalizes via pre_feedforward_layernorm_2 on flat residual.
    flat = h.reshape(-1, h.shape[-1])
    normed = layer.pre_feedforward_layernorm_2(flat)
    e = layer.experts
    gu = F.linear(normed, e.gate_up_proj[1])
    g, u = gu.chunk(2, dim=-1)
    mid = F.silu(g) * u
    expected_flat = F.linear(mid, e.down_proj[1])
    expected = expected_flat.reshape(h.shape)
    assert torch.allclose(out[1], expected, atol=1e-5)


def test_run_selected_experts_returns_expert_id_keyed_dict():
    m = _mk_model()
    h = torch.randn((1, 3, 8))
    out = pt_moe.run_selected_experts(m, h, layer_idx=0, expert_ids=[0, 2, 3])
    assert set(out.keys()) == {0, 2, 3}
    for v in out.values():
        assert v.shape == (1, 3, 8)


def test_aggregate_experts_applies_post_feedforward_layernorms():
    m = _mk_model()
    per_pos_expert_outs = {
        0: torch.full((1, 1, 8), 1.0),
        1: torch.full((1, 1, 8), 2.0),
    }
    ids = [0, 1]
    weights = torch.tensor([[0.25, 0.75]])
    shared = torch.full((1, 1, 8), 10.0)
    out = pt_moe.aggregate_experts(
        m, layer_idx=0,
        expert_outputs=per_pos_expert_outs, top_k_ids=ids,
        top_k_weights=weights, shared_out=shared,
    )
    layer = m.model.layers[0]
    moe_branch = 0.25 * per_pos_expert_outs[0] + 0.75 * per_pos_expert_outs[1]
    expected = (
        layer.post_feedforward_layernorm_1(shared)
        + layer.post_feedforward_layernorm_2(moe_branch)
    )
    assert out.shape == (1, 1, 8)
    assert torch.allclose(out, expected, atol=1e-5)
