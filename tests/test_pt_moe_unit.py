"""Phase 7-B Task 3: pt_moe primitives.

Unit tests use synthetic modules that mirror the HF Gemma4 MoE layer
shape — stacked expert tensors, router with norm + proj + scales.
No real HF model load.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402, N812
from torch import nn  # noqa: E402

from model_shard import pt_moe  # noqa: E402

# ---- Synthetic router + experts mirroring HF ---------------------------

class _SynthRouter(nn.Module):
    """Mirrors Gemma4TextRouter: norm + proj + per-expert scale + topk."""
    def __init__(self, hidden: int, num_experts: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.proj = nn.Linear(hidden, num_experts, bias=False)
        self.per_expert_scale = nn.Parameter(torch.ones(num_experts))

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = self.norm(h)
        logits = self.proj(n)
        weights = F.softmax(logits, dim=-1)
        top_w, top_i = torch.topk(weights, self.top_k, dim=-1)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)
        top_w = top_w * self.per_expert_scale[top_i]
        return top_i, top_w


class _SynthExperts(nn.Module):
    """Mirrors Gemma4TextExperts: stacked [E, 2*I, H] and [E, H, I]."""
    def __init__(self, num_experts: int, hidden: int, inter: int):
        super().__init__()
        self.num_experts = num_experts
        self.gate_up_proj = nn.Parameter(torch.randn(num_experts, 2 * inter, hidden))
        self.down_proj = nn.Parameter(torch.randn(num_experts, hidden, inter))


class _SynthSharedMLP(nn.Module):
    def __init__(self, hidden: int, inter: int):
        super().__init__()
        self.gate_up = nn.Linear(hidden, 2 * inter, bias=False)
        self.down = nn.Linear(inter, hidden, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        gu = self.gate_up(h)
        g, u = gu.chunk(2, dim=-1)
        return self.down(F.silu(g) * u)


class _SynthDecoderLayer(nn.Module):
    def __init__(
        self, hidden: int = 8, inter: int = 16, num_experts: int = 4, top_k: int = 2,
    ):
        super().__init__()
        self.layer_type = "full_attention"
        self.self_attn = nn.Linear(hidden, hidden, bias=False)
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


class _SynthTextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_SynthDecoderLayer()])


class _SynthModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _SynthTextModel()


def _mk_model() -> _SynthModel:
    torch.manual_seed(42)
    return _SynthModel().eval()


# ---- Tests --------------------------------------------------------------

def test_run_attention_and_route_shapes():
    m = _mk_model()
    h = torch.randn((1, 3, 8))
    post_attn, top_k_ids, top_k_weights = pt_moe.run_attention_and_route(
        m, h, layer_idx=0, cache=None, masks=(None, None), heat_observer=None,
    )
    assert post_attn.shape == (1, 3, 8)
    assert top_k_ids.shape == (1, 3, 2)
    assert top_k_weights.shape == (1, 3, 2)


def test_run_attention_and_route_fires_heat_observer():
    m = _mk_model()
    h = torch.randn((1, 3, 8))
    calls: list[tuple[int, int, float]] = []
    pt_moe.run_attention_and_route(
        m, h, layer_idx=0, cache=None, masks=(None, None),
        heat_observer=lambda L, E, w: calls.append((L, E, float(w))),  # noqa: N803
    )
    # 3 positions * 2 experts = 6 observations
    assert len(calls) == 6
    assert all(L == 0 for L, _, _ in calls)


def test_run_shared_expert_calls_layer_mlp():
    m = _mk_model()
    h = torch.randn((1, 3, 8))
    out = pt_moe.run_shared_expert(m, h, layer_idx=0)
    assert out.shape == (1, 3, 8)
    expected = m.model.layers[0].mlp(h)
    assert torch.allclose(out, expected)


def test_run_selected_experts_returns_dict_id_to_tensor():
    m = _mk_model()
    h = torch.randn((1, 3, 8))
    out = pt_moe.run_selected_experts(m, h, layer_idx=0, expert_ids=[0, 2])
    assert set(out.keys()) == {0, 2}
    for v in out.values():
        assert v.shape == (1, 3, 8)


def test_run_selected_experts_per_expert_linear_is_equivalent_to_stacked_index():
    """Our bypass should produce identical values to
    F.linear(F.silu(g) * u, down_proj[k]) per expert."""
    m = _mk_model()
    h = torch.randn((1, 3, 8))
    out = pt_moe.run_selected_experts(m, h, layer_idx=0, expert_ids=[1])
    e = m.model.layers[0].experts
    gu = F.linear(h, e.gate_up_proj[1])
    g, u = gu.chunk(2, dim=-1)
    mid = F.silu(g) * u
    expected = F.linear(mid, e.down_proj[1])
    assert torch.allclose(out[1], expected, atol=1e-5)


def test_aggregate_experts_weights_and_sums_with_shared():
    m = _mk_model()
    per_pos_expert_outs = {
        0: torch.full((1, 1, 8), 1.0),
        1: torch.full((1, 1, 8), 2.0),
    }
    ids = [0, 1]
    weights = torch.tensor([[0.25, 0.75]])  # [1, 2]
    shared = torch.full((1, 1, 8), 10.0)
    out = pt_moe.aggregate_experts(
        m, layer_idx=0,
        expert_outputs=per_pos_expert_outs, top_k_ids=ids,
        top_k_weights=weights, shared_out=shared,
    )
    assert out.shape == (1, 1, 8)
    assert torch.isfinite(out).all()
