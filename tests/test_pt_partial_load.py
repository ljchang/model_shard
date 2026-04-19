"""Phase 7-B Task 4: pt_partial_load — slice / attach / detach expert.

Synthetic model with stacked gate_up_proj / down_proj tensors (same shape
as HF Gemma4TextExperts but smaller).
"""
from __future__ import annotations

import threading

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from model_shard import pt_partial_load  # noqa: E402


class _SynthExperts(nn.Module):
    def __init__(self, num_experts: int = 4, hidden: int = 4, inter: int = 8):
        super().__init__()
        self.num_experts = num_experts
        self.gate_up_proj = nn.Parameter(
            torch.arange(num_experts * 2 * inter * hidden, dtype=torch.bfloat16)
            .reshape(num_experts, 2 * inter, hidden)
        )
        self.down_proj = nn.Parameter(
            torch.arange(num_experts * hidden * inter, dtype=torch.bfloat16)
            .reshape(num_experts, hidden, inter)
        )


class _SynthDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = _SynthExperts()


class _SynthTextModel(nn.Module):
    def __init__(self, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([_SynthDecoderLayer() for _ in range(num_layers)])


class _SynthModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _SynthTextModel()


def _mk() -> _SynthModel:
    return _SynthModel()


def test_slice_expert_returns_gate_up_and_down_tensors():
    m = _mk()
    lock = threading.Lock()
    tensors = pt_partial_load.slice_expert(m, layer_idx=0, expert_id=2, lock=lock)
    assert len(tensors) == 2
    gate_up, down = tensors
    assert gate_up.shape == (2 * 8, 4)
    assert down.shape == (4, 8)


def test_slice_expert_returns_cpu_detached_copies():
    m = _mk()
    lock = threading.Lock()
    gate_up, down = pt_partial_load.slice_expert(m, 0, 1, lock)
    assert gate_up.device.type == "cpu"
    assert down.device.type == "cpu"
    assert not gate_up.requires_grad
    before = m.model.layers[0].experts.gate_up_proj[1].clone()
    gate_up.fill_(0)
    after = m.model.layers[0].experts.gate_up_proj[1]
    assert torch.equal(before.cpu(), after.cpu())


def test_attach_expert_writes_values_in_place():
    m = _mk()
    lock = threading.Lock()
    new_gate_up = torch.full((2 * 8, 4), 42.0, dtype=torch.bfloat16)
    new_down = torch.full((4, 8), 42.0, dtype=torch.bfloat16)
    pt_partial_load.attach_expert(m, 0, 3, [new_gate_up, new_down], lock)
    assert torch.equal(
        m.model.layers[0].experts.gate_up_proj[3].cpu(),
        new_gate_up.cpu(),
    )
    assert torch.equal(
        m.model.layers[0].experts.down_proj[3].cpu(),
        new_down.cpu(),
    )


def test_detach_expert_zeroes_slots():
    m = _mk()
    lock = threading.Lock()
    pt_partial_load.detach_expert(m, 0, 2, lock)
    assert torch.all(m.model.layers[0].experts.gate_up_proj[2] == 0)
    assert torch.all(m.model.layers[0].experts.down_proj[2] == 0)


def test_attach_expert_rejects_wrong_shape():
    m = _mk()
    lock = threading.Lock()
    bad_gate_up = torch.zeros((3, 4), dtype=torch.bfloat16)  # wrong shape
    bad_down = torch.zeros((4, 8), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="gate_up"):
        pt_partial_load.attach_expert(m, 0, 0, [bad_gate_up, bad_down], lock)


def test_slice_attach_roundtrip_preserves_values():
    m = _mk()
    lock = threading.Lock()
    original = [
        m.model.layers[0].experts.gate_up_proj[1].clone().cpu(),
        m.model.layers[0].experts.down_proj[1].clone().cpu(),
    ]
    sliced = pt_partial_load.slice_expert(m, 0, 1, lock)
    pt_partial_load.detach_expert(m, 0, 1, lock)
    assert torch.all(m.model.layers[0].experts.gate_up_proj[1] == 0)
    pt_partial_load.attach_expert(m, 0, 1, sliced, lock)
    assert torch.equal(
        m.model.layers[0].experts.gate_up_proj[1].cpu(), original[0]
    )
    assert torch.equal(
        m.model.layers[0].experts.down_proj[1].cpu(), original[1]
    )
