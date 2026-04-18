"""Unit tests for detach_expert (inverse of Phase 5b attach_expert)."""
from __future__ import annotations

import threading
import types

import mlx.core as mx
import pytest

from model_shard.mlx_engine import LoadedModel
from model_shard.partial_load import attach_expert, detach_expert


def _make_fake_lm(num_experts: int, held: list[int]) -> LoadedModel:
    def _stack(stride: int) -> mx.array:
        vals = mx.arange(num_experts * 4 * 4 * stride, dtype=mx.float32)
        return vals.reshape((num_experts, 4, 4 * stride))
    projs = {
        name: types.SimpleNamespace(
            weight=_stack(1), scales=_stack(2), biases=_stack(2),
        )
        for name in ("gate_proj", "up_proj", "down_proj")
    }
    switch_glu = types.SimpleNamespace(**projs)
    experts = types.SimpleNamespace(switch_glu=switch_glu)
    layer = types.SimpleNamespace(experts=experts)
    text_model = types.SimpleNamespace(layers=[layer])
    language_model = types.SimpleNamespace(model=text_model)
    mlx_model = types.SimpleNamespace(language_model=language_model)
    return LoadedModel(
        mlx_model=mlx_model,
        language_model=language_model,
        text_model=text_model,
        processor=None,
        num_layers=1,
        held_ids_per_layer={0: tuple(held)} if held else {},
    )


def test_detach_expert_shrinks_stack_by_one():
    lock = threading.Lock()
    lm = _make_fake_lm(num_experts=4, held=[0, 3, 6, 9])
    detach_expert(lm, layer_idx=0, expert_id=6, mlx_lock=lock)
    sg = lm.text_model.layers[0].experts.switch_glu
    assert sg.gate_proj.weight.shape[0] == 3
    assert lm.held_ids_per_layer[0] == (0, 3, 9)


def test_detach_expert_preserves_other_rows_bit_exactly():
    lock = threading.Lock()
    lm = _make_fake_lm(num_experts=4, held=[0, 3, 6, 9])
    sg_before = lm.text_model.layers[0].experts.switch_glu
    expected_rows = {
        (proj, attr, local_slot): getattr(getattr(sg_before, proj), attr)[local_slot]
        for proj in ("gate_proj", "up_proj", "down_proj")
        for attr in ("weight", "scales", "biases")
        for local_slot in (0, 1, 3)
    }
    detach_expert(lm, layer_idx=0, expert_id=6, mlx_lock=lock)
    sg_after = lm.text_model.layers[0].experts.switch_glu
    old_to_new = {0: 0, 1: 1, 3: 2}
    for (proj, attr, old_slot), expected in expected_rows.items():
        new_slot = old_to_new[old_slot]
        actual = getattr(getattr(sg_after, proj), attr)[new_slot]
        assert mx.array_equal(actual, expected).item(), (
            f"{proj}.{attr} row old_slot={old_slot} did not survive intact"
        )


def test_attach_detach_roundtrip_is_identity():
    lock = threading.Lock()
    lm = _make_fake_lm(num_experts=4, held=[0, 3, 6, 9])
    sg = lm.text_model.layers[0].experts.switch_glu
    before = {
        (proj, attr): mx.array(getattr(getattr(sg, proj), attr))
        for proj in ("gate_proj", "up_proj", "down_proj")
        for attr in ("weight", "scales", "biases")
    }
    # Shapes must match per-expert slice of the stacked tensors:
    #   gate_proj.weight  (4, 4),  gate_proj.scales  (4, 8),  gate_proj.biases  (4, 8)
    #   up_proj.weight    (4, 4),  up_proj.scales    (4, 8),  up_proj.biases    (4, 8)
    #   down_proj.weight  (4, 4),  down_proj.scales  (4, 8),  down_proj.biases  (4, 8)
    new_tensors = [
        mx.full((4, 4), float(10)),   # gate_proj.weight
        mx.full((4, 8), float(11)),   # gate_proj.scales
        mx.full((4, 8), float(12)),   # gate_proj.biases
        mx.full((4, 4), float(20)),   # up_proj.weight
        mx.full((4, 8), float(21)),   # up_proj.scales
        mx.full((4, 8), float(22)),   # up_proj.biases
        mx.full((4, 4), float(30)),   # down_proj.weight
        mx.full((4, 8), float(31)),   # down_proj.scales
        mx.full((4, 8), float(32)),   # down_proj.biases
    ]
    attach_expert(lm, layer_idx=0, expert_id=42, tensors=new_tensors, mlx_lock=lock)
    assert lm.held_ids_per_layer[0] == (0, 3, 6, 9, 42)
    detach_expert(lm, layer_idx=0, expert_id=42, mlx_lock=lock)
    assert lm.held_ids_per_layer[0] == (0, 3, 6, 9)
    sg_after = lm.text_model.layers[0].experts.switch_glu
    for (proj, attr), expected in before.items():
        actual = getattr(getattr(sg_after, proj), attr)
        assert mx.array_equal(actual, expected).item(), (
            f"{proj}.{attr} changed after attach→detach roundtrip"
        )


def test_detach_expert_raises_on_not_held():
    lock = threading.Lock()
    lm = _make_fake_lm(num_experts=4, held=[0, 3, 6, 9])
    with pytest.raises(KeyError, match="not held"):
        detach_expert(lm, layer_idx=0, expert_id=42, mlx_lock=lock)


def test_detach_expert_raises_on_unknown_layer():
    lock = threading.Lock()
    lm = _make_fake_lm(num_experts=4, held=[0, 3, 6, 9])
    with pytest.raises(KeyError):
        detach_expert(lm, layer_idx=99, expert_id=0, mlx_lock=lock)
