"""Unit tests for slice_expert / attach_expert using synthetic LoadedModel."""
from __future__ import annotations

import threading
import types
from typing import Any

import mlx.core as mx
import pytest

from model_shard.mlx_engine import LoadedModel
from model_shard.partial_load import attach_expert, slice_expert


def _make_fake_lm(num_experts: int, held: list[int]) -> LoadedModel:
    """Build a LoadedModel shell whose text_model.layers[0].experts.switch_glu
    has synthetic (num_experts, 4, 4) tensors for the 9 proj/attr slots.
    Only layer 0 is wired; enough for slice_expert / attach_expert tests."""
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
        mlx_model=mlx_model,   # type: ignore[arg-type]
        language_model=language_model,  # type: ignore[arg-type]
        text_model=text_model,  # type: ignore[arg-type]
        processor=None,  # type: ignore[arg-type]
        num_layers=1,
        held_ids_per_layer={0: tuple(held)} if held else {},
    )


def test_slice_expert_returns_nine_tensors_at_local_slot():
    lock = threading.Lock()
    # Held = [0, 3, 6, 9]; local slot of global id 6 is 2.
    lm = _make_fake_lm(num_experts=4, held=[0, 3, 6, 9])
    tensors = slice_expert(lm, layer_idx=0, expert_id=6, mlx_lock=lock)
    assert len(tensors) == 9
    # Verify we sliced along axis 0 at local slot 2.
    sg = lm.text_model.layers[0].experts.switch_glu
    for i, (proj_name, attr) in enumerate([
        ("gate_proj", "weight"), ("gate_proj", "scales"), ("gate_proj", "biases"),
        ("up_proj",   "weight"), ("up_proj",   "scales"), ("up_proj",   "biases"),
        ("down_proj", "weight"), ("down_proj", "scales"), ("down_proj", "biases"),
    ]):
        expected = getattr(getattr(sg, proj_name), attr)[2]
        assert mx.array_equal(tensors[i], expected).item()


def test_slice_expert_raises_when_not_held():
    lock = threading.Lock()
    lm = _make_fake_lm(num_experts=4, held=[0, 3, 6, 9])
    with pytest.raises(KeyError):
        slice_expert(lm, layer_idx=0, expert_id=42, mlx_lock=lock)


def test_attach_expert_grows_stack_by_one():
    lock = threading.Lock()
    lm = _make_fake_lm(num_experts=4, held=[0, 3, 6, 9])
    # Build 9 synthetic tensors with identifiable values.
    # Shapes match _make_fake_lm: weight=(4,4), scales=(4,8), biases=(4,8) per proj.
    new_tensors = [mx.full((4, 4), 0.0)]   # gate_proj weight
    new_tensors += [mx.full((4, 8), float(i)) for i in range(1, 3)]  # gate_proj scales, biases
    new_tensors += [mx.full((4, 4), 3.0)]   # up_proj weight
    new_tensors += [mx.full((4, 8), float(i)) for i in range(4, 6)]  # up_proj scales, biases
    new_tensors += [mx.full((4, 4), 6.0)]   # down_proj weight
    new_tensors += [mx.full((4, 8), float(i)) for i in range(7, 9)]  # down_proj scales, biases
    attach_expert(lm, layer_idx=0, expert_id=42, tensors=new_tensors, mlx_lock=lock)
    sg = lm.text_model.layers[0].experts.switch_glu
    assert sg.gate_proj.weight.shape[0] == 5
    assert lm.held_ids_per_layer[0] == (0, 3, 6, 9, 42)
    # New tensor landed at the new tail row.
    assert mx.array_equal(sg.gate_proj.weight[4], new_tensors[0]).item()


def test_attach_expert_raises_on_duplicate():
    lock = threading.Lock()
    lm = _make_fake_lm(num_experts=4, held=[0, 3, 6, 9])
    # 9 tensors with correct shapes for _make_fake_lm: weight=(4,4), scales/biases=(4,8)
    dummy = (
        [mx.zeros((4, 4))]
        + [mx.zeros((4, 8)) for _ in range(2)]
        + [mx.zeros((4, 4))]
        + [mx.zeros((4, 8)) for _ in range(2)]
        + [mx.zeros((4, 4))]
        + [mx.zeros((4, 8)) for _ in range(2)]
    )
    with pytest.raises(ValueError):
        attach_expert(lm, layer_idx=0, expert_id=3, tensors=dummy, mlx_lock=lock)


def test_attach_expert_requires_nine_tensors():
    lock = threading.Lock()
    lm = _make_fake_lm(num_experts=4, held=[0, 3, 6, 9])
    with pytest.raises(ValueError):
        attach_expert(lm, layer_idx=0, expert_id=42, tensors=[mx.zeros((4, 4))], mlx_lock=lock)


def test_attach_then_slice_roundtrips():
    lock = threading.Lock()
    lm = _make_fake_lm(num_experts=4, held=[0, 3, 6, 9])
    # Shapes match _make_fake_lm: weight=(4,4), scales=(4,8), biases=(4,8) per proj.
    sentinels = (
        [mx.full((4, 4), 10.0)]
        + [mx.full((4, 8), 10.0 + i) for i in range(1, 3)]
        + [mx.full((4, 4), 20.0)]
        + [mx.full((4, 8), 20.0 + i) for i in range(1, 3)]
        + [mx.full((4, 4), 30.0)]
        + [mx.full((4, 8), 30.0 + i) for i in range(1, 3)]
    )
    attach_expert(lm, layer_idx=0, expert_id=42, tensors=sentinels, mlx_lock=lock)
    sliced = slice_expert(lm, layer_idx=0, expert_id=42, mlx_lock=lock)
    for a, b in zip(sentinels, sliced):
        assert mx.array_equal(a, b).item()
