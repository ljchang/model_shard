"""Bit-exact correctness proof for Phase 5b migration.

Load two sliced LoadedModels with disjoint held expert sets for layer 15.
Slice expert E from A, attach to B, assert run_selected_experts matches
the full-model baseline bit-exactly on the no-sort path."""
from __future__ import annotations

import threading

import mlx.core as mx
import pytest

from model_shard.mlx_engine import load_model, load_model_partial
from model_shard.moe import run_selected_experts
from model_shard.partial_load import attach_expert, slice_expert

pytestmark = pytest.mark.slow

_HF_ID = "mlx-community/gemma-4-26b-a4b-it-4bit"
_LAYER = 15
_MIGRATED_EXPERT = 3


@pytest.fixture(scope="module")
def lm_full():
    return load_model(_HF_ID)


@pytest.fixture(scope="module")
def lm_a():
    return load_model_partial(_HF_ID, {_LAYER: [0, 3, 6, 9]})


@pytest.fixture(scope="module")
def lm_b():
    return load_model_partial(_HF_ID, {_LAYER: [1, 4, 7, 10]})


def _synthetic_h(lm_full) -> mx.array:
    # Keep B*Seq = 1*7 = 7 — firmly on the no-sort path per 5a §7.5.
    mx.random.seed(42)
    hidden = lm_full.text_model.layers[_LAYER].pre_feedforward_layernorm_2.weight.shape[0]
    return mx.random.normal((1, 7, hidden)).astype(mx.bfloat16)


@pytest.fixture(scope="module")
def lm_b_post_migrate(lm_a, lm_b):
    """Perform the A→B migration of expert 3 once per module run.

    Both bit-exact tests depend on this so the attach happens exactly once
    regardless of test execution order."""
    lock = threading.Lock()
    tensors = slice_expert(lm_a, _LAYER, _MIGRATED_EXPERT, lock)
    attach_expert(lm_b, _LAYER, _MIGRATED_EXPERT, tensors, lock)
    return lm_b


def test_slice_from_a_equals_attach_on_b(lm_full, lm_a, lm_b_post_migrate):
    assert _MIGRATED_EXPERT in lm_b_post_migrate.held_ids_per_layer[_LAYER]

    h = _synthetic_h(lm_full)
    out_a = run_selected_experts(lm_a, h, _LAYER, [_MIGRATED_EXPERT])
    out_b = run_selected_experts(lm_b_post_migrate, h, _LAYER, [_MIGRATED_EXPERT])
    assert mx.array_equal(out_a[_MIGRATED_EXPERT], out_b[_MIGRATED_EXPERT]).item()


def test_both_equal_full_model_baseline(lm_full, lm_a, lm_b_post_migrate):
    h = _synthetic_h(lm_full)
    out_full = run_selected_experts(lm_full, h, _LAYER, [_MIGRATED_EXPERT])
    out_a = run_selected_experts(lm_a, h, _LAYER, [_MIGRATED_EXPERT])
    out_b = run_selected_experts(lm_b_post_migrate, h, _LAYER, [_MIGRATED_EXPERT])
    assert mx.array_equal(out_full[_MIGRATED_EXPERT], out_a[_MIGRATED_EXPERT]).item()
    assert mx.array_equal(out_full[_MIGRATED_EXPERT], out_b[_MIGRATED_EXPERT]).item()
