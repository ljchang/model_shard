"""Slow: two runs of the same prompt produce byte-identical provenance chains."""
from __future__ import annotations

import mlx.core as mx
import pytest

from model_shard.mlx_engine import load_model
from model_shard.moe import run_selected_experts
from model_shard.provenance import build_entry
from model_shard.request import OpDescriptor, OpType

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def lm(shards_model_id: str):
    return load_model(shards_model_id)


def test_compute_hash_deterministic_for_same_tensor(lm):
    """Two runs of run_selected_experts on the same input produce the same
    output tensor AND therefore the same ProvenanceEntry hash."""
    mx.random.seed(7)
    hidden = lm.text_model.layers[15].pre_feedforward_layernorm_2.weight.shape[0]
    h = mx.random.normal((1, 3, hidden)).astype(mx.bfloat16)

    out1 = run_selected_experts(lm, h, 15, [3])
    out2 = run_selected_experts(lm, h, 15, [3])
    assert mx.array_equal(out1[3], out2[3]).item()

    e1 = build_entry(
        node_id="test",
        op=OpDescriptor(op_type=OpType.OP_EXPERT, layer_idx=15, expert_id=3),
        output_tensor=out1[3],
        parent_hashes=(b"\xaa" * 32,),
    )
    e2 = build_entry(
        node_id="test",
        op=OpDescriptor(op_type=OpType.OP_EXPERT, layer_idx=15, expert_id=3),
        output_tensor=out2[3],
        parent_hashes=(b"\xaa" * 32,),
    )
    assert e1.hash == e2.hash
