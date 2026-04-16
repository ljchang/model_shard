"""Smoke-level integration test for the Phase 1 distributed pipeline.

One short prompt, 5 tokens, distributed == reference. The formal 5-prompt
acceptance test lives in test_tier1_tokens.py.
"""

from __future__ import annotations

import pytest

from model_shard.mlx_engine import LoadedModel
from model_shard.orchestrator import Orchestrator
from model_shard.reference import ReferenceModel
from model_shard.shard_map import ShardMap


@pytest.mark.slow
def test_distributed_pipeline_matches_reference_short(
    loaded_model: LoadedModel, three_node_pipeline: ShardMap
) -> None:
    """3-node pipeline greedy output must match single-process reference."""
    ref = ReferenceModel(loaded_model)
    shard_map = three_node_pipeline

    prompt_tokens = ref.tokenize("The capital of France is")
    expected = ref.generate_greedy(prompt_tokens, max_new_tokens=5)

    orch = Orchestrator(
        shard_map=shard_map, total_layers=loaded_model.num_layers, hidden_size=2816
    )
    got = orch.generate_greedy(prompt_tokens, max_new_tokens=5)

    assert got == expected, f"distributed {got} != reference {expected}"
