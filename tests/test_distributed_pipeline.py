"""Smoke-level integration test for the Phase 1 distributed pipeline.

One short prompt, 5 tokens, distributed == reference. The formal 5-prompt
acceptance test lives in test_tier1_tokens.py.
"""

from __future__ import annotations

import pytest

from model_shard.client import Client
from model_shard.mlx_engine import LoadedModel
from model_shard.reference import ReferenceModel
from tests.conftest import DistributedCluster


@pytest.mark.slow
def test_distributed_pipeline_matches_reference_short(
    loaded_model: LoadedModel, three_node_pipeline: DistributedCluster
) -> None:
    """Peer-to-peer 3-node pipeline must produce the same greedy tokens as
    ReferenceModel. The client just connects to the head and streams tokens."""
    ref = ReferenceModel(loaded_model)
    head = three_node_pipeline.shard_map.lookup("layer_0-10")

    prompt_tokens = ref.tokenize("The capital of France is")
    expected = ref.generate_greedy(prompt_tokens, max_new_tokens=5)

    client = Client(head_address=head.address)
    got = client.generate(prompt_tokens, max_new_tokens=5)

    assert got == expected, f"distributed {got} != reference {expected}"
