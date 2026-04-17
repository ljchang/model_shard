"""Tier 1 acceptance under Phase 3 expert splitting of layer 15.

Streams tokens from a 3-node decentralized pipeline where layer 15's 128
experts are partitioned round-robin across the three shards (0/3/6/..,
1/4/7/.., 2/5/8/..). The orchestrator on the mid shard fans attention's
routed top-k to peer shards via the ExpertRequest RPC and aggregates; the
final tokens must match the Phase 1 reference manifest exactly.

If a token diverges, either ``ExpertOrchestrator.run_split_layer`` is not
equivalent to the atomic layer (should have been caught by Task 9's proof),
or the node isn't wiring split_layers / orchestrator into every run_layers
call (prefill AND decode).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from model_shard.client import Client
from tests.conftest import DistributedCluster

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_MANIFEST = REPO_ROOT / "artifacts" / "ref" / "manifest.json"
TIER1_MAX_TOKENS = 32


@pytest.fixture(scope="module")
def reference_manifest() -> dict[str, Any]:
    if not REFERENCE_MANIFEST.exists():
        pytest.skip(
            "reference artifacts missing — run: "
            "uv run python scripts/run_reference.py "
            "--prompt-set tests/prompts.json --out-dir artifacts/ref"
        )
    return dict(json.loads(REFERENCE_MANIFEST.read_text()))


@pytest.mark.slow
@pytest.mark.parametrize("prompt_idx", range(5))
def test_tier1_distributed_with_expert_split_layer15(
    three_node_pipeline_expert_split: DistributedCluster,
    reference_manifest: dict[str, Any],
    prompt_idx: int,
) -> None:
    record = reference_manifest["prompts"][prompt_idx]
    prompt_tokens = list(record["prompt_tokens"])
    expected_prefix = list(record["generated_tokens"])[:TIER1_MAX_TOKENS]

    head = three_node_pipeline_expert_split.shard_map.lookup("layer_0-10")
    client = Client(head_address=head.address)
    got = client.generate(prompt_tokens, max_new_tokens=TIER1_MAX_TOKENS)

    assert got == expected_prefix, (
        f"prompt {prompt_idx} ({record['text']!r}): "
        f"distributed {got[:10]}... != reference {expected_prefix[:10]}..."
    )
