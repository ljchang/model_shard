"""Tier 1 acceptance: distributed greedy tokens exactly match the reference.

Streams tokens from a 3-node decentralized pipeline (client → head, nodes
forward activations peer-to-peer, tail samples, tokens flow back to head
and out to client). Compares to the reference manifest captured by
``scripts/run_reference.py``.

If the reference manifest doesn't exist, tests skip with a hint to run the
capture script.
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
TIER1_MAX_TOKENS = 32  # subset of the 64 captured — faster, still a strong signal


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
def test_tier1_distributed_tokens_match_reference(
    three_node_pipeline: DistributedCluster,
    reference_manifest: dict[str, Any],
    prompt_idx: int,
) -> None:
    record = reference_manifest["prompts"][prompt_idx]
    prompt_tokens = list(record["prompt_tokens"])
    expected_prefix = list(record["generated_tokens"])[:TIER1_MAX_TOKENS]

    head = three_node_pipeline.shard_map.lookup("layer_0-10")
    client = Client(head_address=head.address)
    got = client.generate(prompt_tokens, max_new_tokens=TIER1_MAX_TOKENS)

    assert got == expected_prefix, (
        f"prompt {prompt_idx} ({record['text']!r}): "
        f"distributed {got[:10]}... != reference {expected_prefix[:10]}..."
    )
