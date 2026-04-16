"""Tier 1 acceptance: distributed greedy tokens exactly match the single-process reference.

Runs the orchestrator against a 3-node pipeline on all 5 canonical prompts
from ``tests/prompts.json``. Compares the generated token sequences against
the reference manifest captured by ``scripts/run_reference.py``.

If the reference manifest doesn't exist, tests skip with a hint to run the
capture script.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from model_shard.mlx_engine import LoadedModel
from model_shard.orchestrator import Orchestrator
from model_shard.shard_map import ShardMap

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
    loaded_model: LoadedModel,
    three_node_pipeline: ShardMap,
    reference_manifest: dict[str, Any],
    prompt_idx: int,
) -> None:
    record = reference_manifest["prompts"][prompt_idx]
    prompt_tokens = list(record["prompt_tokens"])
    expected_prefix = list(record["generated_tokens"])[:TIER1_MAX_TOKENS]

    orch = Orchestrator(
        shard_map=three_node_pipeline,
        total_layers=loaded_model.num_layers,
        hidden_size=2816,
    )
    got = orch.generate_greedy(prompt_tokens, max_new_tokens=TIER1_MAX_TOKENS)

    assert got == expected_prefix, (
        f"prompt {prompt_idx} ({record['text']!r}): "
        f"distributed {got[:10]}... != reference {expected_prefix[:10]}..."
    )
