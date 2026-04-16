"""Tier 2 acceptance: per-shard-boundary hidden states match the reference oracle
within a numerical tolerance.

Runs a single-shot prefill (no decode) on each canonical prompt, capturing the
hidden state emitted by each non-final shard, and compares to the reference
``hidden_states_<i>.npz`` arrays at the matching layer indices.

Tier 1 (exact-match generated tokens) is the user-visible gate; Tier 2 adds
diagnostic depth — if tokens ever drift, Tier 2 localizes which shard
introduced the drift.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from model_shard.mlx_engine import LoadedModel
from model_shard.orchestrator import Orchestrator
from model_shard.shard_map import ShardMap

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = REPO_ROOT / "artifacts" / "ref"
REFERENCE_MANIFEST = REFERENCE_DIR / "manifest.json"

# The reference captures hidden states in fp32 (cast from bf16). Distributed
# hidden states are received in bf16 on the wire, cast up to fp32 for
# comparison. The Phase-1 plan sets a 1e-3 tolerance.
TIER2_MAX_ABS_DIFF = 1e-3


@pytest.fixture(scope="module")
def reference_manifest() -> dict[str, Any]:
    if not REFERENCE_MANIFEST.exists():
        pytest.skip(
            "reference artifacts missing — run: "
            "uv run python scripts/run_reference.py "
            "--prompt-set tests/prompts.json --out-dir artifacts/ref"
        )
    return dict(json.loads(REFERENCE_MANIFEST.read_text()))


def _hidden_as_fp32_numpy(h: mx.array) -> np.ndarray:
    return np.array(h.astype(mx.float32))


@pytest.mark.slow
@pytest.mark.parametrize("prompt_idx", range(5))
def test_tier2_shard_boundary_hidden_states_match_reference(
    loaded_model: LoadedModel,
    three_node_pipeline: ShardMap,
    reference_manifest: dict[str, Any],
    prompt_idx: int,
) -> None:
    record = reference_manifest["prompts"][prompt_idx]
    prompt_tokens = list(record["prompt_tokens"])

    ref_hidden_path = REFERENCE_DIR / record["hidden_states_file"]
    ref_arrays = np.load(ref_hidden_path)

    orch = Orchestrator(
        shard_map=three_node_pipeline,
        total_layers=loaded_model.num_layers,
        hidden_size=2816,
    )
    result = orch.prefill_with_capture(prompt_tokens)

    assert result.boundary_captures, "expected at least one boundary capture"

    for cap in result.boundary_captures:
        key = f"layer_{cap.next_layer_idx}"
        assert key in ref_arrays.files, (
            f"reference missing {key} (prompt {prompt_idx}, file {ref_hidden_path})"
        )
        ref = ref_arrays[key]
        dist = _hidden_as_fp32_numpy(cap.hidden)
        assert ref.shape == dist.shape, (
            f"prompt {prompt_idx} {key}: shape {dist.shape} vs ref {ref.shape}"
        )
        max_abs = float(np.max(np.abs(ref - dist)))
        assert max_abs < TIER2_MAX_ABS_DIFF, (
            f"prompt {prompt_idx} {key}: max abs diff {max_abs:.2e} "
            f">= tolerance {TIER2_MAX_ABS_DIFF:.0e}"
        )
