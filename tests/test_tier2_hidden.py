"""Tier 2 acceptance: per-shard-boundary hidden states match the reference
oracle within a numerical tolerance.

Runs a single prefill (max_new_tokens=1) via the Client, then reads the
debug-captured hidden states from each non-tail node in the in-process
cluster. Compares them to the reference ``hidden_states_<i>.npz`` arrays
at the matching layer indices.

The ``Node.debug_captures_for`` attribute is an in-process-only test hook;
production clients have no such access.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from model_shard.client import Client
from tests.conftest import DistributedCluster

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = REPO_ROOT / "artifacts" / "ref"
REFERENCE_MANIFEST = REFERENCE_DIR / "manifest.json"

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
    three_node_pipeline: DistributedCluster,
    reference_manifest: dict[str, Any],
    prompt_idx: int,
) -> None:
    record = reference_manifest["prompts"][prompt_idx]
    prompt_tokens = list(record["prompt_tokens"])

    ref_arrays = np.load(REFERENCE_DIR / record["hidden_states_file"])

    # Clear any captures from previous test cases so we read only this prompt's.
    for node in three_node_pipeline.nodes_by_id.values():
        node.clear_debug_captures()

    head = three_node_pipeline.shard_map.lookup("layer_0-10")
    client = Client(head_address=head.address)

    # Single-shot prefill: generate exactly 1 token. During that call, every
    # non-tail node records the hidden state it forwards to its downstream
    # peer. After the call returns, we read those captures.
    tokens = client.generate(prompt_tokens, max_new_tokens=1)
    assert len(tokens) == 1

    # Collect (next_layer_idx, hidden) tuples from each non-tail node. The
    # first capture per non-tail node is from the prefill forward pass.
    captures: list[tuple[int, mx.array]] = []
    for node in three_node_pipeline.nodes_by_id.values():
        if not node.is_tail:
            for req_captures in node._debug_captures.values():
                if req_captures:
                    captures.append(req_captures[0])

    captures.sort(key=lambda c: c[0])

    assert captures, "expected at least one boundary capture from non-tail nodes"

    for next_layer, hidden in captures:
        key = f"layer_{next_layer}"
        assert key in ref_arrays.files, (
            f"reference missing {key} (prompt {prompt_idx})"
        )
        ref = ref_arrays[key]
        dist = _hidden_as_fp32_numpy(hidden)
        assert ref.shape == dist.shape, (
            f"prompt {prompt_idx} {key}: shape {dist.shape} vs ref {ref.shape}"
        )
        max_abs = float(np.max(np.abs(ref - dist)))
        assert max_abs < TIER2_MAX_ABS_DIFF, (
            f"prompt {prompt_idx} {key}: max abs diff {max_abs:.2e} "
            f">= tolerance {TIER2_MAX_ABS_DIFF:.0e}"
        )
