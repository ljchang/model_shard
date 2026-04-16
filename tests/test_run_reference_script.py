"""End-to-end smoke test for scripts/run_reference.py.

Runs the script on a tiny 1-prompt set with a short generation length and
verifies the manifest + hidden-state artifact are shaped correctly. The real
5-prompt capture happens outside the test suite.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_reference.py"


@pytest.mark.slow
def test_script_produces_manifest_and_hidden_states(tmp_path: Path) -> None:
    prompt_set = tmp_path / "tiny_prompts.json"
    prompt_set.write_text(json.dumps({"prompts": ["Hello"]}))
    out_dir = tmp_path / "out"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--prompt-set",
            str(prompt_set),
            "--out-dir",
            str(out_dir),
            "--max-new-tokens",
            "3",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["num_layers"] == 30
    assert manifest["captured_hidden_states"] is True
    assert len(manifest["prompts"]) == 1

    record = manifest["prompts"][0]
    assert record["text"] == "Hello"
    assert len(record["generated_tokens"]) == 3

    hs_path = out_dir / record["hidden_states_file"]
    assert hs_path.exists()
    arrays = np.load(hs_path)
    # 30 per-layer snapshots + final_hidden + logits.
    assert f"layer_{manifest['num_layers'] - 1}" in arrays.files
    assert "final_hidden" in arrays.files
    assert "logits" in arrays.files
