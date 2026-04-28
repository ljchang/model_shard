"""Phase 7-B Task 1: pyproject.toml pytorch optional-deps + cuda marker + _COMPUTE_LOCK existence."""
from __future__ import annotations

import tomllib
from pathlib import Path


def _pyproject() -> dict:
    with open(Path(__file__).parent.parent / "pyproject.toml", "rb") as f:
        return tomllib.load(f)


def test_pyproject_has_pytorch_optional_group():
    data = _pyproject()
    optional = data.get("project", {}).get("optional-dependencies", {})
    assert "pytorch" in optional
    group = optional["pytorch"]
    names = {dep.split(">=")[0].split("==")[0].strip() for dep in group}
    assert "torch" in names
    assert "transformers" in names
    assert "accelerate" in names


def test_pyproject_has_cuda_pytest_marker():
    data = _pyproject()
    markers = data.get("tool", {}).get("pytest", {}).get("ini_options", {}).get("markers", [])
    assert any(m.startswith("cuda:") or m == "cuda" for m in markers), (
        f"cuda marker not declared in [tool.pytest.ini_options] markers list: {markers}"
    )


def test_compute_lock_exists():
    """_COMPUTE_LOCK is the canonical backend-neutral compute lock name.
    The Phase 7-B _MLX_COMPUTE_LOCK alias was retired in Phase 7-C-4."""
    from model_shard import node
    assert hasattr(node, "_COMPUTE_LOCK"), "_COMPUTE_LOCK must exist"
    assert not hasattr(node, "_MLX_COMPUTE_LOCK"), (
        "_MLX_COMPUTE_LOCK alias was retired in Phase 7-C-4"
    )
