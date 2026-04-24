"""Phase 7-C-3b Task 6: HF id → local MLX cache resolution.

When the model_id passed to load_model is an HF repo id (e.g.
"google/gemma-4-26B-A4B-it"), the MLX backend should transparently
load from a local MLX bf16 conversion if one exists at the conventional
cache path. This lets all cluster nodes gossip the same canonical
HF id while letting MLX read locally without an HF download."""
from __future__ import annotations

from pathlib import Path

import pytest

from model_shard.mlx_engine import _resolve_local_for_mlx


def test_local_path_passes_through(tmp_path: Path) -> None:
    """If the input is already a local directory path, return it unchanged."""
    (tmp_path / "config.json").write_text("{}")
    result = _resolve_local_for_mlx(str(tmp_path))
    assert result == str(tmp_path)


def test_hf_id_resolves_to_cache_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the input is an HF id and the conventional cache directory
    exists, return the cache path instead of the HF id."""
    cache_root = tmp_path / "mlx-models"
    cache_dir = cache_root / "gemma-4-26b-a4b-it-bf16"
    cache_dir.mkdir(parents=True)
    (cache_dir / "config.json").write_text("{}")
    monkeypatch.setattr(
        "model_shard.mlx_engine._MLX_MODEL_CACHE_ROOT", cache_root,
    )
    result = _resolve_local_for_mlx("google/gemma-4-26B-A4B-it")
    assert result == str(cache_dir)


def test_hf_id_passes_through_when_cache_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the input is an HF id but no local cache exists, return the HF
    id unchanged so the caller (mlx_vlm.load) downloads from HF."""
    cache_root = tmp_path / "mlx-models"
    cache_root.mkdir(parents=True)  # exists but is empty
    monkeypatch.setattr(
        "model_shard.mlx_engine._MLX_MODEL_CACHE_ROOT", cache_root,
    )
    result = _resolve_local_for_mlx("google/gemma-4-26B-A4B-it")
    assert result == "google/gemma-4-26B-A4B-it"


def test_env_var_override_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MLX_MODEL_BF16_LOCAL_PATH env var overrides cache lookup."""
    explicit = tmp_path / "explicit-path"
    explicit.mkdir()
    (explicit / "config.json").write_text("{}")
    monkeypatch.setenv("MLX_MODEL_BF16_LOCAL_PATH", str(explicit))
    result = _resolve_local_for_mlx("google/gemma-4-26B-A4B-it")
    assert result == str(explicit)
