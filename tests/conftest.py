"""Shared fixtures. Model loading is expensive — we share one instance across
the whole session, and only tests marked `slow` depend on it."""

from typing import Any

import pytest


@pytest.fixture(scope="session")
def loaded_model() -> Any:
    """Loads Gemma 4 26B A4B (4-bit) once per test session.

    Invoked lazily — tests that don't request it never trigger the load.
    """
    from model_shard.mlx_engine import load_model

    return load_model("mlx-community/gemma-4-26b-a4b-it-4bit")
