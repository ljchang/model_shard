"""Phase 7-C-3a Task 6: bf16 single-process memory smoke test.

Runs FIRST in the slow regression cascade — confirms the bf16 conversion
fits comfortably in M5 unified memory before the long fixture regenerations.
If this fails, Tasks 8-11 will also fail; better to know now."""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
psutil = pytest.importorskip("psutil")

from model_shard.mlx_engine import load_model  # noqa: E402

_MAX_RESIDENT_BYTES = 80 * 1024**3  # 80 GB ceiling


@pytest.mark.slow
def test_bf16_single_process_fits_in_80gb(shards_model_id: str) -> None:
    """Load the canonical model in this process and assert RSS < 80 GB.

    bf16 Gemma 4 26B is ~54 GB on disk; with overhead and any working
    set, 80 GB is a generous ceiling that catches accidental
    in-place-mutation bloat or full-precision-promotion bugs."""
    proc = psutil.Process()
    rss_before = proc.memory_info().rss
    lm = load_model(shards_model_id)
    rss_after = proc.memory_info().rss
    delta = rss_after - rss_before
    assert lm.num_layers == 30, f"unexpected num_layers={lm.num_layers}"
    assert rss_after < _MAX_RESIDENT_BYTES, (
        f"bf16 load grew RSS to {rss_after / 1024**3:.1f} GB "
        f"(delta {delta / 1024**3:.1f} GB), above the {_MAX_RESIDENT_BYTES / 1024**3:.0f} GB ceiling. "
        "Either the bf16 model is bigger than expected or something is "
        "leaking; investigate before regenerating fixtures."
    )
