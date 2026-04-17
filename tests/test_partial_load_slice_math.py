"""Fast unit tests for the pure axis-0 slice helper used by the partial loader."""

from __future__ import annotations

import numpy as np
import pytest

from model_shard.partial_load import _slice_stacked_by_axis0


def test_slice_3d_by_ids() -> None:
    arr = np.arange(128 * 4 * 5, dtype=np.int32).reshape(128, 4, 5)
    out = _slice_stacked_by_axis0(arr, [0, 3, 127])
    assert out.shape == (3, 4, 5)
    assert np.array_equal(out[0], arr[0])
    assert np.array_equal(out[1], arr[3])
    assert np.array_equal(out[2], arr[127])


def test_slice_2d_by_ids() -> None:
    arr = np.arange(128 * 7, dtype=np.int32).reshape(128, 7)
    out = _slice_stacked_by_axis0(arr, [5, 42])
    assert out.shape == (2, 7)
    assert np.array_equal(out[0], arr[5])
    assert np.array_equal(out[1], arr[42])


def test_slice_preserves_dtype() -> None:
    arr = np.zeros((128, 3), dtype=np.uint32)
    out = _slice_stacked_by_axis0(arr, [1])
    assert out.dtype == np.uint32
    assert out.shape == (1, 3)


def test_slice_empty_ids_returns_empty_axis0() -> None:
    arr = np.zeros((128, 3), dtype=np.int32)
    out = _slice_stacked_by_axis0(arr, [])
    assert out.shape == (0, 3)


def test_slice_preserves_id_order() -> None:
    """Order of returned rows follows the caller's id order, not sorted."""
    arr = np.arange(128 * 2, dtype=np.int32).reshape(128, 2)
    out = _slice_stacked_by_axis0(arr, [10, 5, 100])
    assert np.array_equal(out[0], arr[10])
    assert np.array_equal(out[1], arr[5])
    assert np.array_equal(out[2], arr[100])


def test_slice_out_of_bounds_raises() -> None:
    arr = np.zeros((128, 3), dtype=np.int32)
    with pytest.raises((IndexError, ValueError)):
        _slice_stacked_by_axis0(arr, [999])
