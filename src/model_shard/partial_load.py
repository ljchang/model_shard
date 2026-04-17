"""Partial expert-weight loading for Phase 5a.

A shard can declare which routed experts it holds per layer (via
ShardSpec.moe_experts). This module provides a custom safetensors reader
that slices the stacked (128, out, in) expert projection tensors at load
time so the shard's resident memory contains only the held experts'
weights.

Chassis weights (attention, dense mlp, norms, embeddings, LM head, router)
load unchanged on every node.
"""

from __future__ import annotations

import numpy as np


def _slice_stacked_by_axis0(
    arr: np.ndarray, ids: list[int]
) -> np.ndarray:
    """Return the rows of `arr` at positions `ids` along axis 0.

    Order is preserved: the returned array's row `i` is `arr[ids[i]]`.
    Raises IndexError or ValueError if any id is out of bounds.
    """
    if not ids:
        return arr[0:0]
    return arr[ids]


__all__ = ["_slice_stacked_by_axis0"]
