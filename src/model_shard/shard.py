"""Shard types.

A shard is a named, self-contained unit of model computation. Phase 1 only
defines LayerGroupShard (contiguous transformer layers [start, end)). Phase 3
adds ExpertShard; at that point we refactor to a shared base, not before.

Layer ranges are half-open in Python-slice style: LayerGroupShard(0, 10)
covers layers 0..9.

The MLX-backed run() method lands in Week 2 once mlx_engine.py exists.
"""


class LayerGroupShard:
    def __init__(self, start_layer: int, end_layer: int) -> None:
        if start_layer < 0:
            raise ValueError(f"start_layer must be >= 0, got {start_layer}")
        if end_layer <= start_layer:
            raise ValueError(
                f"end_layer ({end_layer}) must be > start_layer ({start_layer})"
            )
        self.start_layer = start_layer
        self.end_layer = end_layer

    @property
    def id(self) -> str:
        return f"layer_{self.start_layer}-{self.end_layer}"

    @property
    def layer_count(self) -> int:
        return self.end_layer - self.start_layer

    def contains(self, layer_idx: int) -> bool:
        return self.start_layer <= layer_idx < self.end_layer
