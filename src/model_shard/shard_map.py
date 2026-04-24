"""Static YAML-backed shard directory.

In Phase 1 this is hardcoded: a YAML file maps shard_id -> ShardSpec (network
address + layer range). Phases 2+ replace this with a gossip-driven,
dynamically-updating map behind the same lookup() / all_shards() interface.

Config format:

    model_id: "<str>"   # cluster-wide canonical model id (HF id or local path)
    shards:
      <shard_id>:
        host: <str>
        port: <int>
        start_layer: <int>
        end_layer: <int>    # half-open: [start_layer, end_layer)
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class NodeAddress:
    host: str
    port: int


@dataclass(frozen=True)
class ShardSpec:
    shard_id: str
    address: NodeAddress
    start_layer: int
    end_layer: int
    # Layer-index -> tuple of expert IDs this shard hosts for that layer.
    # Empty dict if this shard does not participate in expert-level sharding.
    moe_experts: dict[int, tuple[int, ...]] = field(default_factory=dict)

    @property
    def udp_port(self) -> int:
        """SWIM UDP port; derived as tcp_port + 1000.

        See `docs/superpowers/specs/2026-04-16-phase2-gossip-discovery-design.md`
        §7.1. If a future deployment needs an explicit field, add `swim_port`
        to the YAML schema and override this derivation.
        """
        return self.address.port + 1000


class ShardMap:
    """Cluster-wide shard directory.

    ``model_id`` is the cluster-wide canonical model identifier — either an HF
    repo id (e.g. an HF model id) or a local directory path for MLX
    conversions. Required at YAML-load time: ``from_yaml`` raises
    ``ValueError`` if the ``model_id`` field is absent or empty.

    The ``__init__`` default (``model_id=""``) is preserved for tests that
    construct ``ShardMap`` programmatically and inject an explicit
    ``backend`` into ``Node`` (bypassing the default-backend path that reads
    ``shard_map.model_id``). Production loading always goes through
    ``from_yaml`` and gets the non-empty enforcement for free.
    """

    def __init__(
        self, entries: dict[str, ShardSpec], model_id: str = ""
    ) -> None:
        self._entries = dict(entries)
        self.model_id = model_id

    def lookup(self, shard_id: str) -> ShardSpec:
        try:
            return self._entries[shard_id]
        except KeyError as e:
            raise KeyError(f"shard_id {shard_id!r} missing from shard map") from e

    def all_shards(self) -> list[str]:
        return list(self._entries.keys())

    @classmethod
    def from_yaml(cls, path: Path) -> "ShardMap":
        raw = yaml.safe_load(path.read_text())
        if not isinstance(raw, dict) or "shards" not in raw:
            raise ValueError(f"config {path} missing top-level 'shards' key")
        shards_cfg = raw["shards"]
        if not isinstance(shards_cfg, dict):
            raise ValueError(f"'shards' in {path} must be a mapping")

        model_id_raw = raw.get("model_id")
        if model_id_raw is None:
            raise ValueError(
                f"config {path}: 'model_id' top-level field is required"
            )
        if not isinstance(model_id_raw, str) or not model_id_raw:
            raise ValueError(
                f"config {path}: model_id must be a non-empty string, "
                f"got {model_id_raw!r}"
            )

        entries: dict[str, ShardSpec] = {}
        for shard_id, spec in shards_cfg.items():
            if not isinstance(spec, dict):
                raise ValueError(f"shard {shard_id!r} entry must be a mapping")
            for required_field in ("host", "port", "start_layer", "end_layer"):
                if required_field not in spec:
                    raise ValueError(
                        f"shard {shard_id!r} missing {required_field!r}"
                    )

            port_raw = spec["port"]
            if not isinstance(port_raw, int) or isinstance(port_raw, bool):
                raise ValueError(
                    f"shard {shard_id!r} has non-integer port {port_raw!r}"
                )

            start_layer = spec["start_layer"]
            end_layer = spec["end_layer"]
            if (
                not isinstance(start_layer, int)
                or not isinstance(end_layer, int)
                or isinstance(start_layer, bool)
                or isinstance(end_layer, bool)
            ):
                raise ValueError(
                    f"shard {shard_id!r} has non-integer layer range "
                    f"({start_layer!r}, {end_layer!r})"
                )
            if end_layer <= start_layer:
                raise ValueError(
                    f"shard {shard_id!r} has end_layer ({end_layer}) <= "
                    f"start_layer ({start_layer})"
                )

            moe_raw = spec.get("moe_experts", {})
            if not isinstance(moe_raw, dict):
                raise ValueError(
                    f"shard {shard_id!r} moe_experts must be a mapping, got "
                    f"{type(moe_raw).__name__}"
                )
            moe_experts: dict[int, tuple[int, ...]] = {}
            for layer_key, ids in moe_raw.items():
                if not isinstance(layer_key, int) or isinstance(layer_key, bool):
                    raise ValueError(
                        f"shard {shard_id!r} moe_experts key {layer_key!r} "
                        f"must be int"
                    )
                if not isinstance(ids, list) or not all(
                    isinstance(i, int) and not isinstance(i, bool) for i in ids
                ):
                    raise ValueError(
                        f"shard {shard_id!r} moe_experts[{layer_key}] must be "
                        f"a list of ints"
                    )
                moe_experts[layer_key] = tuple(ids)

            sid = str(shard_id)
            entries[sid] = ShardSpec(
                shard_id=sid,
                address=NodeAddress(host=str(spec["host"]), port=port_raw),
                start_layer=start_layer,
                end_layer=end_layer,
                moe_experts=moe_experts,
            )
        return cls(entries, model_id=model_id_raw)
