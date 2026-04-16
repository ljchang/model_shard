"""Static YAML-backed shard directory.

In Phase 1 this is hardcoded: a YAML file maps shard_id -> ShardSpec (network
address + layer range). Phases 2+ replace this with a gossip-driven,
dynamically-updating map behind the same lookup() / all_shards() interface.

Config format:

    shards:
      <shard_id>:
        host: <str>
        port: <int>
        start_layer: <int>
        end_layer: <int>    # half-open: [start_layer, end_layer)
"""

from dataclasses import dataclass
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

    @property
    def udp_port(self) -> int:
        """SWIM UDP port; derived as tcp_port + 1000.

        See `docs/superpowers/specs/2026-04-16-phase2-gossip-discovery-design.md`
        §7.1. If a future deployment needs an explicit field, add `swim_port`
        to the YAML schema and override this derivation.
        """
        return self.address.port + 1000


class ShardMap:
    def __init__(self, entries: dict[str, ShardSpec]) -> None:
        self._entries = dict(entries)

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

        entries: dict[str, ShardSpec] = {}
        for shard_id, spec in shards_cfg.items():
            if not isinstance(spec, dict):
                raise ValueError(f"shard {shard_id!r} entry must be a mapping")
            for field in ("host", "port", "start_layer", "end_layer"):
                if field not in spec:
                    raise ValueError(f"shard {shard_id!r} missing {field!r}")

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

            sid = str(shard_id)
            entries[sid] = ShardSpec(
                shard_id=sid,
                address=NodeAddress(host=str(spec["host"]), port=port_raw),
                start_layer=start_layer,
                end_layer=end_layer,
            )
        return cls(entries)
