"""Static YAML-backed shard directory.

In Phase 1 this is hardcoded: a YAML file maps shard_id -> (host, port). Phases
2+ will replace this with a gossip-driven, dynamically-updating map behind the
same lookup() / all_shards() interface.

Config format:

    shards:
      <shard_id>:
        host: <str>
        port: <int>
"""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class NodeAddress:
    host: str
    port: int


class ShardMap:
    def __init__(self, entries: dict[str, NodeAddress]) -> None:
        self._entries = dict(entries)

    def lookup(self, shard_id: str) -> NodeAddress:
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

        entries: dict[str, NodeAddress] = {}
        for shard_id, spec in shards_cfg.items():
            if not isinstance(spec, dict):
                raise ValueError(f"shard {shard_id!r} entry must be a mapping")
            if "host" not in spec:
                raise ValueError(f"shard {shard_id!r} missing 'host'")
            if "port" not in spec:
                raise ValueError(f"shard {shard_id!r} missing 'port'")
            port_raw = spec["port"]
            if not isinstance(port_raw, int) or isinstance(port_raw, bool):
                raise ValueError(
                    f"shard {shard_id!r} has non-integer port {port_raw!r}"
                )
            entries[str(shard_id)] = NodeAddress(host=str(spec["host"]), port=port_raw)
        return cls(entries)
