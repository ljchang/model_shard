"""Request and ProvenanceEntry data types.

A Request is the unit of work that traverses the computation DAG. It carries
its prompt, a running token position, and an append-only provenance chain of
the shards/nodes that have touched it. The `hash` field on each ProvenanceEntry
is unused in Phase 1 but exists for Phase 6 verification.
"""

import time
from dataclasses import dataclass, field


@dataclass
class ProvenanceEntry:
    shard_id: str
    node_id: str
    timestamp: float
    hash: bytes = b""


@dataclass
class Request:
    request_id: str
    sequence_id: str
    prompt_token_ids: list[int]
    position: int = 0
    provenance: list[ProvenanceEntry] = field(default_factory=list)

    def append_provenance(self, *, shard_id: str, node_id: str, hash: bytes = b"") -> None:
        self.provenance.append(
            ProvenanceEntry(
                shard_id=shard_id,
                node_id=node_id,
                timestamp=time.time(),
                hash=hash,
            )
        )
