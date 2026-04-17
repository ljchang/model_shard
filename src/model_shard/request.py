"""Request and ProvenanceEntry data types.

A Request is the unit of work that traverses the computation DAG. It carries
its prompt, a running token position, and an append-only provenance chain of
the shards/nodes that have touched it. Phase 6-B populates the ``hash``,
``parent_hashes``, and ``op`` fields so the chain forms a verifiable DAG
matching Gemma's computation graph.
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass, field
from enum import IntEnum


class OpType(IntEnum):
    """Operation taxonomy for Phase 6-B provenance entries.

    Int values match the ``OpType`` protobuf enum in ``wire.proto`` (Task 1)."""

    OP_UNSPECIFIED     = 0
    OP_EMBED           = 1
    OP_LAYER_ATOMIC    = 2
    OP_ATTENTION_ROUTE = 3
    OP_EXPERT          = 4
    OP_AGGREGATE       = 5
    OP_FINALIZE        = 6
    OP_SHARED_EXPERT   = 7


@dataclass(frozen=True)
class OpDescriptor:
    """Structured description of the operation a ProvenanceEntry records.

    ``pack()`` produces a deterministic 9-byte representation used as input
    to the BLAKE2b hash (see ``provenance.compute_hash``). Layout:
    ``uint8 op_type || uint32 layer_idx (LE) || uint32 expert_id (LE)``.
    """

    op_type: OpType
    layer_idx: int = 0
    expert_id: int = 0

    def pack(self) -> bytes:
        return struct.pack("<BII", int(self.op_type), self.layer_idx, self.expert_id)


@dataclass(frozen=True)
class ProvenanceEntry:
    """One node's claim about one operation in a forward pass.

    Phase 1 shape (``shard_id``, ``node_id``, ``timestamp``, ``hash``) is
    preserved so existing callers still work. Phase 6-B adds
    ``parent_hashes`` (for DAG parents) and ``op`` (for the operation type
    and indices). Both default to empty so Phase 1 tests need no change.
    """

    shard_id: str
    node_id: str
    timestamp: float
    hash: bytes = b""
    parent_hashes: tuple[bytes, ...] = ()
    op: OpDescriptor | None = None


@dataclass
class Request:
    request_id: str
    sequence_id: str
    prompt_token_ids: list[int]
    position: int = 0
    provenance: list[ProvenanceEntry] = field(default_factory=list)

    def append_provenance(
        self,
        *,
        shard_id: str,
        node_id: str,
        hash: bytes = b"",
        parent_hashes: tuple[bytes, ...] = (),
        op: OpDescriptor | None = None,
    ) -> None:
        self.provenance.append(
            ProvenanceEntry(
                shard_id=shard_id,
                node_id=node_id,
                timestamp=time.time(),
                hash=hash,
                parent_hashes=parent_hashes,
                op=op,
            )
        )


__all__ = ["OpDescriptor", "OpType", "ProvenanceEntry", "Request"]
