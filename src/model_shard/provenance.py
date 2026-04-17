"""Phase 6-B provenance hashing + entry construction + pb adapters.

Pure module: no threading, no MLX evaluation side-effects beyond
byte serialization via mlx_engine.tensor_to_bytes.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterable

import mlx.core as mx

from model_shard._pb import wire_pb2
from model_shard.mlx_engine import tensor_to_bytes
from model_shard.request import OpDescriptor, OpType, ProvenanceEntry


def compute_hash(
    *,
    parent_hashes: tuple[bytes, ...] | Iterable[bytes],
    node_id: str,
    op: OpDescriptor,
    output_bytes: bytes,
) -> bytes:
    """BLAKE2b-256 over (concat(parents) || node_id utf-8 || op.pack() || output_bytes).

    Input tensor bytes are elided: ``parent_hashes`` already transitively
    commit to the input of this op (the prev op's output IS this op's input
    for linear ops; for OP_AGGREGATE, all expert/shared hashes together
    commit to all inputs)."""
    h = hashlib.blake2b(digest_size=32)
    for parent in parent_hashes:
        h.update(parent)
    h.update(node_id.encode("utf-8"))
    h.update(op.pack())
    h.update(output_bytes)
    return h.digest()


def build_entry(
    *,
    node_id: str,
    op: OpDescriptor,
    output_tensor: mx.array,
    parent_hashes: tuple[bytes, ...] | Iterable[bytes],
) -> ProvenanceEntry:
    """Construct a ProvenanceEntry by serializing ``output_tensor`` and
    computing the BLAKE2b digest. ``shard_id`` is set equal to ``node_id``
    (Phase 6-B: the two are the same; retained as separate fields for Phase 1
    compat)."""
    parents_tuple = tuple(parent_hashes)
    output_bytes = tensor_to_bytes(output_tensor)
    digest = compute_hash(
        parent_hashes=parents_tuple,
        node_id=node_id,
        op=op,
        output_bytes=output_bytes,
    )
    return ProvenanceEntry(
        shard_id=node_id,
        node_id=node_id,
        timestamp=time.time(),
        hash=digest,
        parent_hashes=parents_tuple,
        op=op,
    )


def entry_to_pb(entry: ProvenanceEntry) -> wire_pb2.ProvenanceEntryPb:
    pb = wire_pb2.ProvenanceEntryPb(
        hash=entry.hash,
        node_id=entry.node_id,
        timestamp=entry.timestamp,
    )
    pb.parent_hashes.extend(entry.parent_hashes)
    if entry.op is not None:
        pb.op.op_type = int(entry.op.op_type)
        pb.op.layer_idx = entry.op.layer_idx
        pb.op.expert_id = entry.op.expert_id
    return pb


def entry_from_pb(pb: wire_pb2.ProvenanceEntryPb) -> ProvenanceEntry:
    op: OpDescriptor | None = None
    if pb.HasField("op"):
        op = OpDescriptor(
            op_type=OpType(int(pb.op.op_type)),
            layer_idx=int(pb.op.layer_idx),
            expert_id=int(pb.op.expert_id),
        )
    return ProvenanceEntry(
        shard_id=pb.node_id,
        node_id=pb.node_id,
        timestamp=float(pb.timestamp),
        hash=bytes(pb.hash),
        parent_hashes=tuple(bytes(p) for p in pb.parent_hashes),
        op=op,
    )


class ProvenanceError(ValueError):
    """Raised by validate_chain on any rule violation. Callers convert this
    into Error{ERR_INVALID_PROVENANCE, is_final=true} for the client."""


__all__ = [
    "ProvenanceError",
    "build_entry",
    "compute_hash",
    "entry_from_pb",
    "entry_to_pb",
]
