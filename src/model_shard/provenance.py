"""Phase 6-B provenance hashing + entry construction + pb adapters.

Pure module: no threading, no MLX evaluation side-effects beyond
byte serialization via mlx_engine.tensor_to_bytes.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Iterable

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


def validate_chain(
    chain: list[ProvenanceEntry],
    *,
    shard_lookup: Callable[[str], tuple[int, int]],
    total_layers: int,
    split_layers_for_shard: Callable[[str], set[int]],
    live_owners_of: Callable[[int, int], set[str]],
    tail_tensor_bytes: bytes | None,
) -> None:
    """Enforce D8 rules 1-5 from the Phase 6-B spec. Raises ``ProvenanceError``
    with a descriptive message on the first violation.

    Parameters
    ----------
    chain
        The full chain to validate.
    shard_lookup
        ``shard_id -> (start_layer, end_layer)``.
    total_layers
        30 for Gemma 4 26B A4B.
    split_layers_for_shard
        ``shard_id -> set of layer indices that are split on that shard``.
    live_owners_of
        ``(layer_idx, expert_id) -> set[str]`` of live authorized owners.
        In production, bound to Phase 5b's ``Node.owners_of``.
    tail_tensor_bytes
        If provided, rule 4 (hash tail check) is run.
    """
    if not chain:
        raise ProvenanceError("empty chain")

    # Rule 1: starts with OP_EMBED, OP_FINALIZE iff last.
    first = chain[0]
    if first.op is None or first.op.op_type != OpType.OP_EMBED:
        raise ProvenanceError("chain must begin with OP_EMBED")
    for i, e in enumerate(chain):
        if e.op is not None and e.op.op_type == OpType.OP_FINALIZE and i != len(chain) - 1:
            raise ProvenanceError("OP_FINALIZE must be the last entry if present")

    # Rule 2: layer completeness.
    layers_covered: set[int] = set()
    for e in chain:
        if e.op is None:
            continue
        if e.op.op_type == OpType.OP_LAYER_ATOMIC:
            layers_covered.add(e.op.layer_idx)
        if e.op.op_type == OpType.OP_AGGREGATE:
            layers_covered.add(e.op.layer_idx)
    has_finalize = any(
        e.op is not None and e.op.op_type == OpType.OP_FINALIZE for e in chain
    )
    if has_finalize:
        for layer in range(total_layers):
            if layer not in layers_covered:
                raise ProvenanceError(f"chain missing layer {layer}")
    elif layers_covered:
        highest = max(layers_covered)
        for layer in range(highest + 1):
            if layer not in layers_covered:
                raise ProvenanceError(f"chain missing layer {layer}")

    # Rule 3: split-layer DAG shape.
    split_ops_by_layer: dict[int, dict[str, list[ProvenanceEntry]]] = {}
    for e in chain:
        if e.op is None:
            continue
        if e.op.op_type in (
            OpType.OP_ATTENTION_ROUTE,
            OpType.OP_SHARED_EXPERT,
            OpType.OP_EXPERT,
            OpType.OP_AGGREGATE,
        ):
            bucket = split_ops_by_layer.setdefault(e.op.layer_idx, {})
            kind = e.op.op_type.name
            bucket.setdefault(kind, []).append(e)

    for layer_idx, bucket in split_ops_by_layer.items():
        ar_list = bucket.get("OP_ATTENTION_ROUTE", [])
        shared_list = bucket.get("OP_SHARED_EXPERT", [])
        expert_list = bucket.get("OP_EXPERT", [])
        agg_list = bucket.get("OP_AGGREGATE", [])
        if len(ar_list) != 1:
            raise ProvenanceError(
                f"split layer {layer_idx}: expected exactly one OP_ATTENTION_ROUTE, got {len(ar_list)}"
            )
        if len(shared_list) != 1:
            raise ProvenanceError(
                f"split layer {layer_idx}: expected exactly one OP_SHARED_EXPERT, got {len(shared_list)}"
            )
        if len(expert_list) == 0:
            raise ProvenanceError(
                f"split layer {layer_idx}: no OP_EXPERT entries"
            )
        if len(agg_list) != 1:
            raise ProvenanceError(
                f"split layer {layer_idx}: expected exactly one OP_AGGREGATE, got {len(agg_list)}"
            )
        seen_ids: set[int] = set()
        for e in expert_list:
            assert e.op is not None
            if e.op.expert_id in seen_ids:
                raise ProvenanceError(
                    f"split layer {layer_idx}: duplicate expert_id {e.op.expert_id}"
                )
            seen_ids.add(e.op.expert_id)
        agg = agg_list[0]
        parent_set = set(agg.parent_hashes)
        if shared_list[0].hash not in parent_set:
            raise ProvenanceError(
                f"split layer {layer_idx}: OP_AGGREGATE parent_hashes missing OP_SHARED_EXPERT"
            )
        for e in expert_list:
            if e.hash not in parent_set:
                assert e.op is not None
                raise ProvenanceError(
                    f"split layer {layer_idx}: OP_AGGREGATE parent_hashes missing OP_EXPERT {e.op.expert_id}"
                )

    # Rule 5: authorization.
    for e in chain:
        if e.op is None:
            continue
        t = e.op.op_type
        sid = e.node_id
        start_end = shard_lookup(sid)
        start, end = start_end
        if t == OpType.OP_EMBED:
            if start != 0:
                raise ProvenanceError(
                    f"OP_EMBED unauthorized: node {sid!r} is not head (start_layer != 0)"
                )
        elif t == OpType.OP_FINALIZE:
            if end != total_layers:
                raise ProvenanceError(
                    f"OP_FINALIZE unauthorized: node {sid!r} is not tail"
                )
        elif t == OpType.OP_LAYER_ATOMIC:
            layer = e.op.layer_idx
            if not (start <= layer < end):
                raise ProvenanceError(
                    f"OP_LAYER_ATOMIC layer {layer} unauthorized: node {sid!r} range [{start}, {end})"
                )
            if layer in split_layers_for_shard(sid):
                raise ProvenanceError(
                    f"OP_LAYER_ATOMIC layer {layer} unauthorized: node {sid!r} treats this layer as split"
                )
        elif t in (
            OpType.OP_ATTENTION_ROUTE,
            OpType.OP_SHARED_EXPERT,
            OpType.OP_AGGREGATE,
        ):
            layer = e.op.layer_idx
            if not (start <= layer < end):
                raise ProvenanceError(
                    f"{t.name} layer {layer} unauthorized: node {sid!r} range [{start}, {end})"
                )
            if layer not in split_layers_for_shard(sid):
                raise ProvenanceError(
                    f"{t.name} layer {layer} unauthorized: node {sid!r} doesn't treat this layer as split"
                )
        elif t == OpType.OP_EXPERT:
            owners = live_owners_of(e.op.layer_idx, e.op.expert_id)
            if sid not in owners:
                raise ProvenanceError(
                    f"OP_EXPERT layer {e.op.layer_idx} expert {e.op.expert_id} "
                    f"unauthorized: node {sid!r} not in live owners {owners}"
                )

    # Rule 4: hash tail check.
    if tail_tensor_bytes is not None:
        tail = chain[-1]
        if tail.op is None:
            raise ProvenanceError("tail entry has no op descriptor")
        expected = compute_hash(
            parent_hashes=tail.parent_hashes,
            node_id=tail.node_id,
            op=tail.op,
            output_bytes=tail_tensor_bytes,
        )
        if expected != tail.hash:
            raise ProvenanceError(
                "tail entry hash mismatch: recomputed digest differs from recorded"
            )


__all__ = [
    "ProvenanceError",
    "build_entry",
    "compute_hash",
    "entry_from_pb",
    "entry_to_pb",
    "validate_chain",
]
