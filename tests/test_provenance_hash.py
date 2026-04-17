"""Hash-level unit tests for Phase 6-B provenance."""
from __future__ import annotations

import mlx.core as mx

from model_shard.provenance import (
    build_entry,
    compute_hash,
    entry_from_pb,
    entry_to_pb,
)
from model_shard.request import OpDescriptor, OpType


def test_compute_hash_is_deterministic():
    h1 = compute_hash(
        parent_hashes=(b"\xaa" * 32,),
        node_id="head",
        op=OpDescriptor(op_type=OpType.OP_EMBED),
        output_bytes=b"\x01\x02\x03",
    )
    h2 = compute_hash(
        parent_hashes=(b"\xaa" * 32,),
        node_id="head",
        op=OpDescriptor(op_type=OpType.OP_EMBED),
        output_bytes=b"\x01\x02\x03",
    )
    assert h1 == h2
    assert len(h1) == 32  # BLAKE2b-256 digest size


def test_compute_hash_depends_on_parent_hashes():
    base = compute_hash(
        parent_hashes=(b"\xaa" * 32,),
        node_id="head",
        op=OpDescriptor(op_type=OpType.OP_EMBED),
        output_bytes=b"\x01",
    )
    different = compute_hash(
        parent_hashes=(b"\xbb" * 32,),
        node_id="head",
        op=OpDescriptor(op_type=OpType.OP_EMBED),
        output_bytes=b"\x01",
    )
    assert base != different


def test_compute_hash_depends_on_node_id():
    base = compute_hash(
        parent_hashes=(b"\xaa" * 32,),
        node_id="head",
        op=OpDescriptor(op_type=OpType.OP_EMBED),
        output_bytes=b"\x01",
    )
    different = compute_hash(
        parent_hashes=(b"\xaa" * 32,),
        node_id="tail",
        op=OpDescriptor(op_type=OpType.OP_EMBED),
        output_bytes=b"\x01",
    )
    assert base != different


def test_compute_hash_depends_on_op():
    base = compute_hash(
        parent_hashes=(b"\xaa" * 32,),
        node_id="head",
        op=OpDescriptor(op_type=OpType.OP_EMBED),
        output_bytes=b"\x01",
    )
    different = compute_hash(
        parent_hashes=(b"\xaa" * 32,),
        node_id="head",
        op=OpDescriptor(op_type=OpType.OP_LAYER_ATOMIC, layer_idx=1),
        output_bytes=b"\x01",
    )
    assert base != different


def test_compute_hash_depends_on_output_bytes():
    base = compute_hash(
        parent_hashes=(b"\xaa" * 32,),
        node_id="head",
        op=OpDescriptor(op_type=OpType.OP_EMBED),
        output_bytes=b"\x01",
    )
    different = compute_hash(
        parent_hashes=(b"\xaa" * 32,),
        node_id="head",
        op=OpDescriptor(op_type=OpType.OP_EMBED),
        output_bytes=b"\x02",
    )
    assert base != different


def test_compute_hash_multiple_parents_order_matters():
    h_ab = compute_hash(
        parent_hashes=(b"\xaa" * 32, b"\xbb" * 32),
        node_id="mid",
        op=OpDescriptor(op_type=OpType.OP_AGGREGATE, layer_idx=15),
        output_bytes=b"",
    )
    h_ba = compute_hash(
        parent_hashes=(b"\xbb" * 32, b"\xaa" * 32),
        node_id="mid",
        op=OpDescriptor(op_type=OpType.OP_AGGREGATE, layer_idx=15),
        output_bytes=b"",
    )
    assert h_ab != h_ba  # order-sensitive so DAG hashing is unambiguous


def test_build_entry_sets_hash_and_op():
    tensor = mx.full((2, 2), 1.0, dtype=mx.bfloat16)
    parents: tuple[bytes, ...] = ()
    entry = build_entry(
        node_id="head",
        op=OpDescriptor(op_type=OpType.OP_EMBED),
        output_tensor=tensor,
        parent_hashes=parents,
    )
    assert entry.node_id == "head"
    assert entry.shard_id == "head"  # shard_id == node_id in 6-B
    assert entry.op is not None and entry.op.op_type == OpType.OP_EMBED
    assert entry.parent_hashes == ()
    assert len(entry.hash) == 32


def test_entry_pb_roundtrip():
    tensor = mx.full((1, 8), 2.0, dtype=mx.bfloat16)
    entry = build_entry(
        node_id="mid",
        op=OpDescriptor(op_type=OpType.OP_ATTENTION_ROUTE, layer_idx=15),
        output_tensor=tensor,
        parent_hashes=(b"\xcc" * 32,),
    )
    pb = entry_to_pb(entry)
    roundtripped = entry_from_pb(pb)
    assert roundtripped.node_id == entry.node_id
    assert roundtripped.shard_id == entry.shard_id
    assert roundtripped.hash == entry.hash
    assert roundtripped.parent_hashes == entry.parent_hashes
    assert roundtripped.op is not None
    assert roundtripped.op.op_type == OpType.OP_ATTENTION_ROUTE
    assert roundtripped.op.layer_idx == 15
