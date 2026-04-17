"""Dataclass tests for Phase 6-B provenance extensions."""
from __future__ import annotations

from model_shard.request import (
    OpDescriptor,
    OpType,
    ProvenanceEntry,
    Request,
)


def test_op_type_int_values_match_wire_enum():
    from model_shard._pb import wire_pb2
    assert int(OpType.OP_EMBED) == wire_pb2.OP_EMBED
    assert int(OpType.OP_LAYER_ATOMIC) == wire_pb2.OP_LAYER_ATOMIC
    assert int(OpType.OP_ATTENTION_ROUTE) == wire_pb2.OP_ATTENTION_ROUTE
    assert int(OpType.OP_EXPERT) == wire_pb2.OP_EXPERT
    assert int(OpType.OP_AGGREGATE) == wire_pb2.OP_AGGREGATE
    assert int(OpType.OP_FINALIZE) == wire_pb2.OP_FINALIZE
    assert int(OpType.OP_SHARED_EXPERT) == wire_pb2.OP_SHARED_EXPERT


def test_op_descriptor_pack_is_deterministic():
    d1 = OpDescriptor(op_type=OpType.OP_EXPERT, layer_idx=15, expert_id=7)
    d2 = OpDescriptor(op_type=OpType.OP_EXPERT, layer_idx=15, expert_id=7)
    assert d1.pack() == d2.pack()


def test_op_descriptor_pack_differentiates():
    d1 = OpDescriptor(op_type=OpType.OP_EXPERT, layer_idx=15, expert_id=7)
    d2 = OpDescriptor(op_type=OpType.OP_EXPERT, layer_idx=15, expert_id=8)
    assert d1.pack() != d2.pack()


def test_op_descriptor_pack_is_exactly_9_bytes():
    d = OpDescriptor(op_type=OpType.OP_EMBED, layer_idx=0, expert_id=0)
    assert len(d.pack()) == 9  # uint8 + uint32 + uint32


def test_provenance_entry_frozen_with_new_fields():
    e = ProvenanceEntry(
        shard_id="head",
        node_id="head",
        timestamp=1.0,
        hash=b"\xaa" * 32,
        parent_hashes=(b"\xbb" * 32,),
        op=OpDescriptor(op_type=OpType.OP_LAYER_ATOMIC, layer_idx=0),
    )
    assert e.parent_hashes == (b"\xbb" * 32,)
    assert e.op is not None
    assert e.op.op_type == OpType.OP_LAYER_ATOMIC
    try:
        e.node_id = "other"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("ProvenanceEntry should be frozen")


def test_provenance_entry_backward_compat_phase1_shape():
    # Old-style construction (only shard_id/node_id/timestamp/hash) still works.
    e = ProvenanceEntry(shard_id="s", node_id="n", timestamp=1.0)
    assert e.parent_hashes == ()
    assert e.op is None


def test_request_append_provenance_extended_kwargs():
    r = Request(request_id="r", sequence_id="s", prompt_token_ids=[1, 2])
    r.append_provenance(
        shard_id="head",
        node_id="head",
        hash=b"\xaa" * 32,
        parent_hashes=(b"\xbb" * 32,),
        op=OpDescriptor(op_type=OpType.OP_EMBED),
    )
    assert len(r.provenance) == 1
    entry = r.provenance[0]
    assert entry.op is not None
    assert entry.op.op_type == OpType.OP_EMBED
    assert entry.parent_hashes == (b"\xbb" * 32,)


def test_request_append_provenance_phase1_compat():
    # Old-style call (no op, no parent_hashes) still works.
    r = Request(request_id="r", sequence_id="s", prompt_token_ids=[1, 2])
    r.append_provenance(shard_id="head", node_id="head")
    assert r.provenance[0].op is None
    assert r.provenance[0].parent_hashes == ()
