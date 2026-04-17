"""Phase 6-B wire protocol roundtrip tests."""
from __future__ import annotations

from model_shard._pb import wire_pb2


def test_op_type_enum_values():
    assert wire_pb2.OP_TYPE_UNSPECIFIED == 0
    assert wire_pb2.OP_EMBED == 1
    assert wire_pb2.OP_LAYER_ATOMIC == 2
    assert wire_pb2.OP_ATTENTION_ROUTE == 3
    assert wire_pb2.OP_EXPERT == 4
    assert wire_pb2.OP_AGGREGATE == 5
    assert wire_pb2.OP_FINALIZE == 6
    assert wire_pb2.OP_SHARED_EXPERT == 7


def test_err_invalid_provenance_present():
    assert wire_pb2.ERR_INVALID_PROVENANCE == 6


def test_op_descriptor_roundtrip():
    d = wire_pb2.OpDescriptorPb(
        op_type=wire_pb2.OP_EXPERT, layer_idx=15, expert_id=7
    )
    raw = d.SerializeToString()
    parsed = wire_pb2.OpDescriptorPb()
    parsed.ParseFromString(raw)
    assert parsed.op_type == wire_pb2.OP_EXPERT
    assert parsed.layer_idx == 15
    assert parsed.expert_id == 7


def test_provenance_entry_roundtrip():
    e = wire_pb2.ProvenanceEntryPb(
        hash=b"\x01" * 32,
        parent_hashes=[b"\x02" * 32, b"\x03" * 32],
        node_id="layer_0-10",
        timestamp=1234.5,
    )
    e.op.op_type = wire_pb2.OP_AGGREGATE
    e.op.layer_idx = 15
    raw = e.SerializeToString()
    parsed = wire_pb2.ProvenanceEntryPb()
    parsed.ParseFromString(raw)
    assert parsed.hash == b"\x01" * 32
    assert list(parsed.parent_hashes) == [b"\x02" * 32, b"\x03" * 32]
    assert parsed.node_id == "layer_0-10"
    assert parsed.op.op_type == wire_pb2.OP_AGGREGATE
    assert parsed.op.layer_idx == 15


def test_activation_carries_provenance():
    a = wire_pb2.Activation(
        protocol_version=1, request_id="r", next_layer_idx=10,
    )
    e = a.provenance.add()
    e.hash = b"\xaa" * 32
    e.node_id = "head"
    e.op.op_type = wire_pb2.OP_EMBED
    raw = a.SerializeToString()
    parsed = wire_pb2.Activation()
    parsed.ParseFromString(raw)
    assert len(parsed.provenance) == 1
    assert parsed.provenance[0].node_id == "head"


def test_expert_request_carries_provenance():
    r = wire_pb2.ExpertRequest(
        protocol_version=1, request_id="r", layer_idx=15,
    )
    r.expert_ids.append(7)
    e = r.provenance.add()
    e.hash = b"\xbb" * 32
    e.op.op_type = wire_pb2.OP_ATTENTION_ROUTE
    e.op.layer_idx = 15
    raw = r.SerializeToString()
    parsed = wire_pb2.ExpertRequest()
    parsed.ParseFromString(raw)
    assert len(parsed.provenance) == 1
    assert parsed.provenance[0].op.op_type == wire_pb2.OP_ATTENTION_ROUTE


def test_expert_response_carries_provenance():
    r = wire_pb2.ExpertResponse(
        protocol_version=1, request_id="r", layer_idx=15,
    )
    r.expert_ids.append(7)
    e = r.provenance.add()
    e.hash = b"\xcc" * 32
    e.op.op_type = wire_pb2.OP_EXPERT
    e.op.layer_idx = 15
    e.op.expert_id = 7
    raw = r.SerializeToString()
    parsed = wire_pb2.ExpertResponse()
    parsed.ParseFromString(raw)
    assert len(parsed.provenance) == 1
    assert parsed.provenance[0].op.expert_id == 7
