"""Round-trip tests for the Phase 5b protobuf additions."""
from __future__ import annotations

from model_shard._pb import wire_pb2


def test_expert_weight_request_fields():
    req = wire_pb2.ExpertWeightRequest(
        protocol_version=1, request_id="abc", layer_idx=15, expert_id=7
    )
    raw = req.SerializeToString()
    parsed = wire_pb2.ExpertWeightRequest()
    parsed.ParseFromString(raw)
    assert parsed.protocol_version == 1
    assert parsed.request_id == "abc"
    assert parsed.layer_idx == 15
    assert parsed.expert_id == 7


def test_expert_weight_transfer_nine_descriptors():
    t = wire_pb2.ExpertWeightTransfer(
        protocol_version=1, request_id="abc", layer_idx=15, expert_id=7,
        tensor_count=9,
    )
    for i in range(9):
        d = t.tensors.add()
        d.shape.extend([704, 352])
        d.dtype = wire_pb2.DTYPE_BFLOAT16
        d.quant = wire_pb2.QUANT_NONE
        d.byte_count = 100 + i
    raw = t.SerializeToString()
    parsed = wire_pb2.ExpertWeightTransfer()
    parsed.ParseFromString(raw)
    assert parsed.tensor_count == 9
    assert len(parsed.tensors) == 9
    assert [int(d.byte_count) for d in parsed.tensors] == [100 + i for i in range(9)]


def test_envelope_oneof_recognises_new_payloads():
    env = wire_pb2.Envelope()
    env.expert_weight_request.protocol_version = 1
    env.expert_weight_request.request_id = "r"
    env.expert_weight_request.layer_idx = 15
    env.expert_weight_request.expert_id = 7
    assert env.WhichOneof("payload") == "expert_weight_request"

    env2 = wire_pb2.Envelope()
    env2.expert_weight_transfer.protocol_version = 1
    env2.expert_weight_transfer.request_id = "r"
    env2.expert_weight_transfer.layer_idx = 15
    env2.expert_weight_transfer.expert_id = 7
    env2.expert_weight_transfer.tensor_count = 9
    assert env2.WhichOneof("payload") == "expert_weight_transfer"


def test_ping_carries_heat_and_ownership():
    p = wire_pb2.Ping(protocol_version=1, from_shard_id="a", from_incarnation=1)
    hr = p.heat.add()
    hr.shard_id = "a"
    hr.ts_unix_ms = 1234
    entry = hr.entries.add()
    entry.layer_idx = 15
    entry.expert_id = 7
    entry.heat_ema_x100 = 500

    od = p.ownership.add()
    od.shard_id = "a"
    od.layer_idx = 15
    od.expert_id = 7
    od.action = 0

    raw = p.SerializeToString()
    parsed = wire_pb2.Ping()
    parsed.ParseFromString(raw)
    assert len(parsed.heat) == 1
    assert parsed.heat[0].entries[0].heat_ema_x100 == 500
    assert len(parsed.ownership) == 1
    assert parsed.ownership[0].expert_id == 7
