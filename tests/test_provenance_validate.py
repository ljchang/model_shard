"""Validation-rule tests for Phase 6-B provenance.

Each test constructs a synthetic chain and asserts validate_chain either
accepts or rejects it per D8 rules 1-5."""
from __future__ import annotations

import pytest

from model_shard.provenance import ProvenanceError, validate_chain
from model_shard.request import OpDescriptor, OpType, ProvenanceEntry


def _entry(
    *, node_id: str, op_type: OpType, layer_idx: int = 0, expert_id: int = 0,
    hash_: bytes = b"\x00" * 32, parent_hashes: tuple[bytes, ...] = (),
) -> ProvenanceEntry:
    """Construct a synthetic entry; hash is whatever the caller provides
    (not recomputed). For validation tests we care about rules 1-3 and 5
    independently of hash content."""
    return ProvenanceEntry(
        shard_id=node_id,
        node_id=node_id,
        timestamp=0.0,
        hash=hash_,
        parent_hashes=parent_hashes,
        op=OpDescriptor(op_type=op_type, layer_idx=layer_idx, expert_id=expert_id),
    )


# Standard test cluster shape used by these unit tests:
# - head: shard_id="head", start_layer=0, end_layer=10
# - mid:  shard_id="mid",  start_layer=10, end_layer=20, split layer 15
# - tail: shard_id="tail", start_layer=20, end_layer=30, is tail
_TOTAL_LAYERS = 30
_SPLIT_LAYERS = {15}


def _mk_owners_view():
    # Simple: expert E at layer 15 is owned by mid for E%3==1, else tail/head.
    def owners_of(layer_idx: int, expert_id: int) -> set[str]:
        if expert_id % 3 == 0:
            return {"head"}
        if expert_id % 3 == 1:
            return {"mid"}
        return {"tail"}
    return owners_of


def _mk_shard_view():
    shards = {"head": (0, 10), "mid": (10, 20), "tail": (20, 30)}
    return lambda sid: shards.get(sid, (0, 0))


def _mk_wellformed_chain() -> list[ProvenanceEntry]:
    """Construct a valid 40-entry chain for the test cluster."""
    prev: bytes = b"\x00" * 32
    chain: list[ProvenanceEntry] = []

    e = _entry(node_id="head", op_type=OpType.OP_EMBED, hash_=b"\x01" * 32)
    chain.append(e)
    prev = e.hash

    for layer in range(0, 15):
        owner = "head" if layer < 10 else "mid"
        e = _entry(
            node_id=owner, op_type=OpType.OP_LAYER_ATOMIC, layer_idx=layer,
            hash_=bytes([layer + 2]) + b"\x00" * 31,
            parent_hashes=(prev,),
        )
        chain.append(e)
        prev = e.hash

    ar = _entry(node_id="mid", op_type=OpType.OP_ATTENTION_ROUTE, layer_idx=15,
                hash_=b"\x10" * 32, parent_hashes=(prev,))
    chain.append(ar)
    shared = _entry(node_id="mid", op_type=OpType.OP_SHARED_EXPERT, layer_idx=15,
                    hash_=b"\x11" * 32, parent_hashes=(ar.hash,))
    chain.append(shared)
    exp_hashes = []
    for eid in (0, 1, 2):
        owner = {0: "head", 1: "mid", 2: "tail"}[eid]
        e = _entry(
            node_id=owner, op_type=OpType.OP_EXPERT, layer_idx=15, expert_id=eid,
            hash_=bytes([0x20 + eid]) + b"\x00" * 31,
            parent_hashes=(ar.hash,),
        )
        chain.append(e)
        exp_hashes.append(e.hash)
    agg = _entry(
        node_id="mid", op_type=OpType.OP_AGGREGATE, layer_idx=15,
        hash_=b"\x30" * 32,
        parent_hashes=(shared.hash, *exp_hashes),
    )
    chain.append(agg)
    prev = agg.hash

    for layer in range(16, 30):
        owner = "mid" if layer < 20 else "tail"
        e = _entry(
            node_id=owner, op_type=OpType.OP_LAYER_ATOMIC, layer_idx=layer,
            hash_=bytes([0x40 + (layer - 16)]) + b"\x00" * 31,
            parent_hashes=(prev,),
        )
        chain.append(e)
        prev = e.hash

    fin = _entry(
        node_id="tail", op_type=OpType.OP_FINALIZE,
        hash_=b"\xff" * 32, parent_hashes=(prev,),
    )
    chain.append(fin)

    return chain


def test_validate_accepts_wellformed_chain():
    chain = _mk_wellformed_chain()
    validate_chain(
        chain,
        shard_lookup=_mk_shard_view(),
        total_layers=_TOTAL_LAYERS,
        split_layers_for_shard=lambda sid: _SPLIT_LAYERS if sid == "mid" else set(),
        live_owners_of=_mk_owners_view(),
        tail_tensor_bytes=None,
    )


def test_validate_rejects_missing_embed():
    chain = _mk_wellformed_chain()[1:]
    with pytest.raises(ProvenanceError, match="OP_EMBED"):
        validate_chain(
            chain, shard_lookup=_mk_shard_view(), total_layers=_TOTAL_LAYERS,
            split_layers_for_shard=lambda sid: _SPLIT_LAYERS if sid == "mid" else set(),
            live_owners_of=_mk_owners_view(), tail_tensor_bytes=None,
        )


def test_validate_rejects_finalize_in_middle():
    chain = _mk_wellformed_chain()
    chain.insert(5, _entry(node_id="tail", op_type=OpType.OP_FINALIZE,
                            hash_=b"\xfe" * 32, parent_hashes=(chain[4].hash,)))
    with pytest.raises(ProvenanceError, match="OP_FINALIZE"):
        validate_chain(
            chain, shard_lookup=_mk_shard_view(), total_layers=_TOTAL_LAYERS,
            split_layers_for_shard=lambda sid: _SPLIT_LAYERS if sid == "mid" else set(),
            live_owners_of=_mk_owners_view(), tail_tensor_bytes=None,
        )


def test_validate_rejects_skipped_layer():
    chain = _mk_wellformed_chain()
    chain = [e for e in chain
             if not (e.op and e.op.op_type == OpType.OP_LAYER_ATOMIC
                     and e.op.layer_idx == 12)]
    with pytest.raises(ProvenanceError, match="layer 12"):
        validate_chain(
            chain, shard_lookup=_mk_shard_view(), total_layers=_TOTAL_LAYERS,
            split_layers_for_shard=lambda sid: _SPLIT_LAYERS if sid == "mid" else set(),
            live_owners_of=_mk_owners_view(), tail_tensor_bytes=None,
        )


def test_validate_rejects_unauthorized_layer_node():
    chain = _mk_wellformed_chain()
    for i, e in enumerate(chain):
        if e.op and e.op.op_type == OpType.OP_LAYER_ATOMIC and e.op.layer_idx == 12:
            chain[i] = _entry(
                node_id="tail", op_type=OpType.OP_LAYER_ATOMIC, layer_idx=12,
                hash_=e.hash, parent_hashes=e.parent_hashes,
            )
            break
    with pytest.raises(ProvenanceError, match="unauthorized"):
        validate_chain(
            chain, shard_lookup=_mk_shard_view(), total_layers=_TOTAL_LAYERS,
            split_layers_for_shard=lambda sid: _SPLIT_LAYERS if sid == "mid" else set(),
            live_owners_of=_mk_owners_view(), tail_tensor_bytes=None,
        )


def test_validate_rejects_unauthorized_expert_owner():
    chain = _mk_wellformed_chain()
    for i, e in enumerate(chain):
        if e.op and e.op.op_type == OpType.OP_EXPERT and e.op.expert_id == 1:
            chain[i] = _entry(
                node_id="head", op_type=OpType.OP_EXPERT, layer_idx=15, expert_id=1,
                hash_=e.hash, parent_hashes=e.parent_hashes,
            )
            break
    with pytest.raises(ProvenanceError, match="unauthorized"):
        validate_chain(
            chain, shard_lookup=_mk_shard_view(), total_layers=_TOTAL_LAYERS,
            split_layers_for_shard=lambda sid: _SPLIT_LAYERS if sid == "mid" else set(),
            live_owners_of=_mk_owners_view(), tail_tensor_bytes=None,
        )


def test_validate_rejects_missing_shared_expert():
    chain = [e for e in _mk_wellformed_chain()
             if not (e.op and e.op.op_type == OpType.OP_SHARED_EXPERT)]
    with pytest.raises(ProvenanceError, match="OP_SHARED_EXPERT"):
        validate_chain(
            chain, shard_lookup=_mk_shard_view(), total_layers=_TOTAL_LAYERS,
            split_layers_for_shard=lambda sid: _SPLIT_LAYERS if sid == "mid" else set(),
            live_owners_of=_mk_owners_view(), tail_tensor_bytes=None,
        )


def test_validate_rejects_aggregate_missing_expert_parent():
    chain = _mk_wellformed_chain()
    for i, e in enumerate(chain):
        if e.op and e.op.op_type == OpType.OP_AGGREGATE:
            chain[i] = _entry(
                node_id="mid", op_type=OpType.OP_AGGREGATE, layer_idx=15,
                hash_=e.hash, parent_hashes=e.parent_hashes[:-1],
            )
            break
    with pytest.raises(ProvenanceError, match="parent"):
        validate_chain(
            chain, shard_lookup=_mk_shard_view(), total_layers=_TOTAL_LAYERS,
            split_layers_for_shard=lambda sid: _SPLIT_LAYERS if sid == "mid" else set(),
            live_owners_of=_mk_owners_view(), tail_tensor_bytes=None,
        )


def test_validate_rejects_duplicate_expert_in_split_layer():
    chain = _mk_wellformed_chain()
    idx = next(i for i, e in enumerate(chain)
               if e.op and e.op.op_type == OpType.OP_EXPERT and e.op.expert_id == 1)
    dup = _entry(
        node_id="mid", op_type=OpType.OP_EXPERT, layer_idx=15, expert_id=1,
        hash_=b"\xab" * 32, parent_hashes=chain[idx].parent_hashes,
    )
    chain.insert(idx + 1, dup)
    with pytest.raises(ProvenanceError, match="duplicate"):
        validate_chain(
            chain, shard_lookup=_mk_shard_view(), total_layers=_TOTAL_LAYERS,
            split_layers_for_shard=lambda sid: _SPLIT_LAYERS if sid == "mid" else set(),
            live_owners_of=_mk_owners_view(), tail_tensor_bytes=None,
        )


def test_validate_rejects_tampered_tail_hash():
    chain = _mk_wellformed_chain()
    chain = chain[:-1]
    tampered_bytes = b"\xff" * 64
    with pytest.raises(ProvenanceError, match="hash"):
        validate_chain(
            chain, shard_lookup=_mk_shard_view(), total_layers=_TOTAL_LAYERS,
            split_layers_for_shard=lambda sid: _SPLIT_LAYERS if sid == "mid" else set(),
            live_owners_of=_mk_owners_view(),
            tail_tensor_bytes=tampered_bytes,
        )
