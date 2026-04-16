"""Tests for Request and ProvenanceEntry data types."""

import time

from model_shard.request import ProvenanceEntry, Request


def test_request_basic_construction() -> None:
    r = Request(
        request_id="req-1",
        sequence_id="seq-1",
        prompt_token_ids=[10, 20, 30],
    )
    assert r.request_id == "req-1"
    assert r.sequence_id == "seq-1"
    assert r.prompt_token_ids == [10, 20, 30]
    assert r.position == 0
    assert r.provenance == []


def test_request_append_provenance_populates_entry() -> None:
    r = Request(request_id="r", sequence_id="s", prompt_token_ids=[])
    before = time.time()
    r.append_provenance(shard_id="layer_0-10", node_id="n1")
    after = time.time()

    assert len(r.provenance) == 1
    entry = r.provenance[0]
    assert entry.shard_id == "layer_0-10"
    assert entry.node_id == "n1"
    assert before <= entry.timestamp <= after


def test_provenance_preserves_append_order() -> None:
    r = Request(request_id="r", sequence_id="s", prompt_token_ids=[])
    r.append_provenance(shard_id="a", node_id="n1")
    r.append_provenance(shard_id="b", node_id="n2")
    r.append_provenance(shard_id="c", node_id="n1")

    assert [p.shard_id for p in r.provenance] == ["a", "b", "c"]
    assert [p.node_id for p in r.provenance] == ["n1", "n2", "n1"]


def test_provenance_entry_hash_defaults_empty() -> None:
    """Hash field exists for Phase 6 verification; unused for now, must default."""
    entry = ProvenanceEntry(shard_id="s", node_id="n", timestamp=1.0)
    assert entry.hash == b""


def test_provenance_entry_hash_can_be_set() -> None:
    entry = ProvenanceEntry(shard_id="s", node_id="n", timestamp=1.0, hash=b"\xde\xad")
    assert entry.hash == b"\xde\xad"


def test_request_position_can_be_advanced() -> None:
    """Orchestrator increments position after each sampled token."""
    r = Request(request_id="r", sequence_id="s", prompt_token_ids=[1, 2, 3])
    assert r.position == 0
    r.position = 3  # after prefill
    assert r.position == 3
    r.position += 1
    assert r.position == 4
