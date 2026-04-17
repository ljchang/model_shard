"""Fast integration tests: chain carriage on Activation, Node validation."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from model_shard.node import Node, _provenance_enabled
from model_shard.shard_map import NodeAddress, ShardMap, ShardSpec


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("ENABLE_GOSSIP", "false")
    monkeypatch.setenv("ENABLE_PARTIAL_LOAD", "false")
    monkeypatch.setenv("ENABLE_DYNAMIC_MIGRATION", "false")
    monkeypatch.setenv("ENABLE_EXPERT_RETRY", "false")
    yield


def _mk_spec(sid: str, port: int, start: int, end: int) -> ShardSpec:
    return ShardSpec(
        shard_id=sid,
        address=NodeAddress(host="127.0.0.1", port=port),
        start_layer=start, end_layer=end,
        moe_experts={},
    )


def test_provenance_gate_env_var_default_off():
    # Default is off.
    assert _provenance_enabled() is False


def test_provenance_gate_on_when_env_set(monkeypatch):
    monkeypatch.setenv("ENABLE_PROVENANCE", "true")
    assert _provenance_enabled() is True


def test_node_has_provenance_enabled_attribute(monkeypatch):
    monkeypatch.setenv("ENABLE_PROVENANCE", "true")
    spec_head = _mk_spec("head", 30500, 0, 10)
    spec_tail = _mk_spec("tail", 30501, 10, 30)
    sm = ShardMap({"head": spec_head, "tail": spec_tail})
    n = Node(shard=spec_head, shard_map=sm, loaded_model=MagicMock(), total_layers=30)
    assert n._provenance_enabled is True


def test_node_provenance_disabled_when_env_off():
    spec_head = _mk_spec("head", 30502, 0, 10)
    spec_tail = _mk_spec("tail", 30503, 10, 30)
    sm = ShardMap({"head": spec_head, "tail": spec_tail})
    n = Node(shard=spec_head, shard_map=sm, loaded_model=MagicMock(), total_layers=30)
    assert n._provenance_enabled is False


def test_node_shard_lookup_returns_layer_range():
    spec_head = _mk_spec("head", 30504, 0, 10)
    spec_mid = _mk_spec("mid", 30505, 10, 20)
    spec_tail = _mk_spec("tail", 30506, 20, 30)
    sm = ShardMap({"head": spec_head, "mid": spec_mid, "tail": spec_tail})
    n = Node(shard=spec_head, shard_map=sm, loaded_model=MagicMock(), total_layers=30)
    assert n._shard_lookup("head") == (0, 10)
    assert n._shard_lookup("mid") == (10, 20)
    assert n._shard_lookup("tail") == (20, 30)
    # Unknown shard: returns (0, 0) which means unauthorized.
    assert n._shard_lookup("unknown") == (0, 0)


def test_node_split_layers_for_shard():
    spec_head = _mk_spec("head", 30507, 0, 10)
    spec_mid = ShardSpec(
        shard_id="mid",
        address=NodeAddress(host="127.0.0.1", port=30508),
        start_layer=10, end_layer=20,
        moe_experts={15: (0, 1, 2)},
    )
    spec_tail = _mk_spec("tail", 30509, 20, 30)
    sm = ShardMap({"head": spec_head, "mid": spec_mid, "tail": spec_tail})
    n = Node(shard=spec_head, shard_map=sm, loaded_model=MagicMock(), total_layers=30)
    assert n._split_layers_for_shard("mid") == {15}
    assert n._split_layers_for_shard("head") == set()
    assert n._split_layers_for_shard("unknown") == set()


def test_node_has_pending_finalize_dict(monkeypatch):
    monkeypatch.setenv("ENABLE_PROVENANCE", "true")
    spec_head = _mk_spec("head", 30510, 0, 10)
    spec_tail = _mk_spec("tail", 30511, 10, 30)
    sm = ShardMap({"head": spec_head, "tail": spec_tail})
    n = Node(shard=spec_head, shard_map=sm, loaded_model=MagicMock(), total_layers=30)
    assert n._pending_finalize == {}


def test_orchestrator_produces_split_layer_chain():
    """Deferred to Task 8 slow Tier 1.

    The split-layer chain's construction is entangled with real MLX router
    behavior — the router's top_k_indices drive which experts get OP_EXPERT
    entries, and the specific hash values depend on actual tensor bytes.
    A fast test that mocked all of this would test mock plumbing rather
    than real DAG shape. The slow Tier 1 test in Task 8
    (``test_provenance_tier1.py``) exercises the full end-to-end chain
    construction with a real Gemma 4 model."""
    import pytest
    pytest.skip("deferred to Task 8 slow Tier 1 — see docstring")
