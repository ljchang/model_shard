"""Pure state machine tests. Virtual clock; no sockets, no threads."""

import random

from model_shard.membership.config import SwimConfig
from model_shard.membership.records import MemberState
from model_shard.membership.state import MembershipState, PeerSpec


def make_state(
    self_id: str = "n0",
    peers: tuple[str, ...] = ("n1", "n2"),
    seed: int = 0,
    cfg: SwimConfig | None = None,
) -> MembershipState:
    """Test helper: build a MembershipState with the named peers."""
    self_spec = PeerSpec(shard_id=self_id, host="127.0.0.1", udp_port=10000)
    peer_specs = [
        PeerSpec(shard_id=p, host="127.0.0.1", udp_port=10000 + i + 1)
        for i, p in enumerate(peers)
    ]
    return MembershipState(
        self_spec=self_spec,
        peer_specs=peer_specs,
        rng=random.Random(seed),
        config=cfg or SwimConfig(),
    )


def test_initial_view_contains_self_alive_at_incarnation_zero() -> None:
    s = make_state()
    view = s.view()
    assert "n0" in view
    rec = view["n0"]
    assert rec.state == MemberState.ALIVE
    assert rec.incarnation == 0


def test_initial_view_contains_each_peer_alive_at_incarnation_zero() -> None:
    s = make_state(peers=("n1", "n2", "n3"))
    view = s.view()
    for name in ("n1", "n2", "n3"):
        assert name in view, f"missing peer {name}"
        assert view[name].state == MemberState.ALIVE
        assert view[name].incarnation == 0


def test_view_returns_a_copy_not_internal_reference() -> None:
    s = make_state()
    view = s.view()
    view.clear()
    # Mutating the returned dict must not affect internal state.
    assert "n0" in s.view()
