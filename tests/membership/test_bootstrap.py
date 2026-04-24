
from model_shard.membership.bootstrap import (
    bootstrap_sequential,
)
from model_shard.membership.records import (
    MemberRecord,
    MembershipDeltaMsg,
    MemberState,
)
from model_shard.membership.state import PeerSpec


def _rec(shard_id: str, port: int = 10001) -> MemberRecord:
    return MemberRecord(
        shard_id=shard_id,
        host="127.0.0.1",
        udp_port=port,
        state=MemberState.ALIVE,
        incarnation=0,
        model_id="",
        last_state_change=0.0,
        suspect_deadline=None,
    )


def test_bootstrap_skips_self_in_seed_list() -> None:
    self_spec = PeerSpec("me", "127.0.0.1", 10000)
    peers = [self_spec, PeerSpec("a", "127.0.0.1", 10001)]
    sent: list[tuple[str, int]] = []

    def fake_request(addr: tuple[str, int], _payload: bytes, _timeout: float) -> bytes | None:
        sent.append(addr)
        return None  # all seeds time out

    bootstrap_sequential(self_spec, peers, fake_request, timeout_s=0.5)
    # 'me' must not be in sent list.
    assert ("127.0.0.1", 10000) not in sent
    assert ("127.0.0.1", 10001) in sent


def test_bootstrap_returns_first_responding_seed_view() -> None:
    self_spec = PeerSpec("me", "127.0.0.1", 10000)
    peers = [
        self_spec,
        PeerSpec("a", "127.0.0.1", 10001),
        PeerSpec("b", "127.0.0.1", 10002),
    ]
    from model_shard.membership.messages import encode_membership_envelope
    delta = MembershipDeltaMsg(members=[_rec("me", 10000), _rec("a"), _rec("b", 10002)])
    delta_bytes = encode_membership_envelope(delta)

    def fake_request(addr: tuple[str, int], payload: bytes, timeout: float) -> bytes | None:
        # 'a' (port 10001) succeeds; 'b' would too but should never be tried.
        if addr == ("127.0.0.1", 10001):
            return delta_bytes
        raise AssertionError(f"unexpected request to {addr}")

    result = bootstrap_sequential(self_spec, peers, fake_request, timeout_s=0.5)
    assert result.success is True
    assert {m.shard_id for m in result.members} >= {"me", "a", "b"}


def test_bootstrap_returns_failure_when_all_seeds_silent() -> None:
    self_spec = PeerSpec("me", "127.0.0.1", 10000)
    peers = [self_spec, PeerSpec("a", "127.0.0.1", 10001)]

    def fake_request(*_args: object, **_kwargs: object) -> bytes | None:
        return None

    result = bootstrap_sequential(self_spec, peers, fake_request, timeout_s=0.5)
    assert result.success is False
    assert result.members == []
