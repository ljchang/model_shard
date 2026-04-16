import socket
import threading

from model_shard.membership.config import SwimConfig
from model_shard.membership.records import MemberState, StateTransition
from model_shard.membership.runner import MembershipRunner
from model_shard.membership.state import PeerSpec


def _free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_runner_starts_and_stops_cleanly() -> None:
    self_spec = PeerSpec("n0", "127.0.0.1", _free_udp_port())
    runner = MembershipRunner(self_spec=self_spec, peers=[], config=SwimConfig())
    runner.start()
    assert runner.is_alive()
    runner.stop()
    assert not runner.is_alive()


def test_runner_observer_fires_on_state_transitions() -> None:
    self_spec = PeerSpec("n0", "127.0.0.1", _free_udp_port())
    peer = PeerSpec("ghost", "127.0.0.1", _free_udp_port())  # never started
    cfg = SwimConfig(t_ping_ms=200, t_timeout_ms=100, t_suspect_ms=400)
    seen: list[StateTransition] = []
    cb_done = threading.Event()

    def cb(t: StateTransition) -> None:
        seen.append(t)
        if t.new_record.state == MemberState.DEAD:
            cb_done.set()

    runner = MembershipRunner(self_spec=self_spec, peers=[peer], config=cfg)
    runner.subscribe(cb)
    runner.start()
    try:
        # Within ~1s the runner should ping ghost, fail, suspect, and dead it.
        assert cb_done.wait(timeout=3.0)
        states = [t.new_record.state for t in seen if t.shard_id == "ghost"]
        assert MemberState.SUSPECT in states
        assert MemberState.DEAD in states
    finally:
        runner.stop()


def test_observer_exception_does_not_wedge_runner() -> None:
    self_spec = PeerSpec("n0", "127.0.0.1", _free_udp_port())
    peer = PeerSpec("ghost", "127.0.0.1", _free_udp_port())
    cfg = SwimConfig(t_ping_ms=200, t_timeout_ms=100, t_suspect_ms=400)
    other_seen = threading.Event()

    runner = MembershipRunner(self_spec=self_spec, peers=[peer], config=cfg)
    runner.subscribe(lambda _t: (_ for _ in ()).throw(RuntimeError("boom")))
    runner.subscribe(lambda _t: other_seen.set())
    runner.start()
    try:
        assert other_seen.wait(timeout=3.0)
    finally:
        runner.stop()
