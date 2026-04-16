from model_shard.membership.config import SwimConfig


def test_swim_config_has_spec_default_values() -> None:
    cfg = SwimConfig()
    assert cfg.t_ping_ms == 1000
    assert cfg.t_tick_ms == 100
    assert cfg.t_timeout_ms == 500
    assert cfg.k_indirect == 2
    assert cfg.t_suspect_ms == 4 * cfg.t_ping_ms
    assert cfg.k_gossip == 3
    assert cfg.mtu_bytes == 1400


def test_swim_config_is_frozen() -> None:
    cfg = SwimConfig()
    import dataclasses
    assert dataclasses.is_dataclass(cfg)
    try:
        cfg.t_ping_ms = 999  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("SwimConfig should be frozen")
