"""Tunable timing/fanout constants for the SWIM membership protocol.

Defaults match the Phase 2 design spec (`docs/superpowers/specs/...`). They are
chosen for the localhost-3-node prototype but remain reasonable up to ~30
nodes. All time values are in milliseconds; the state machine and runner
convert to seconds internally where useful.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SwimConfig:
    t_ping_ms: int = 1000        # interval between protocol-period pings per peer
    t_tick_ms: int = 100         # state machine clock granularity
    t_timeout_ms: int = 500      # direct ping ack deadline (half of t_ping_ms)
    k_indirect: int = 2          # ping-req fanout when a direct ping times out
    k_gossip: int = 3            # max membership deltas piggybacked per message
    mtu_bytes: int = 1400        # safe single-datagram size; messages exceeding it are dropped
    t_suspect_ms: int = 4000     # suspect-deadline window; default = 4 * t_ping_ms
