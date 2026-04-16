"""SWIM-style membership discovery for the model_shard cluster.

Public surface re-exported here is what `node.py` and tests should import.
Internal modules (state, messages, transport, runner, bootstrap) may be
imported directly when writing tests against a single layer.
"""

from model_shard.membership.config import SwimConfig
from model_shard.membership.runner import MembershipRunner
from model_shard.membership.state import MembershipState, PeerSpec

__all__ = ["MembershipRunner", "MembershipState", "PeerSpec", "SwimConfig"]
