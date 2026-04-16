"""SWIM-style membership discovery for the model_shard cluster.

Public surface re-exported here is what `node.py` and tests should import.
Internal modules (state, messages, transport, runner, bootstrap) may be
imported directly when writing tests against a single layer.
"""

from model_shard.membership.config import SwimConfig

__all__ = ["SwimConfig"]
