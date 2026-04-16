"""Run a single Phase 1 node process.

Usage:
    uv run python scripts/run_node.py \\
        --config config/shards.yaml \\
        --shard-id layer_0-10 \\
        [--model mlx-community/gemma-4-26b-a4b-it-4bit]
"""

from __future__ import annotations

import argparse
import logging
import signal
from pathlib import Path
from types import FrameType

from model_shard.mlx_engine import load_model
from model_shard.node import Node
from model_shard.shard_map import ShardMap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shard-id", required=True)
    parser.add_argument("--model", default="mlx-community/gemma-4-26b-a4b-it-4bit")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("run_node")

    shard_map = ShardMap.from_yaml(args.config)
    shard = shard_map.lookup(args.shard_id)
    # Derive total layer count from the map (max end_layer across shards).
    total_layers = max(
        shard_map.lookup(sid).end_layer for sid in shard_map.all_shards()
    )

    log.info("loading model %s", args.model)
    lm = load_model(args.model)
    log.info(
        "loaded: %d layers; serving shard %s layers [%d, %d) on %s:%d",
        lm.num_layers,
        shard.shard_id,
        shard.start_layer,
        shard.end_layer,
        shard.address.host,
        shard.address.port,
    )
    if lm.num_layers != total_layers:
        log.warning(
            "model has %d layers but config expects %d — using model value",
            lm.num_layers,
            total_layers,
        )

    node = Node(
        shard=shard,
        shard_map=shard_map,
        loaded_model=lm,
        total_layers=lm.num_layers,
    )

    def _handle_signal(signum: int, _frame: FrameType | None) -> None:
        log.info("received signal %d, shutting down", signum)
        node.shutdown()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    node.serve_forever()
    log.info("node stopped")


if __name__ == "__main__":
    main()
