"""Run a single Phase 1 node process.

Usage:
    uv run python scripts/run_node.py \\
        --config config/shards.yaml \\
        --shard head

Set SHARD_DRY_RUN=true to skip model loading (uses MagicMock); useful for
membership/gossip tests that do not exercise inference.

A tiny HTTP debug endpoint is served at tcp_port + 2000 and responds to
GET /membership with a JSON snapshot of the gossip membership view.
"""

from __future__ import annotations

import argparse
import http.server
import json
import logging
import os
import signal
import socketserver
import threading
from pathlib import Path
from types import FrameType

from model_shard.mlx_engine import LoadedModel
from model_shard.node import Node
from model_shard.shard_map import ShardMap


def _start_membership_debug_endpoint(node: Node, debug_port: int) -> None:
    handler_node = node

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/membership":
                if handler_node.membership is None:
                    payload: dict[str, object] = {}
                else:
                    view = handler_node.membership.state.view()
                    payload = {
                        sid: {"state": rec.state.name, "incarnation": rec.incarnation}
                        for sid, rec in view.items()
                    }
            elif self.path == "/loads":
                if handler_node.membership is None:
                    payload = {}
                else:
                    payload = {
                        sid: {
                            "queue_depth_ema": lr.queue_depth_ema,
                            "ts_unix_ms": lr.ts_unix_ms,
                        }
                        for sid, lr in handler_node.membership.latest_loads().items()
                    }
                    # Include this node's own load so the view is complete.
                    # latest_loads() is peer-only by design (caches inbound).
                    self_lr = handler_node.self_load_report()
                    payload[self_lr.shard_id] = {
                        "queue_depth_ema": self_lr.queue_depth_ema,
                        "ts_unix_ms": self_lr.ts_unix_ms,
                    }
            else:
                self.send_response(404)
                self.end_headers()
                return
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args: object) -> None:
            pass  # silence

    srv = socketserver.TCPServer(("127.0.0.1", debug_port), Handler)
    srv.allow_reuse_address = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    # Accept both --shard and --shard-id for backwards compatibility.
    shard_group = parser.add_mutually_exclusive_group(required=True)
    shard_group.add_argument("--shard", dest="shard_id")
    shard_group.add_argument("--shard-id", dest="shard_id")
    parser.add_argument(
        "--model",
        default=None,
        help="Model id; if omitted, uses model_id from --config shards.yaml.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("run_node")

    shard_map = ShardMap.from_yaml(args.config)
    model_id = args.model or shard_map.model_id
    if not model_id:
        parser.error(
            "no model id available: pass --model or set model_id in shards.yaml"
        )
    shard = shard_map.lookup(args.shard_id)
    # Derive total layer count from the map (max end_layer across shards).
    total_layers = max(
        shard_map.lookup(sid).end_layer for sid in shard_map.all_shards()
    )

    lm: LoadedModel
    if os.environ.get("SHARD_DRY_RUN") == "true":
        from unittest.mock import MagicMock

        mock = MagicMock()
        mock.num_layers = total_layers
        lm = mock
        log.info(
            "SHARD_DRY_RUN: skipping model load; serving shard %s layers [%d, %d) on %s:%d",
            shard.shard_id,
            shard.start_layer,
            shard.end_layer,
            shard.address.host,
            shard.address.port,
        )
    else:
        from model_shard.mlx_engine import load_model

        log.info("loading model %s", model_id)
        lm = load_model(model_id)
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
        total_layers=total_layers,
    )

    _start_membership_debug_endpoint(node, debug_port=shard.address.port + 2000)

    def _handle_signal(signum: int, _frame: FrameType | None) -> None:
        log.info("received signal %d, shutting down", signum)
        node.shutdown()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    node.serve_forever()
    log.info("node stopped")


if __name__ == "__main__":
    main()
