"""Node's inbound handler for ExpertRequest.

Phase 3 Task 13: the node dispatch must route an incoming
``expert_request`` envelope to a handler that (a) decodes the out-of-band
hidden-state tensor, (b) runs ``moe.run_selected_experts`` for the
requested expert ids, (c) stacks per-expert outputs on axis=2, and (d)
returns an ``ExpertResponse`` on the same TCP stream. If any requested
expert id is not hosted by this shard, the handler must respond with
``Error{ERR_WRONG_SHARD}`` instead.

Both tests run a real ``Node.serve_forever`` in a daemon thread and speak
the on-wire protocol over a real loopback socket.
"""

from __future__ import annotations

import random
import socket
import threading
import time

import mlx.core as mx
import pytest

from model_shard._pb import wire_pb2
from model_shard.envelope import recv_envelope, send_envelope
from model_shard.mlx_engine import bytes_to_tensor, tensor_to_bytes
from model_shard.node import Node
from model_shard.shard_map import NodeAddress, ShardMap, ShardSpec


def _dtype_to_wire(dt: mx.Dtype) -> int:
    if dt == mx.bfloat16:
        return int(wire_pb2.DTYPE_BFLOAT16)
    if dt == mx.float32:
        return int(wire_pb2.DTYPE_FLOAT32)
    if dt == mx.float16:
        return int(wire_pb2.DTYPE_FLOAT16)
    raise ValueError(f"unsupported dtype: {dt}")


def _free_port() -> int:
    for _ in range(100):
        p = random.randint(30000, 60000)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", p))
            return p
        except OSError:
            continue
    raise RuntimeError("no free port")


def _wait_listening(host: str, port: int, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError(f"{host}:{port} never came up")


def _build_expert_request(
    request_id: str,
    layer_idx: int,
    expert_ids: list[int],
    h: mx.array,
) -> tuple[wire_pb2.Envelope, bytes]:
    env = wire_pb2.Envelope()
    env.expert_request.protocol_version = 1
    env.expert_request.request_id = request_id
    env.expert_request.layer_idx = layer_idx
    env.expert_request.expert_ids.extend(expert_ids)
    raw = tensor_to_bytes(h)
    env.expert_request.h_spec.shape.extend(list(h.shape))
    env.expert_request.h_spec.dtype = _dtype_to_wire(h.dtype)
    env.expert_request.h_spec.quant = wire_pb2.QUANT_NONE
    env.expert_request.h_spec.byte_count = len(raw)
    return env, raw


def _solo_node(loaded_model, port: int, moe_experts: dict[int, tuple[int, ...]]) -> Node:
    """Build a two-shard ShardMap where ``solo`` hosts all layers and a
    throw-away ``dummy`` ShardSpec serves only to satisfy
    ``_resolve_downstream`` (a solo shard is both head and tail, so it would
    try to dial a downstream head on the loopback). This handler-focused test
    never actually forwards activations, so the dummy is never connected to.
    """
    spec = ShardSpec(
        shard_id="solo",
        address=NodeAddress("127.0.0.1", port),
        start_layer=0,
        end_layer=30,
        moe_experts=moe_experts,
    )
    # Downstream-for-solo (the tail dials a head at start_layer=0). We make
    # ``dummy`` occupy start_layer=0 with end_layer=0 (empty range) so the
    # resolver picks it; we never actually write to its address.
    dummy = ShardSpec(
        shard_id="dummy",
        address=NodeAddress("127.0.0.1", 1),  # reserved port; unreachable
        start_layer=0,
        end_layer=0,
    )
    sm = ShardMap({"solo": spec, "dummy": dummy})
    return Node(
        shard=spec,
        shard_map=sm,
        loaded_model=loaded_model,
        total_layers=30,
    )


@pytest.mark.slow
def test_node_expert_request_handler_returns_valid_response(
    loaded_model, monkeypatch
) -> None:
    # Gossip would try UDP membership against the dummy peer; bypass for this
    # narrow handler-focused test.
    monkeypatch.setenv("ENABLE_GOSSIP", "false")
    port = _free_port()
    node = _solo_node(loaded_model, port, moe_experts={15: (3, 6, 9)})

    t = threading.Thread(target=node.serve_forever, daemon=True)
    t.start()
    _wait_listening("127.0.0.1", port)

    try:
        s = socket.create_connection(("127.0.0.1", port), timeout=10)
        try:
            hidden = loaded_model.text_model.config.hidden_size
            # bf16 matches the production activation dtype.
            h = mx.random.normal((1, 2, hidden)).astype(mx.bfloat16)
            mx.eval(h)
            env, raw = _build_expert_request("r1", 15, [3, 6, 9], h)
            stream = s.makefile("rwb")
            send_envelope(stream, env, raw)
            stream.flush()
            resp_env, resp_tensor = recv_envelope(stream)
        finally:
            s.close()

        assert resp_env.WhichOneof("payload") == "expert_response", (
            f"unexpected payload: {resp_env.WhichOneof('payload')} "
            f"(error detail={resp_env.error.detail if resp_env.WhichOneof('payload') == 'error' else ''})"
        )
        assert list(resp_env.expert_response.expert_ids) == [3, 6, 9]
        assert resp_env.expert_response.layer_idx == 15
        stacked = bytes_to_tensor(
            resp_tensor,
            shape=list(resp_env.expert_response.outputs_spec.shape),
            dtype=resp_env.expert_response.outputs_spec.dtype,
        )
        assert stacked.shape == (1, 2, 3, hidden)
    finally:
        node.shutdown()
        t.join(timeout=3)


@pytest.mark.slow
def test_node_expert_request_wrong_shard_returns_error(
    loaded_model, monkeypatch
) -> None:
    """Requesting an expert id this shard does not host -> Error{ERR_WRONG_SHARD}."""
    monkeypatch.setenv("ENABLE_GOSSIP", "false")
    port = _free_port()
    # Host only (3, 6, 9); client asks for (3, 6, 99) -> 99 is not ours.
    node = _solo_node(loaded_model, port, moe_experts={15: (3, 6, 9)})

    t = threading.Thread(target=node.serve_forever, daemon=True)
    t.start()
    _wait_listening("127.0.0.1", port)

    try:
        s = socket.create_connection(("127.0.0.1", port), timeout=10)
        try:
            hidden = loaded_model.text_model.config.hidden_size
            h = mx.random.normal((1, 2, hidden)).astype(mx.bfloat16)
            mx.eval(h)
            env, raw = _build_expert_request("r2", 15, [3, 6, 99], h)
            stream = s.makefile("rwb")
            send_envelope(stream, env, raw)
            stream.flush()
            resp_env, _ = recv_envelope(stream)
        finally:
            s.close()

        assert resp_env.WhichOneof("payload") == "error", (
            f"expected Error, got {resp_env.WhichOneof('payload')}"
        )
        assert resp_env.error.code == wire_pb2.ERR_WRONG_SHARD, (
            f"expected ERR_WRONG_SHARD, got {resp_env.error.code}"
        )
        assert resp_env.error.request_id == "r2"
    finally:
        node.shutdown()
        t.join(timeout=3)


@pytest.mark.slow
def test_expert_request_handler_replies_error_on_compute_failure(
    loaded_model, monkeypatch
) -> None:
    """If ``run_selected_experts`` raises, the handler must reply with
    ``Error{ERR_SHARD_UNAVAILABLE}`` (with exception detail) rather than
    silently dropping the envelope and making the caller wait for TCP
    timeout. Monkeypatch the moe hook to force a RuntimeError.
    """
    monkeypatch.setenv("ENABLE_GOSSIP", "false")

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    # ``_handle_expert_request`` imports ``run_selected_experts`` lazily from
    # ``model_shard.moe`` every call, so patch it on the source module.
    import model_shard.moe as _moe

    monkeypatch.setattr(_moe, "run_selected_experts", _boom)

    port = _free_port()
    node = _solo_node(loaded_model, port, moe_experts={15: (3, 6, 9)})

    t = threading.Thread(target=node.serve_forever, daemon=True)
    t.start()
    _wait_listening("127.0.0.1", port)

    try:
        s = socket.create_connection(("127.0.0.1", port), timeout=10)
        try:
            hidden = loaded_model.text_model.config.hidden_size
            h = mx.random.normal((1, 2, hidden)).astype(mx.bfloat16)
            mx.eval(h)
            env, raw = _build_expert_request("r3", 15, [3, 6, 9], h)
            stream = s.makefile("rwb")
            send_envelope(stream, env, raw)
            stream.flush()
            resp_env, _ = recv_envelope(stream)
        finally:
            s.close()

        assert resp_env.WhichOneof("payload") == "error", (
            f"expected Error, got {resp_env.WhichOneof('payload')}"
        )
        assert resp_env.error.code == wire_pb2.ERR_SHARD_UNAVAILABLE, (
            f"expected ERR_SHARD_UNAVAILABLE, got {resp_env.error.code}"
        )
        assert resp_env.error.request_id == "r3"
        assert "boom" in resp_env.error.detail
    finally:
        node.shutdown()
        t.join(timeout=3)


@pytest.mark.slow
def test_expert_request_handler_rejects_byte_count_mismatch(
    loaded_model, monkeypatch
) -> None:
    """h_spec.byte_count must match the out-of-band tensor payload length.
    Mismatch -> Error{ERR_INTERNAL}.
    """
    monkeypatch.setenv("ENABLE_GOSSIP", "false")
    port = _free_port()
    node = _solo_node(loaded_model, port, moe_experts={15: (3, 6, 9)})

    t = threading.Thread(target=node.serve_forever, daemon=True)
    t.start()
    _wait_listening("127.0.0.1", port)

    try:
        s = socket.create_connection(("127.0.0.1", port), timeout=10)
        try:
            hidden = loaded_model.text_model.config.hidden_size
            h = mx.random.normal((1, 2, hidden)).astype(mx.bfloat16)
            mx.eval(h)
            env, raw = _build_expert_request("r4", 15, [3, 6, 9], h)
            # Lie about the payload size: real len(raw) is correct, but we
            # declare one more byte. Handler must catch this.
            env.expert_request.h_spec.byte_count = len(raw) + 1
            stream = s.makefile("rwb")
            send_envelope(stream, env, raw)
            stream.flush()
            resp_env, _ = recv_envelope(stream)
        finally:
            s.close()

        assert resp_env.WhichOneof("payload") == "error", (
            f"expected Error, got {resp_env.WhichOneof('payload')}"
        )
        assert resp_env.error.code == wire_pb2.ERR_INTERNAL, (
            f"expected ERR_INTERNAL, got {resp_env.error.code}"
        )
        assert resp_env.error.request_id == "r4"
        assert "byte_count" in resp_env.error.detail
    finally:
        node.shutdown()
        t.join(timeout=3)


@pytest.mark.slow
def test_expert_request_handler_rejects_empty_expert_ids(
    loaded_model, monkeypatch
) -> None:
    """Empty expert_ids is malformed (nothing to stack) -> Error{ERR_INTERNAL}."""
    monkeypatch.setenv("ENABLE_GOSSIP", "false")
    port = _free_port()
    node = _solo_node(loaded_model, port, moe_experts={15: (3, 6, 9)})

    t = threading.Thread(target=node.serve_forever, daemon=True)
    t.start()
    _wait_listening("127.0.0.1", port)

    try:
        s = socket.create_connection(("127.0.0.1", port), timeout=10)
        try:
            hidden = loaded_model.text_model.config.hidden_size
            h = mx.random.normal((1, 2, hidden)).astype(mx.bfloat16)
            mx.eval(h)
            # Empty expert_ids list.
            env, raw = _build_expert_request("r5", 15, [], h)
            stream = s.makefile("rwb")
            send_envelope(stream, env, raw)
            stream.flush()
            resp_env, _ = recv_envelope(stream)
        finally:
            s.close()

        assert resp_env.WhichOneof("payload") == "error", (
            f"expected Error, got {resp_env.WhichOneof('payload')}"
        )
        assert resp_env.error.code == wire_pb2.ERR_INTERNAL, (
            f"expected ERR_INTERNAL, got {resp_env.error.code}"
        )
        assert resp_env.error.request_id == "r5"
        assert "empty expert_ids" in resp_env.error.detail
    finally:
        node.shutdown()
        t.join(timeout=3)
