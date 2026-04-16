"""Thin wrappers around transport framing + protobuf (de)serialization.

Every wire message is a framed `wire_pb2.Envelope` followed by an optional
tensor payload (see transport.py for the framing spec).
"""

from typing import BinaryIO

from model_shard._pb import wire_pb2
from model_shard.transport import read_frame, write_frame


def send_envelope(
    stream: BinaryIO, env: wire_pb2.Envelope, tensor_bytes: bytes = b""
) -> None:
    write_frame(stream, env.SerializeToString(), tensor_bytes)


def recv_envelope(stream: BinaryIO) -> tuple[wire_pb2.Envelope, bytes]:
    msg_bytes, tensor_bytes = read_frame(stream)
    env = wire_pb2.Envelope()
    env.ParseFromString(msg_bytes)
    return env, tensor_bytes
