"""Wire format of the acquisition protocol. See docs/PROTOCOL.md."""

from __future__ import annotations

import struct
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from enum import IntEnum, IntFlag
from typing import Protocol

from epr.core.errors import ChecksumError, ProtocolError

MAGIC = b"EPRF"
PROTOCOL_VERSION = 1

HEADER_FORMAT = "<4sBBHQIII"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

MAX_PAYLOAD_SIZE = 64 * 1024 * 1024
UINT32_MAX = 2**32 - 1


class PacketType(IntEnum):
    METADATA = 0x01
    RGB = 0x02
    NIR = 0x03
    THERMAL = 0x04
    FRAME_SET_END = 0x05
    STATUS = 0x10


class Flags(IntFlag):
    NONE = 0x0000
    LAST_CHUNK = 0x0001


class ByteStream(Protocol):
    def read(self, size: int, /) -> bytes: ...


@dataclass(frozen=True)
class Packet:
    type: PacketType
    frame_set_id: int
    sequence: int
    payload: bytes = b""
    flags: Flags = Flags.LAST_CHUNK

    @property
    def is_last_chunk(self) -> bool:
        return bool(self.flags & Flags.LAST_CHUNK)

    def encode(self) -> bytes:
        if len(self.payload) > MAX_PAYLOAD_SIZE:
            raise ProtocolError(f"payload of {len(self.payload)} bytes exceeds the maximum")
        header = struct.pack(
            HEADER_FORMAT,
            MAGIC,
            PROTOCOL_VERSION,
            int(self.type),
            int(self.flags),
            self.frame_set_id,
            self.sequence,
            len(self.payload),
            zlib.crc32(self.payload),
        )
        return header + self.payload


def read_packet(stream: ByteStream) -> Packet | None:
    """Read one packet, or return None at a clean end of stream."""
    header = _read_exactly(stream, HEADER_SIZE, allow_empty=True)
    if header is None:
        return None

    magic, version, packet_type, flags, frame_set_id, sequence, length, crc = struct.unpack(
        HEADER_FORMAT, header
    )
    if magic != MAGIC:
        raise ProtocolError(f"expected magic {MAGIC!r}, got {magic!r}")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol version {version}")
    try:
        packet_type = PacketType(packet_type)
    except ValueError as exc:
        raise ProtocolError(f"unknown packet type 0x{packet_type:02x}") from exc
    if length > MAX_PAYLOAD_SIZE:
        raise ProtocolError(f"declared payload of {length} bytes exceeds the maximum")

    payload = _read_exactly(stream, length) if length else b""
    if zlib.crc32(payload) != crc:
        raise ChecksumError(
            f"payload CRC mismatch in {packet_type.name} packet, sequence {sequence}"
        )
    return Packet(
        type=packet_type,
        frame_set_id=frame_set_id,
        sequence=sequence,
        payload=payload,
        flags=Flags(flags),
    )


def iter_packets(stream: ByteStream) -> Iterator[Packet]:
    while (packet := read_packet(stream)) is not None:
        yield packet


def _read_exactly(stream: ByteStream, size: int, *, allow_empty: bool = False) -> bytes | None:
    buffer = bytearray()
    while len(buffer) < size:
        chunk = stream.read(size - len(buffer))
        if not chunk:
            if not buffer and allow_empty:
                return None
            raise ProtocolError(f"stream ended after {len(buffer)} of {size} bytes")
        buffer.extend(chunk)
    return bytes(buffer)
