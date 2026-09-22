"""Turns a frame set into the packet sequence defined in docs/PROTOCOL.md."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from epr.core.domain_models.frame_set import Channel, FrameSet
from epr.device.protocol.packets import UINT32_MAX, Flags, Packet, PacketType

CHANNEL_PACKET_TYPE = {
    Channel.RGB: PacketType.RGB,
    Channel.NIR: PacketType.NIR,
    Channel.THERMAL: PacketType.THERMAL,
}

DEFAULT_CHUNK_SIZE = 256 * 1024


class FrameSetEncoder:
    """Stateful encoder: the sequence number runs across the whole connection."""

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = chunk_size
        self._sequence = 0

    @property
    def sequence(self) -> int:
        return self._sequence

    def encode(self, frame_set: FrameSet) -> Iterator[Packet]:
        frame_set_id = frame_set.frame_set_id
        metadata = frame_set.metadata.model_dump_json().encode("utf-8")
        yield self._packet(PacketType.METADATA, frame_set_id, metadata)

        for channel, packet_type in CHANNEL_PACKET_TYPE.items():
            payload = np.ascontiguousarray(frame_set.array(channel)).tobytes()
            for offset in range(0, len(payload), self.chunk_size):
                chunk = payload[offset : offset + self.chunk_size]
                last = offset + self.chunk_size >= len(payload)
                yield self._packet(packet_type, frame_set_id, chunk, last=last)

        yield self._packet(PacketType.FRAME_SET_END, frame_set_id, b"")

    def encode_bytes(self, frame_set: FrameSet) -> Iterator[bytes]:
        for packet in self.encode(frame_set):
            yield packet.encode()

    def _packet(
        self,
        packet_type: PacketType,
        frame_set_id: int,
        payload: bytes,
        *,
        last: bool = True,
    ) -> Packet:
        packet = Packet(
            type=packet_type,
            frame_set_id=frame_set_id,
            sequence=self._sequence,
            payload=payload,
            flags=Flags.LAST_CHUNK if last else Flags.NONE,
        )
        self._sequence = (self._sequence + 1) % (UINT32_MAX + 1)
        return packet
