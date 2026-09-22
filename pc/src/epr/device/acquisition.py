"""Reassembles frame sets from the packet stream.

Sequence gaps, mismatched IDs and missing channels discard the frame set being assembled.
Nothing is reconstructed from partial data.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
from pydantic import ValidationError

from epr.core.domain_models.frame_set import Channel, FrameSet, FrameSetMetadata
from epr.device.protocol.framing import CHANNEL_PACKET_TYPE
from epr.device.protocol.packets import UINT32_MAX, ByteStream, Packet, PacketType, iter_packets

logger = logging.getLogger(__name__)

PACKET_TYPE_CHANNEL = {packet_type: channel for channel, packet_type in CHANNEL_PACKET_TYPE.items()}


@dataclass
class AcquisitionStats:
    completed_frame_sets: int = 0
    rejected_frame_sets: int = 0
    dropped_packets: int = 0
    orphan_packets: int = 0


@dataclass
class _Partial:
    metadata: FrameSetMetadata
    chunks: dict[Channel, list[bytes]] = field(default_factory=dict)
    complete: set[Channel] = field(default_factory=set)


class FrameSetAssembler:
    def __init__(self) -> None:
        self.stats = AcquisitionStats()
        self._expected_sequence: int | None = None
        self._partial: _Partial | None = None

    def push(self, packet: Packet) -> FrameSet | None:
        """Feed one validated packet. Returns a frame set when one is complete."""
        self._check_sequence(packet)

        if packet.type is PacketType.METADATA:
            self._begin(packet)
            return None
        if packet.type in PACKET_TYPE_CHANNEL:
            self._append(packet)
            return None
        if packet.type is PacketType.FRAME_SET_END:
            return self._finish(packet)

        logger.debug("ignoring %s packet", packet.type.name)
        return None

    def _check_sequence(self, packet: Packet) -> None:
        expected = self._expected_sequence
        if expected is not None and packet.sequence != expected:
            gap = (packet.sequence - expected) % (UINT32_MAX + 1)
            self.stats.dropped_packets += gap
            self._reject(f"sequence gap of {gap} packets before sequence {packet.sequence}")
        self._expected_sequence = (packet.sequence + 1) % (UINT32_MAX + 1)

    def _begin(self, packet: Packet) -> None:
        if self._partial is not None:
            self._reject("metadata arrived before the previous frame set ended")
        try:
            metadata = FrameSetMetadata.model_validate_json(packet.payload)
        except ValidationError as exc:
            self.stats.rejected_frame_sets += 1
            logger.warning("frame set %d rejected: invalid metadata: %s", packet.frame_set_id, exc)
            return
        if metadata.frame_set_id != packet.frame_set_id:
            self.stats.rejected_frame_sets += 1
            logger.warning(
                "frame set rejected: header id %d does not match metadata id %d",
                packet.frame_set_id,
                metadata.frame_set_id,
            )
            return
        self._partial = _Partial(metadata=metadata)

    def _append(self, packet: Packet) -> None:
        partial = self._partial
        if partial is None:
            self.stats.orphan_packets += 1
            logger.debug("%s packet without metadata", packet.type.name)
            return
        if packet.frame_set_id != partial.metadata.frame_set_id:
            self._reject(f"channel packet belongs to frame set {packet.frame_set_id}")
            return

        channel = PACKET_TYPE_CHANNEL[packet.type]
        partial.chunks.setdefault(channel, []).append(packet.payload)
        if packet.is_last_chunk:
            partial.complete.add(channel)

    def _finish(self, packet: Packet) -> FrameSet | None:
        partial = self._partial
        if partial is None:
            self.stats.orphan_packets += 1
            return None
        self._partial = None

        if packet.frame_set_id != partial.metadata.frame_set_id:
            self.stats.rejected_frame_sets += 1
            logger.warning("frame set end does not match the frame set being assembled")
            return None

        missing = [channel.value for channel in Channel if channel not in partial.complete]
        if missing:
            self.stats.rejected_frame_sets += 1
            logger.warning(
                "frame set %d rejected: incomplete channels %s",
                partial.metadata.frame_set_id,
                ", ".join(missing),
            )
            return None

        arrays: dict[str, np.ndarray] = {}
        for channel in Channel:
            descriptor = partial.metadata.descriptor(channel)
            payload = b"".join(partial.chunks[channel])
            if len(payload) != descriptor.byte_count:
                self.stats.rejected_frame_sets += 1
                logger.warning(
                    "frame set %d rejected: %s channel has %d bytes, expected %d",
                    partial.metadata.frame_set_id,
                    channel.value,
                    len(payload),
                    descriptor.byte_count,
                )
                return None
            arrays[channel.value] = np.frombuffer(payload, dtype=descriptor.dtype).reshape(
                descriptor.shape
            )

        self.stats.completed_frame_sets += 1
        return FrameSet(metadata=partial.metadata, **arrays)

    def _reject(self, reason: str) -> None:
        if self._partial is None:
            return
        self.stats.rejected_frame_sets += 1
        logger.warning(
            "frame set %d rejected: %s", self._partial.metadata.frame_set_id, reason
        )
        self._partial = None


def assemble_stream(
    stream: ByteStream, assembler: FrameSetAssembler | None = None
) -> Iterator[FrameSet]:
    """Read a byte stream and yield every complete frame set."""
    assembler = assembler if assembler is not None else FrameSetAssembler()
    for packet in iter_packets(stream):
        frame_set = assembler.push(packet)
        if frame_set is not None:
            yield frame_set
