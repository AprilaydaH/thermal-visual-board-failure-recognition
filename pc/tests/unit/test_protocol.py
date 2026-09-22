import io
import struct

import pytest

from epr.core.errors import ChecksumError, ProtocolError
from epr.device.protocol import (
    HEADER_SIZE,
    MAGIC,
    Flags,
    FrameSetEncoder,
    Packet,
    PacketType,
    iter_packets,
    read_packet,
)
from epr.device.simulator import SimulatedDevice


def test_header_is_28_bytes():
    assert HEADER_SIZE == 28


def test_packet_round_trip():
    packet = Packet(
        type=PacketType.THERMAL,
        frame_set_id=2**40,
        sequence=7,
        payload=b"raw thermal",
        flags=Flags.LAST_CHUNK,
    )
    decoded = read_packet(io.BytesIO(packet.encode()))
    assert decoded == packet


def test_empty_stream_returns_none():
    assert read_packet(io.BytesIO(b"")) is None


def test_wrong_magic_is_rejected():
    encoded = bytearray(Packet(PacketType.STATUS, 1, 0, b"{}").encode())
    encoded[0:4] = b"XXXX"
    with pytest.raises(ProtocolError, match="magic"):
        read_packet(io.BytesIO(bytes(encoded)))


def test_unsupported_version_is_rejected():
    encoded = bytearray(Packet(PacketType.STATUS, 1, 0, b"{}").encode())
    encoded[4] = 99
    with pytest.raises(ProtocolError, match="version"):
        read_packet(io.BytesIO(bytes(encoded)))


def test_corrupted_payload_is_detected():
    encoded = bytearray(Packet(PacketType.RGB, 1, 0, b"pixels").encode())
    encoded[-1] ^= 0xFF
    with pytest.raises(ChecksumError):
        read_packet(io.BytesIO(bytes(encoded)))


def test_truncated_payload_is_detected():
    encoded = Packet(PacketType.RGB, 1, 0, b"pixels").encode()
    with pytest.raises(ProtocolError, match="stream ended"):
        read_packet(io.BytesIO(encoded[:-2]))


def test_declared_length_beyond_maximum_is_rejected():
    header = struct.pack("<4sBBHQIII", MAGIC, 1, PacketType.RGB, 0, 1, 0, 2**31, 0)
    with pytest.raises(ProtocolError, match="exceeds the maximum"):
        read_packet(io.BytesIO(header))


def test_large_channels_are_chunked_with_continuous_sequence():
    device = SimulatedDevice(rgb_size=(64, 48), thermal_size=(16, 12))
    encoder = FrameSetEncoder(chunk_size=1024)
    packets = list(encoder.encode(device.capture()))

    assert [packet.sequence for packet in packets] == list(range(len(packets)))
    assert packets[0].type is PacketType.METADATA
    assert packets[-1].type is PacketType.FRAME_SET_END

    rgb_packets = [packet for packet in packets if packet.type is PacketType.RGB]
    assert len(rgb_packets) == 9  # 64 * 48 * 3 bytes in 1024-byte chunks
    assert [packet.is_last_chunk for packet in rgb_packets[:-1]] == [False] * 8
    assert rgb_packets[-1].is_last_chunk


def test_sequence_continues_across_frame_sets():
    device = SimulatedDevice(rgb_size=(32, 24), thermal_size=(8, 6))
    encoder = FrameSetEncoder()
    stream = io.BytesIO()
    for frame_set in device.capture_many(3):
        for chunk in encoder.encode_bytes(frame_set):
            stream.write(chunk)
    stream.seek(0)

    sequences = [packet.sequence for packet in iter_packets(stream)]
    assert sequences == list(range(len(sequences)))
