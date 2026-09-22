"""End to end: simulated gadget -> packets -> acquisition -> raw recording."""

import dataclasses
import stat

import numpy as np
import pytest
import tifffile

from epr.core.domain_models import Channel
from epr.core.errors import StorageError
from epr.device.acquisition import FrameSetAssembler, assemble_stream
from epr.device.protocol import FrameSetEncoder, PacketType
from epr.device.simulator import SimulatedDevice
from epr.device.transport import ChunkedByteStream
from epr.storage.file_storage import RawRecorder

SMALL = {"rgb_size": (64, 48), "thermal_size": (16, 12)}


def make_device(**overrides):
    return SimulatedDevice(**SMALL, **overrides)


def encode(frame_sets, chunk_size=1024):
    encoder = FrameSetEncoder(chunk_size=chunk_size)
    packets = []
    for frame_set in frame_sets:
        packets.extend(encoder.encode(frame_set))
    return packets


def test_frame_sets_survive_the_byte_stream_unchanged():
    expected = list(make_device().capture_many(3))
    stream = ChunkedByteStream(make_device().stream(3, chunk_size=1024))
    assembler = FrameSetAssembler()

    received = list(assemble_stream(stream, assembler))

    assert len(received) == 3
    assert assembler.stats.completed_frame_sets == 3
    assert assembler.stats.rejected_frame_sets == 0
    assert assembler.stats.dropped_packets == 0
    for original, decoded in zip(expected, received, strict=True):
        assert decoded.metadata == original.metadata
        for channel in Channel:
            np.testing.assert_array_equal(decoded.array(channel), original.array(channel))


def test_heating_trend_is_visible_across_frame_sets():
    device = make_device(frame_interval_s=5.0)
    hotspots = [frame_set.thermal.max() for frame_set in device.capture_many(4)]
    assert hotspots == sorted(hotspots)
    assert hotspots[-1] > hotspots[0]


def test_dropped_packet_discards_only_the_affected_frame_set():
    packets = encode(make_device().capture_many(2))
    first_rgb = next(i for i, p in enumerate(packets) if p.type is PacketType.RGB)
    assembler = FrameSetAssembler()

    completed = [
        result
        for i, packet in enumerate(packets)
        if i != first_rgb and (result := assembler.push(packet)) is not None
    ]

    assert len(completed) == 1
    assert completed[0].frame_set_id == 1
    assert assembler.stats.dropped_packets == 1
    assert assembler.stats.rejected_frame_sets == 1


def test_frame_set_without_all_channels_is_rejected():
    packets = [p for p in encode(make_device().capture_many(1)) if p.type is not PacketType.NIR]
    packets = [dataclasses.replace(p, sequence=i) for i, p in enumerate(packets)]
    assembler = FrameSetAssembler()

    assert [assembler.push(packet) for packet in packets] == [None] * len(packets)
    assert assembler.stats.completed_frame_sets == 0
    assert assembler.stats.rejected_frame_sets == 1


def test_truncated_channel_is_rejected():
    packets = encode(make_device().capture_many(1))
    index = next(i for i, p in enumerate(packets) if p.type is PacketType.THERMAL)
    packets[index] = dataclasses.replace(packets[index], payload=packets[index].payload[:-2])
    assembler = FrameSetAssembler()

    for packet in packets:
        assert assembler.push(packet) is None
    assert assembler.stats.rejected_frame_sets == 1


def test_raw_recording_is_immutable_and_lossless(tmp_path):
    frame_set = make_device().capture()
    recorder = RawRecorder(tmp_path / "demo")

    paths = recorder.record(frame_set)

    for path in paths:
        assert path.is_file()
        assert not path.stat().st_mode & stat.S_IWRITE

    np.testing.assert_array_equal(tifffile.imread(paths.thermal), frame_set.thermal)
    assert '"frame_set_id":' in paths.metadata.read_text(encoding="utf-8").replace(" ", "")

    with pytest.raises(StorageError, match="never overwritten"):
        recorder.record(frame_set)
