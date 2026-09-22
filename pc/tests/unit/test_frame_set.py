import numpy as np
import pytest
from pydantic import ValidationError

from epr.core.domain_models import (
    Channel,
    FrameSet,
    FrameSetMetadata,
    ImageDescriptor,
    PixelFormat,
)
from epr.core.errors import IncompleteFrameSetError
from epr.device.simulator import SimulatedDevice


@pytest.fixture
def frame_set():
    device = SimulatedDevice(rgb_size=(64, 48), thermal_size=(16, 12))
    return device.capture()


@pytest.mark.parametrize(
    ("pixel_format", "shape", "byte_count"),
    [
        (PixelFormat.RGB888, (48, 64, 3), 48 * 64 * 3),
        (PixelFormat.MONO8, (48, 64), 48 * 64),
        (PixelFormat.RAW16, (48, 64), 48 * 64 * 2),
    ],
)
def test_descriptor_geometry(pixel_format, shape, byte_count):
    descriptor = ImageDescriptor(width=64, height=48, pixel_format=pixel_format)
    assert descriptor.shape == shape
    assert descriptor.byte_count == byte_count


def test_channels_match_their_descriptors(frame_set):
    for channel in Channel:
        descriptor = frame_set.metadata.descriptor(channel)
        assert frame_set.array(channel).shape == descriptor.shape
        assert frame_set.array(channel).dtype == descriptor.dtype


def test_thermal_is_raw_radiometric(frame_set):
    assert frame_set.thermal.dtype == np.dtype("<u2")
    hotspot_c = frame_set.thermal.max() / 100.0 - 273.15
    assert 20.0 < hotspot_c < 80.0


def test_mismatched_channel_is_rejected(frame_set):
    with pytest.raises(IncompleteFrameSetError, match="rgb channel"):
        FrameSet(
            metadata=frame_set.metadata,
            rgb=np.zeros((2, 2, 3), dtype=np.uint8),
            nir=frame_set.nir,
            thermal=frame_set.thermal,
        )


def test_missing_channel_is_not_guessed(frame_set):
    payload = frame_set.metadata.model_dump()
    del payload["thermal"]
    with pytest.raises(ValidationError):
        FrameSetMetadata.model_validate(payload)


def test_unknown_metadata_field_is_rejected(frame_set):
    payload = frame_set.metadata.model_dump()
    payload["swir"] = {"image": None}
    with pytest.raises(ValidationError):
        FrameSetMetadata.model_validate(payload)


def test_metadata_round_trips_through_json(frame_set):
    restored = FrameSetMetadata.model_validate_json(frame_set.metadata.model_dump_json())
    assert restored == frame_set.metadata
