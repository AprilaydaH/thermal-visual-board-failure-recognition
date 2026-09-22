"""Frame-set domain model.

A frame set is the basic object of the system: RGB + NIR + raw thermal captured together
under one ``frame_set_id``, with the metadata needed to interpret them. Incomplete or
mismatched frame sets are rejected; missing channels are never guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from epr.core.errors import IncompleteFrameSetError

UINT64_MAX = 2**64 - 1


class Channel(str, Enum):
    RGB = "rgb"
    NIR = "nir"
    THERMAL = "thermal"


class PixelFormat(str, Enum):
    RGB888 = "rgb888"
    MONO8 = "mono8"
    RAW16 = "raw16"


class SensorState(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"
    FAULT = "fault"


class ShutterState(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    UNKNOWN = "unknown"


class ThermalCalibrationMode(str, Enum):
    RADIOMETRIC = "radiometric"
    NON_RADIOMETRIC = "non_radiometric"


class IlluminationState(str, Enum):
    OFF = "off"
    WHITE = "white"
    NIR_850 = "nir_850"


_PIXEL_LAYOUT = {
    PixelFormat.RGB888: (3, "|u1"),
    PixelFormat.MONO8: (1, "|u1"),
    PixelFormat.RAW16: (1, "<u2"),
}


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ImageDescriptor(_Model):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    pixel_format: PixelFormat

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(_PIXEL_LAYOUT[self.pixel_format][1])

    @property
    def shape(self) -> tuple[int, ...]:
        channels = _PIXEL_LAYOUT[self.pixel_format][0]
        if channels == 1:
            return (self.height, self.width)
        return (self.height, self.width, channels)

    @property
    def byte_count(self) -> int:
        return int(np.prod(self.shape)) * self.dtype.itemsize


class RgbChannel(_Model):
    image: ImageDescriptor
    exposure_us: int = Field(gt=0)
    gain_db: float = Field(ge=0)
    focus_position: int = Field(ge=0)
    white_balance_k: int = Field(gt=0)


class NirChannel(_Model):
    image: ImageDescriptor
    wavelength_nm: int = Field(gt=0)
    led_current_ma: float = Field(ge=0)
    exposure_us: int = Field(gt=0)
    gain_db: float = Field(ge=0)


class ThermalChannel(_Model):
    image: ImageDescriptor
    calibration_mode: ThermalCalibrationMode
    shutter_state: ShutterState
    sensor_temperature_c: float


class Geometry(_Model):
    distance_mm: float = Field(gt=0)
    calibration_profile_id: str = Field(min_length=1)
    camera_pose: tuple[float, float, float, float, float, float] | None = None


class Environment(_Model):
    ambient_temperature_c: float
    humidity_percent: float = Field(ge=0, le=100)
    illumination_state: IlluminationState


class SensorStatus(_Model):
    rgb: SensorState
    nir: SensorState
    thermal: SensorState
    distance: SensorState
    environment: SensorState

    @property
    def all_ok(self) -> bool:
        return all(
            state is SensorState.OK
            for state in (self.rgb, self.nir, self.thermal, self.distance, self.environment)
        )


class FrameSetMetadata(_Model):
    device_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    frame_set_id: int = Field(ge=0, le=UINT64_MAX)
    timestamp_ns: int = Field(ge=0)
    rgb: RgbChannel
    nir: NirChannel
    thermal: ThermalChannel
    geometry: Geometry
    environment: Environment
    calibration_id: str = Field(min_length=1)
    sensor_status: SensorStatus

    def descriptor(self, channel: Channel) -> ImageDescriptor:
        return getattr(self, channel.value).image


@dataclass(frozen=True)
class FrameSet:
    """Metadata plus the three raw channel arrays, validated against their descriptors."""

    metadata: FrameSetMetadata
    rgb: np.ndarray
    nir: np.ndarray
    thermal: np.ndarray

    def __post_init__(self) -> None:
        for channel in Channel:
            descriptor = self.metadata.descriptor(channel)
            array = self.array(channel)
            if array.shape != descriptor.shape or array.dtype != descriptor.dtype:
                raise IncompleteFrameSetError(
                    f"{channel.value} channel is {array.shape}/{array.dtype}, "
                    f"metadata declares {descriptor.shape}/{descriptor.dtype}"
                )

    def array(self, channel: Channel) -> np.ndarray:
        return getattr(self, channel.value)

    @property
    def frame_set_id(self) -> int:
        return self.metadata.frame_set_id
