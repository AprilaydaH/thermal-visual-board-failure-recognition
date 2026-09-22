"""Simulated acquisition head.

Produces synthetic but structurally correct frame sets so the PC software can be developed
while the physical gadget is unavailable or being modified. One component heats up over
time, which gives the later thermal and LNN work a usable temperature trend.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from epr.core.domain_models.frame_set import (
    Environment,
    FrameSet,
    FrameSetMetadata,
    Geometry,
    IlluminationState,
    ImageDescriptor,
    NirChannel,
    PixelFormat,
    RgbChannel,
    SensorState,
    SensorStatus,
    ShutterState,
    ThermalCalibrationMode,
    ThermalChannel,
)
from epr.core.domain_models.region import BoundingBox, ComponentRegion, RegionSource
from epr.device.protocol.framing import DEFAULT_CHUNK_SIZE, FrameSetEncoder

KELVIN_OFFSET = 273.15
CENTIKELVIN = 100.0


@dataclass(frozen=True)
class SimulatedComponent:
    """Position is normalized to the board image, so any sensor size can be simulated."""

    name: str
    package: str
    x: float
    y: float
    width: float
    height: float
    body_gray: int
    nir_gray: int
    steady_rise_c: float = 0.0
    tau_s: float = 25.0


DEFAULT_COMPONENTS = (
    SimulatedComponent("U1", "QFP64", 0.34, 0.30, 0.22, 0.24, 38, 120, 26.0, 18.0),
    SimulatedComponent("U2", "SOIC8", 0.66, 0.22, 0.12, 0.08, 44, 128, 6.0, 30.0),
    SimulatedComponent("Q1", "SOT23", 0.20, 0.66, 0.06, 0.05, 40, 118, 11.0, 9.0),
    SimulatedComponent("C1", "0805", 0.56, 0.62, 0.05, 0.03, 150, 176, 0.0),
    SimulatedComponent("R1", "0805", 0.66, 0.62, 0.05, 0.03, 30, 96, 1.5, 40.0),
    SimulatedComponent("J1", "HEADER", 0.12, 0.16, 0.09, 0.14, 20, 70, 0.0),
)

BOARD_RGB = (26, 92, 48)
BOARD_NIR = 60


class SimulatedDevice:
    def __init__(
        self,
        *,
        device_id: str = "SIM-0001",
        session_id: str = "sim-session",
        seed: int = 0,
        rgb_size: tuple[int, int] = (1280, 960),
        thermal_size: tuple[int, int] = (160, 120),
        frame_interval_s: float = 1.0,
        ambient_temperature_c: float = 22.0,
        distance_mm: float = 180.0,
        components: tuple[SimulatedComponent, ...] = DEFAULT_COMPONENTS,
    ) -> None:
        self.device_id = device_id
        self.session_id = session_id
        self.rgb_size = rgb_size
        self.thermal_size = thermal_size
        self.frame_interval_s = frame_interval_s
        self.ambient_temperature_c = ambient_temperature_c
        self.distance_mm = distance_mm
        self.components = components
        self._rng = np.random.default_rng(seed)
        self._frame_index = 0

    @property
    def elapsed_s(self) -> float:
        return self._frame_index * self.frame_interval_s

    def capture(self) -> FrameSet:
        index = self._frame_index
        self._frame_index += 1
        elapsed_s = index * self.frame_interval_s

        rgb = self._render_rgb()
        nir = self._render_nir()
        thermal = self._render_thermal(elapsed_s)
        metadata = self._metadata(index, elapsed_s, thermal)
        return FrameSet(metadata=metadata, rgb=rgb, nir=nir, thermal=thermal)

    def component_regions(self) -> tuple[ComponentRegion, ...]:
        """Ground-truth regions, which stand in for the detector until it exists."""
        return tuple(
            ComponentRegion(
                region_id=component.name,
                box=BoundingBox(
                    x=component.x,
                    y=component.y,
                    width=component.width,
                    height=component.height,
                ),
                source=RegionSource.SIMULATED,
                label=component.name,
                package=component.package,
            )
            for component in self.components
        )

    def capture_many(self, count: int) -> Iterator[FrameSet]:
        for _ in range(count):
            yield self.capture()

    def stream(self, count: int, chunk_size: int = DEFAULT_CHUNK_SIZE) -> Iterator[bytes]:
        """Encoded packet bytes, as the device would put them on the wire."""
        encoder = FrameSetEncoder(chunk_size=chunk_size)
        for frame_set in self.capture_many(count):
            yield from encoder.encode_bytes(frame_set)

    def _render_rgb(self) -> np.ndarray:
        width, height = self.rgb_size
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[:, :] = BOARD_RGB
        for component in self.components:
            rows, columns = self._region(component, self.rgb_size)
            image[rows, columns] = component.body_gray
            marking = slice(rows.start, rows.start + max(1, (rows.stop - rows.start) // 3))
            image[marking, columns] = min(255, component.body_gray + 70)
        return self._add_noise(image, 4)

    def _render_nir(self) -> np.ndarray:
        width, height = self.rgb_size
        image = np.full((height, width), BOARD_NIR, dtype=np.uint8)
        for component in self.components:
            rows, columns = self._region(component, self.rgb_size)
            image[rows, columns] = component.nir_gray
        return self._add_noise(image, 3)

    def _render_thermal(self, elapsed_s: float) -> np.ndarray:
        width, height = self.thermal_size
        temperature = np.full((height, width), self.ambient_temperature_c, dtype=np.float64)
        rows, columns = np.mgrid[0:height, 0:width]

        for component in self.components:
            if component.steady_rise_c <= 0:
                continue
            centre_x = (component.x + component.width / 2) * width
            centre_y = (component.y + component.height / 2) * height
            sigma = max(component.width * width, component.height * height) / 2
            rise = component.steady_rise_c * (1 - math.exp(-elapsed_s / component.tau_s))
            distance_sq = (columns - centre_x) ** 2 + (rows - centre_y) ** 2
            temperature += rise * np.exp(-distance_sq / (2 * sigma**2))

        temperature += self._rng.normal(0.0, 0.08, temperature.shape)
        centikelvin = np.round((temperature + KELVIN_OFFSET) * CENTIKELVIN)
        return np.clip(centikelvin, 0, 65535).astype("<u2")

    def _metadata(self, index: int, elapsed_s: float, thermal: np.ndarray) -> FrameSetMetadata:
        width, height = self.rgb_size
        thermal_width, thermal_height = self.thermal_size
        sensor_temperature_c = float(thermal.max()) / CENTIKELVIN - KELVIN_OFFSET

        return FrameSetMetadata(
            device_id=self.device_id,
            session_id=self.session_id,
            frame_set_id=index,
            timestamp_ns=int(elapsed_s * 1e9),
            rgb=RgbChannel(
                image=ImageDescriptor(
                    width=width, height=height, pixel_format=PixelFormat.RGB888
                ),
                exposure_us=8000,
                gain_db=2.0,
                focus_position=512,
                white_balance_k=5200,
            ),
            nir=NirChannel(
                image=ImageDescriptor(
                    width=width, height=height, pixel_format=PixelFormat.MONO8
                ),
                wavelength_nm=850,
                led_current_ma=120.0,
                exposure_us=12000,
                gain_db=4.0,
            ),
            thermal=ThermalChannel(
                image=ImageDescriptor(
                    width=thermal_width,
                    height=thermal_height,
                    pixel_format=PixelFormat.RAW16,
                ),
                calibration_mode=ThermalCalibrationMode.RADIOMETRIC,
                shutter_state=ShutterState.OPEN,
                sensor_temperature_c=round(sensor_temperature_c, 2),
            ),
            geometry=Geometry(
                distance_mm=self.distance_mm,
                calibration_profile_id="sim-profile-1",
            ),
            environment=Environment(
                ambient_temperature_c=self.ambient_temperature_c,
                humidity_percent=41.0,
                illumination_state=IlluminationState.WHITE,
            ),
            calibration_id="sim-calibration-1",
            sensor_status=SensorStatus(
                rgb=SensorState.OK,
                nir=SensorState.OK,
                thermal=SensorState.OK,
                distance=SensorState.OK,
                environment=SensorState.OK,
            ),
        )

    def _region(
        self, component: SimulatedComponent, size: tuple[int, int]
    ) -> tuple[slice, slice]:
        width, height = size
        x0 = int(round(component.x * width))
        y0 = int(round(component.y * height))
        x1 = max(x0 + 1, int(round((component.x + component.width) * width)))
        y1 = max(y0 + 1, int(round((component.y + component.height) * height)))
        return slice(y0, y1), slice(x0, x1)

    def _add_noise(self, image: np.ndarray, amplitude: int) -> np.ndarray:
        noise = self._rng.integers(-amplitude, amplitude + 1, size=image.shape)
        return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
