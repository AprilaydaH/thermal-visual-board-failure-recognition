"""Map a point on the RGB image onto the thermal frame.

G1 uses normalized coordinates and treats the board as filling the same rectangle in every
channel. That is true of the simulator and of a perfectly aligned head. It is *not* true of
the real optics — different FOV and a centimetre baseline — which is why gate G3 exists.

Callers stay in normalized space so G3 can replace the identity with a measured homography
without changing the viewer or the probe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from epr.core.domain_models.frame_set import Channel, FrameSet
from epr.core.domain_models.region import BoundingBox
from epr.processing.thermal.metrics import component_metrics, raw_to_celsius


@dataclass(frozen=True)
class Registration:
    """RGB → thermal map in normalized coordinates.

    ``None`` means identity. A 3×3 homography, when G3 measures one, goes here.
    """

    rgb_to_thermal: np.ndarray | None = None

    def to_thermal(self, x: float, y: float) -> tuple[float, float]:
        _require_normalized(x, y)
        if self.rgb_to_thermal is None:
            return x, y
        matrix = np.asarray(self.rgb_to_thermal, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError("homography must be 3x3")
        wx, wy, w = matrix @ np.array([x, y, 1.0])
        if abs(w) < 1e-12:
            raise ValueError("homography sent the point to infinity")
        return float(wx / w), float(wy / w)


@dataclass(frozen=True)
class ThermalProbe:
    """What the G1 viewer shows after a click on the RGB frame."""

    rgb_xy: tuple[float, float]
    thermal_xy: tuple[float, float]
    rgb_pixel: tuple[int, int]
    thermal_pixel: tuple[int, int]
    temperature_c: float
    region_mean_c: float
    region_max_c: float
    region: BoundingBox


def point_to_pixel(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    _require_normalized(x, y)
    if width <= 0 or height <= 0:
        raise ValueError("image size must be positive")
    return min(int(x * width), width - 1), min(int(y * height), height - 1)


def box_around(x: float, y: float, size: float = 0.05) -> BoundingBox:
    """A small normalized box around a click, clipped to the frame."""
    _require_normalized(x, y)
    if size <= 0 or size > 1:
        raise ValueError("region size must be in (0, 1]")
    half = size / 2
    x0 = min(max(0.0, x - half), 1.0 - 1e-6)
    y0 = min(max(0.0, y - half), 1.0 - 1e-6)
    width = min(size, 1.0 - x0)
    height = min(size, 1.0 - y0)
    return BoundingBox(x=x0, y=y0, width=max(width, 1e-6), height=max(height, 1e-6))


def probe(
    frame_set: FrameSet,
    x: float,
    y: float,
    *,
    registration: Registration | None = None,
    region_size: float = 0.05,
) -> ThermalProbe:
    """Temperature at the thermal pixel that corresponds to an RGB click."""
    registration = registration or Registration()
    tx, ty = registration.to_thermal(x, y)
    tx = min(max(tx, 0.0), 1.0)
    ty = min(max(ty, 0.0), 1.0)

    rgb_h, rgb_w = frame_set.array(Channel.RGB).shape[:2]
    th_h, th_w = frame_set.array(Channel.THERMAL).shape[:2]
    rgb_pixel = point_to_pixel(x, y, rgb_w, rgb_h)
    thermal_pixel = point_to_pixel(tx, ty, th_w, th_h)

    celsius = raw_to_celsius(frame_set.array(Channel.THERMAL))
    temperature = float(celsius[thermal_pixel[1], thermal_pixel[0]])
    region = box_around(tx, ty, region_size)
    metrics = component_metrics(frame_set, region)

    return ThermalProbe(
        rgb_xy=(x, y),
        thermal_xy=(tx, ty),
        rgb_pixel=rgb_pixel,
        thermal_pixel=thermal_pixel,
        temperature_c=temperature,
        region_mean_c=metrics.t_mean_c,
        region_max_c=metrics.t_max_c,
        region=region,
    )


def _require_normalized(x: float, y: float) -> None:
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise ValueError(f"normalized coordinates must be in [0, 1], got {x}, {y}")
