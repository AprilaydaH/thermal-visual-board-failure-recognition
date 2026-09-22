"""Turning pixels into millimetres.

The frame set reports the distance to the board, so a detected box can be expressed as a
physical size. That is what makes a package lookup possible at all.

Two models are offered. A calibrated scale, measured once against a target at a known
distance, is the one to trust on a real rig. The thin-lens model is for when only the lens
data sheet is available.

Both assume the board is flat, perpendicular to the optical axis and at the reported distance,
and that lens distortion has already been corrected. On a stand-mounted macro rig those hold
well near the centre of the frame and degrade towards the corners; a tilted board breaks them,
and that error grows with the tangent of the tilt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from epr.core.domain_models.region import BoundingBox


class Scale(Protocol):
    def mm_per_pixel(self, distance_mm: float) -> float: ...


@dataclass(frozen=True)
class ThinLensOptics:
    """Scale from lens and sensor geometry.

    At macro distances the object is only a few focal lengths away, where the far-field
    approximation ``distance / focal_length`` overstates the scale noticeably. The thin-lens
    magnification ``focal_length / (distance - focal_length)`` is used instead.
    """

    focal_length_mm: float
    sensor_width_mm: float
    image_width_px: int

    def __post_init__(self) -> None:
        if self.focal_length_mm <= 0 or self.sensor_width_mm <= 0:
            raise ValueError("focal length and sensor width must be positive")
        if self.image_width_px <= 0:
            raise ValueError("image width must be positive")

    @property
    def pixel_pitch_mm(self) -> float:
        return self.sensor_width_mm / self.image_width_px

    def mm_per_pixel(self, distance_mm: float) -> float:
        if distance_mm <= self.focal_length_mm:
            raise ValueError("distance must exceed the focal length")
        magnification = self.focal_length_mm / (distance_mm - self.focal_length_mm)
        return self.pixel_pitch_mm / magnification


@dataclass(frozen=True)
class CalibratedScale:
    """Scale measured against a target, then extrapolated linearly with distance.

    Preferred over the lens model: it absorbs the true focal length, the sensor pitch and any
    fixed magnification of the optical path in one measured number.
    """

    mm_per_pixel_at_reference: float
    reference_distance_mm: float

    def __post_init__(self) -> None:
        if self.mm_per_pixel_at_reference <= 0 or self.reference_distance_mm <= 0:
            raise ValueError("calibration values must be positive")

    def mm_per_pixel(self, distance_mm: float) -> float:
        if distance_mm <= 0:
            raise ValueError("distance must be positive")
        return self.mm_per_pixel_at_reference * distance_mm / self.reference_distance_mm


def box_size_mm(
    box: BoundingBox, image_width_px: int, image_height_px: int, mm_per_pixel: float
) -> tuple[float, float]:
    """Physical size of a normalized box, as (longer, shorter) in millimetres."""
    if mm_per_pixel <= 0:
        raise ValueError("mm_per_pixel must be positive")

    width_mm = box.width * image_width_px * mm_per_pixel
    height_mm = box.height * image_height_px * mm_per_pixel
    return max(width_mm, height_mm), min(width_mm, height_mm)


def size_uncertainty_mm(
    length_mm: float, distance_mm: float, distance_uncertainty_mm: float
) -> float:
    """How much the measured size moves when the distance reading is wrong.

    Scale is proportional to distance, so the relative error carries straight through: a 2 mm
    error at 150 mm is 1.3%, which on an 0805 is 27 micrometres and irrelevant, but on a 20 mm
    connector is 0.27 mm and starts to matter.
    """
    if distance_mm <= 0:
        raise ValueError("distance must be positive")
    return abs(length_mm) * abs(distance_uncertainty_mm) / distance_mm
