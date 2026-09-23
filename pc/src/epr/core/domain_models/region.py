"""Component regions.

Boxes are normalized to the frame so the same region addresses the RGB, NIR and thermal
channels despite their very different resolutions. Converting to pixels is always explicit.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RegionSource(str, Enum):
    DETECTOR = "detector"
    TECHNICIAN = "technician"
    SIMULATED = "simulated"


class BoundingBox(BaseModel):
    """Normalized box, origin at the top left of the frame."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(gt=0.0, le=1.0)
    height: float = Field(gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _within_frame(self) -> BoundingBox:
        if self.x + self.width > 1.0 or self.y + self.height > 1.0:
            raise ValueError("box extends beyond the frame")
        return self

    @property
    def centre(self) -> tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height / 2

    def pixel_box(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Return (x0, y0, x1, y1) clamped to the image and at least one pixel wide."""
        x0 = min(int(round(self.x * width)), width - 1)
        y0 = min(int(round(self.y * height)), height - 1)
        x1 = max(x0 + 1, min(int(round((self.x + self.width) * width)), width))
        y1 = max(y0 + 1, min(int(round((self.y + self.height) * height)), height))
        return x0, y0, x1, y1

    def slices(self, width: int, height: int) -> tuple[slice, slice]:
        x0, y0, x1, y1 = self.pixel_box(width, height)
        return slice(y0, y1), slice(x0, x1)


class ComponentRegion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    region_id: str = Field(min_length=1)
    box: BoundingBox
    source: RegionSource = RegionSource.DETECTOR
    label: str | None = None
    package: str | None = None
    # resistor, capacitor, ic, … — same vocabulary as ComponentClass values
    component_class: str | None = None
