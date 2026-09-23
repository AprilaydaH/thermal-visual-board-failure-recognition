"""Display-only images. Never written back to the raw store."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from epr.core.domain_models.region import ComponentRegion
from epr.processing.thermal.metrics import raw_to_celsius

CLASS_COLOURS = {
    "resistor": (80, 200, 120),
    "capacitor": (80, 160, 255),
    "ic": (255, 180, 60),
    "transistor": (220, 100, 255),
    "mosfet": (220, 100, 255),
    "connector": (160, 160, 180),
    "diode": (100, 220, 220),
    "led": (255, 230, 80),
    "inductor": (180, 140, 255),
}


def colorize_thermal(raw: np.ndarray) -> np.ndarray:
    """False-colour RGB preview of a radiometric frame. The raw values are not modified."""
    celsius = raw_to_celsius(raw)
    lo, hi = float(celsius.min()), float(celsius.max())
    scaled = (celsius - lo) / max(hi - lo, 1e-6)
    u8 = np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def draw_probe_marks(
    image: np.ndarray,
    pixel: tuple[int, int],
    colour: tuple[int, int, int],
    *,
    label: str | None = None,
) -> np.ndarray:
    canvas = np.ascontiguousarray(image.copy())
    x, y = pixel
    size = max(10, min(canvas.shape[:2]) // 28)
    cv2.circle(canvas, (x, y), size // 2, colour, 2, cv2.LINE_AA)
    cv2.drawMarker(canvas, (x, y), colour, cv2.MARKER_CROSS, size, 2, cv2.LINE_AA)
    if label:
        cv2.putText(
            canvas,
            label,
            (x + size // 2 + 4, max(y - 6, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            colour,
            1,
            cv2.LINE_AA,
        )
    return canvas


def draw_region_overlays(
    image: np.ndarray,
    regions: Sequence[ComponentRegion],
    *,
    active_id: str | None = None,
) -> np.ndarray:
    """Outline annotated parts on an RGB preview."""
    canvas = np.ascontiguousarray(image.copy())
    height, width = canvas.shape[:2]
    for region in regions:
        x0, y0, x1, y1 = region.box.pixel_box(width, height)
        colour = CLASS_COLOURS.get(region.component_class or "", (140, 200, 220))
        thickness = 3 if region.region_id == active_id else 1
        cv2.rectangle(canvas, (x0, y0), (x1, y1), colour, thickness, cv2.LINE_AA)
        caption = region.label or region.region_id
        if region.component_class:
            caption = f"{caption} · {region.component_class}"
        cv2.putText(
            canvas,
            caption,
            (x0 + 2, max(y0 - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            colour,
            1,
            cv2.LINE_AA,
        )
    return canvas
