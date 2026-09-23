"""Display-only images. Never written back to the raw store."""

from __future__ import annotations

import cv2
import numpy as np

from epr.processing.thermal.metrics import raw_to_celsius


def colorize_thermal(raw: np.ndarray) -> np.ndarray:
    """False-colour RGB preview of a radiometric frame. The raw values are not modified."""
    celsius = raw_to_celsius(raw)
    lo, hi = float(celsius.min()), float(celsius.max())
    scaled = (celsius - lo) / max(hi - lo, 1e-6)
    u8 = np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def draw_probe_marks(
    image: np.ndarray, pixel: tuple[int, int], colour: tuple[int, int, int]
) -> np.ndarray:
    canvas = np.ascontiguousarray(image.copy())
    x, y = pixel
    size = max(6, min(canvas.shape[:2]) // 40)
    cv2.drawMarker(canvas, (x, y), colour, cv2.MARKER_CROSS, size, 2, cv2.LINE_AA)
    return canvas
