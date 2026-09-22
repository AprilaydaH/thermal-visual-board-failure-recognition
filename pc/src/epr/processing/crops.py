"""Component cropping.

Crops are taken per channel at that channel's own resolution, using the normalized box, so a
160x120 thermal frame and a 5 MP RGB frame are addressed by the same region.
"""

from __future__ import annotations

import cv2
import numpy as np

from epr.core.domain_models.region import BoundingBox


def extract_crop(image: np.ndarray, box: BoundingBox, size: int | None = None) -> np.ndarray:
    """Crop a normalized box and optionally resize it to a square of ``size`` pixels."""
    height, width = image.shape[:2]
    rows, columns = box.slices(width, height)
    crop = image[rows, columns]
    if size is None:
        return crop

    interpolation = cv2.INTER_AREA if crop.shape[0] >= size else cv2.INTER_LINEAR
    return cv2.resize(crop, (size, size), interpolation=interpolation)


def stack_crops(crops: list[np.ndarray]) -> np.ndarray:
    return np.stack(crops, axis=0) if crops else np.empty((0,), dtype=np.uint8)
