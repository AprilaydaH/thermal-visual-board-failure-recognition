"""Centre-heatmap encoding and decoding for the component detector.

A component is represented by a Gaussian peak at its centre, plus a size and a sub-cell
offset, rather than by anchor boxes. Components on a board sit on a near-regular grid at wildly
different scales, from an 0402 chip to a full connector, which is exactly the case where anchor
tuning is painful and a centre-based encoding is not.

Detection here is class-agnostic. The detector answers "is a component here", the pretrained
CNN answers "which component is it". The WACV authors reached the same split: per-class
detection on this data is unreliable because several component types are visually almost
identical at board resolution.
"""

from __future__ import annotations

import math

import numpy as np

MIN_RADIUS = 1
GAUSSIAN_SIGMA_RATIO = 6.0


def gaussian_radius(height: float, width: float, min_overlap: float = 0.7) -> float:
    """Radius within which a predicted centre still yields at least ``min_overlap`` IoU.

    Peaks for small components must be tight or neighbouring parts merge; peaks for large
    connectors can be broad. Solving for the overlap gives that scaling for free.
    """
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(max(b1**2 - 4 * c1, 0.0))) / 2

    a2, b2 = 4.0, 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    r2 = (b2 + math.sqrt(max(b2**2 - 4 * a2 * c2, 0.0))) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    r3 = (-b3 + math.sqrt(max(b3**2 - 4 * a3 * c3, 0.0))) / (2 * a3)

    return max(min(r1, r2, r3), MIN_RADIUS)


def draw_gaussian(heatmap: np.ndarray, centre_x: float, centre_y: float, radius: float) -> None:
    """Paint a Gaussian peak in place, keeping the maximum where peaks overlap."""
    radius = int(max(radius, MIN_RADIUS))
    diameter = 2 * radius + 1
    sigma = diameter / GAUSSIAN_SIGMA_RATIO

    grid = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(grid[:, None] ** 2 + grid[None, :] ** 2) / (2 * sigma**2))
    kernel[kernel < np.finfo(np.float32).eps * kernel.max()] = 0

    height, width = heatmap.shape
    x, y = int(centre_x), int(centre_y)
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    if left + right <= 0 or top + bottom <= 0:
        return

    region = heatmap[y - top : y + bottom, x - left : x + right]
    patch = kernel[radius - top : radius + bottom, radius - left : radius + right]
    np.maximum(region, patch, out=region)


def render_targets(
    boxes: np.ndarray, image_size: tuple[int, int], stride: int
) -> dict[str, np.ndarray]:
    """Build heatmap, size, offset and mask targets for one image.

    ``boxes`` are pixel ``(x0, y0, x1, y1)``. Sizes are regressed in stride units so the loss
    is not dominated by large connectors, and the offset recovers the centre position lost to
    the stride, which matters when a component is only a few cells across.
    """
    height, width = image_size
    rows, columns = height // stride, width // stride

    heatmap = np.zeros((1, rows, columns), dtype=np.float32)
    size = np.zeros((2, rows, columns), dtype=np.float32)
    offset = np.zeros((2, rows, columns), dtype=np.float32)
    mask = np.zeros((1, rows, columns), dtype=np.float32)

    for x0, y0, x1, y1 in np.asarray(boxes, dtype=np.float32).reshape(-1, 4):
        box_width, box_height = x1 - x0, y1 - y0
        if box_width <= 0 or box_height <= 0:
            continue

        centre_x = (x0 + x1) / 2 / stride
        centre_y = (y0 + y1) / 2 / stride
        column, row = int(centre_x), int(centre_y)
        if not (0 <= row < rows and 0 <= column < columns):
            continue

        draw_gaussian(
            heatmap[0], centre_x, centre_y, gaussian_radius(box_height / stride, box_width / stride)
        )
        size[:, row, column] = (box_width / stride, box_height / stride)
        offset[:, row, column] = (centre_x - column, centre_y - row)
        mask[0, row, column] = 1.0

    return {"heatmap": heatmap, "size": size, "offset": offset, "mask": mask}


def iou_matrix(boxes: np.ndarray, others: np.ndarray) -> np.ndarray:
    if len(boxes) == 0 or len(others) == 0:
        return np.zeros((len(boxes), len(others)), dtype=np.float32)

    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    others = np.asarray(others, dtype=np.float32).reshape(-1, 4)

    left = np.maximum(boxes[:, None, 0], others[None, :, 0])
    top = np.maximum(boxes[:, None, 1], others[None, :, 1])
    right = np.minimum(boxes[:, None, 2], others[None, :, 2])
    bottom = np.minimum(boxes[:, None, 3], others[None, :, 3])

    intersection = np.clip(right - left, 0, None) * np.clip(bottom - top, 0, None)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    other_area = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    union = area[:, None] + other_area[None, :] - intersection
    return (intersection / np.maximum(union, 1e-9)).astype(np.float32)


def non_maximum_suppression(
    boxes: np.ndarray, scores: np.ndarray, threshold: float = 0.45
) -> np.ndarray:
    """Return indices to keep, highest score first."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.int64)

    order = np.argsort(-np.asarray(scores))
    keep: list[int] = []
    while len(order):
        best = order[0]
        keep.append(int(best))
        if len(order) == 1:
            break
        overlaps = iou_matrix(boxes[best][None], boxes[order[1:]])[0]
        order = order[1:][overlaps <= threshold]
    return np.asarray(keep, dtype=np.int64)


def average_precision(
    predicted: np.ndarray,
    scores: np.ndarray,
    truth: np.ndarray,
    iou_threshold: float = 0.5,
) -> tuple[float, float, float]:
    """All-point-interpolated AP, plus recall and precision for one image.

    Each ground-truth box may be matched once, so duplicate detections count as false
    positives rather than quietly inflating recall.
    """
    n_truth = len(truth)
    if len(predicted) == 0:
        return 0.0, 0.0, 0.0 if n_truth else 1.0

    order = np.argsort(-np.asarray(scores))
    predicted = np.asarray(predicted)[order]
    overlaps = iou_matrix(predicted, truth)

    matched = np.zeros(n_truth, dtype=bool)
    true_positive = np.zeros(len(predicted), dtype=np.float64)
    for index in range(len(predicted)):
        if n_truth == 0:
            break
        candidate = int(np.argmax(overlaps[index]))
        if overlaps[index, candidate] >= iou_threshold and not matched[candidate]:
            matched[candidate] = True
            true_positive[index] = 1.0

    cumulative_tp = np.cumsum(true_positive)
    cumulative_fp = np.cumsum(1.0 - true_positive)
    recall = cumulative_tp / max(n_truth, 1)
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-9)

    # Make precision monotonically decreasing, then integrate over recall.
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    changes = np.where(np.diff(np.concatenate(([0.0], recall))) > 0)[0]
    ap = float(np.sum(np.diff(np.concatenate(([0.0], recall)))[changes] * precision[changes]))

    return ap, float(recall[-1]) if n_truth else 0.0, float(cumulative_tp[-1] / len(predicted))
