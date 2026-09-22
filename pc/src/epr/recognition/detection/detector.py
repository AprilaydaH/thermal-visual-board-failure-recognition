"""Class-agnostic component detector.

A small fully convolutional encoder with a top-down path, predicting a centre heatmap, a box
size and a sub-cell offset at stride 4. Trained from scratch, like the rest of the stack, so
nothing is downloaded on an offline bench machine.

Being fully convolutional, the same weights run on a training tile and on a whole board.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from epr.core.domain_models.region import BoundingBox, ComponentRegion, RegionSource
from epr.recognition.classification.cnn import IMAGENET_FREE_MEAN, IMAGENET_FREE_STD
from epr.recognition.detection.heatmap import non_maximum_suppression

OUTPUT_STRIDE = 4
SIZE_DIVISOR = 16  # four halvings in the encoder
HEATMAP_PRIOR = 0.01


@dataclass(frozen=True)
class DetectorConfig:
    widths: tuple[int, ...] = (32, 64, 128, 192)
    head_width: int = 64
    tile_size: int = 256
    score_threshold: float = 0.3
    nms_iou: float = 0.45
    max_detections: int = 512
    min_box_pixels: int = 5
    # Typical component size in output cells. Starting the regressor near the real scale saves
    # the first epochs, and stops an untrained model emitting zero-sized boxes.
    size_prior_cells: float = 3.0

    def __post_init__(self) -> None:
        if len(self.widths) != 4:
            raise ValueError("widths must describe four encoder stages")
        if self.tile_size % SIZE_DIVISOR:
            raise ValueError(f"tile_size must be a multiple of {SIZE_DIVISOR}")


class DetectorOutput(NamedTuple):
    heatmap: Tensor  # logits
    size: Tensor
    offset: Tensor


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)


class ComponentDetector(nn.Module):
    def __init__(self, config: DetectorConfig | None = None) -> None:
        super().__init__()
        self.config = config or DetectorConfig()
        widths = self.config.widths
        head_width = self.config.head_width

        self.stem = _ConvBlock(3, widths[0], stride=2)
        self.stage4 = _ConvBlock(widths[0], widths[1], stride=2)
        self.stage8 = _ConvBlock(widths[1], widths[2], stride=2)
        self.stage16 = _ConvBlock(widths[2], widths[3], stride=2)

        self.lateral4 = nn.Conv2d(widths[1], head_width, 1)
        self.lateral8 = nn.Conv2d(widths[2], head_width, 1)
        self.lateral16 = nn.Conv2d(widths[3], head_width, 1)
        self.smooth = _ConvBlock(head_width, head_width)

        self.heatmap_head = self._head(head_width, 1)
        self.size_head = self._head(head_width, 2)
        self.offset_head = self._head(head_width, 2)

        # Start from the expected component density, otherwise the focal loss spends its first
        # epochs pushing a saturated heatmap back down.
        prior = -np.log((1 - HEATMAP_PRIOR) / HEATMAP_PRIOR)
        nn.init.constant_(self.heatmap_head[-1].bias, float(prior))
        nn.init.constant_(self.size_head[-1].bias, float(self.config.size_prior_cells))

    @staticmethod
    def _head(width: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, out_channels, 1),
        )

    def forward(self, image: Tensor) -> DetectorOutput:
        c2 = self.stem(image)
        c4 = self.stage4(c2)
        c8 = self.stage8(c4)
        c16 = self.stage16(c8)

        p16 = self.lateral16(c16)
        p8 = self.lateral8(c8) + F.interpolate(p16, size=c8.shape[-2:], mode="nearest")
        p4 = self.lateral4(c4) + F.interpolate(p8, size=c4.shape[-2:], mode="nearest")
        features = self.smooth(p4)

        return DetectorOutput(
            heatmap=self.heatmap_head(features),
            size=self.size_head(features),
            offset=self.offset_head(features),
        )


@dataclass
class Detection:
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))
    scores: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))

    def __len__(self) -> int:
        return len(self.scores)


def decode(
    output: DetectorOutput,
    *,
    score_threshold: float,
    max_detections: int,
    stride: int = OUTPUT_STRIDE,
) -> list[Detection]:
    """Turn network outputs into boxes.

    A 3x3 max pool acts as the peak picker: a cell survives only if it is the local maximum,
    which removes most duplicates before any IoU test.
    """
    scores = torch.sigmoid(output.heatmap)
    peaks = F.max_pool2d(scores, 3, stride=1, padding=1)
    scores = scores * (peaks == scores)

    batch, _, rows, columns = scores.shape
    flat = scores.view(batch, -1)
    k = min(max_detections, flat.shape[1])
    top_scores, indices = torch.topk(flat, k)

    row = torch.div(indices, columns, rounding_mode="floor")
    column = indices % columns

    size = output.size.view(batch, 2, -1).gather(2, indices.unsqueeze(1).expand(-1, 2, -1))
    offset = output.offset.view(batch, 2, -1).gather(2, indices.unsqueeze(1).expand(-1, 2, -1))

    centre_x = (column + offset[:, 0]) * stride
    centre_y = (row + offset[:, 1]) * stride
    half_width = size[:, 0].clamp(min=0) * stride / 2
    half_height = size[:, 1].clamp(min=0) * stride / 2

    boxes = torch.stack(
        [
            centre_x - half_width,
            centre_y - half_height,
            centre_x + half_width,
            centre_y + half_height,
        ],
        dim=-1,
    )

    results = []
    for index in range(batch):
        keep = top_scores[index] >= score_threshold
        results.append(
            Detection(
                boxes=boxes[index][keep].detach().cpu().numpy().astype(np.float32),
                scores=top_scores[index][keep].detach().cpu().numpy().astype(np.float32),
            )
        )
    return results


def image_to_tensor(image: np.ndarray, device: torch.device | None = None) -> Tensor:
    """One HxWx3 uint8 image to a normalized, padded NCHW tensor."""
    array = np.ascontiguousarray(image)
    tensor = torch.from_numpy(array).to(torch.float32).div_(255.0).permute(2, 0, 1)
    tensor = (tensor - IMAGENET_FREE_MEAN) / IMAGENET_FREE_STD
    tensor = tensor.unsqueeze(0)

    height, width = tensor.shape[-2:]
    pad_bottom = (-height) % SIZE_DIVISOR
    pad_right = (-width) % SIZE_DIVISOR
    if pad_bottom or pad_right:
        tensor = F.pad(tensor, (0, pad_right, 0, pad_bottom), mode="replicate")
    return tensor.to(device) if device is not None else tensor


@torch.inference_mode()
def detect(
    model: ComponentDetector,
    image: np.ndarray,
    *,
    device: torch.device | None = None,
    score_threshold: float | None = None,
) -> Detection:
    """Run the detector over a whole board and return pixel boxes."""
    model.eval()
    config = model.config
    tensor = image_to_tensor(image, device)

    output = model(tensor)
    detection = decode(
        output,
        score_threshold=config.score_threshold if score_threshold is None else score_threshold,
        max_detections=config.max_detections,
    )[0]

    height, width = image.shape[:2]
    boxes = detection.boxes
    if len(boxes):
        boxes[:, 0::2] = boxes[:, 0::2].clip(0, width)
        boxes[:, 1::2] = boxes[:, 1::2].clip(0, height)

        big_enough = (boxes[:, 2] - boxes[:, 0] >= config.min_box_pixels) & (
            boxes[:, 3] - boxes[:, 1] >= config.min_box_pixels
        )
        boxes, scores = boxes[big_enough], detection.scores[big_enough]

        keep = non_maximum_suppression(boxes, scores, config.nms_iou)
        return Detection(boxes=boxes[keep], scores=scores[keep])

    return detection


def detect_regions(
    model: ComponentDetector,
    image: np.ndarray,
    *,
    device: torch.device | None = None,
    score_threshold: float | None = None,
) -> list[ComponentRegion]:
    """Detections as normalized regions, ready for the recognition pipeline.

    This is what replaces the simulator's ground-truth regions. Boxes are normalized against
    the RGB frame, which assumes the channels are registered; until that holds, a hotspot can
    still be attributed to a neighbouring component.
    """
    height, width = image.shape[:2]
    detection = detect(model, image, device=device, score_threshold=score_threshold)

    regions = []
    for index, (box, score) in enumerate(zip(detection.boxes, detection.scores, strict=True)):
        x0, y0, x1, y1 = box
        normalized = BoundingBox(
            x=float(x0 / width),
            y=float(y0 / height),
            width=float(min(x1 - x0, width - x0) / width),
            height=float(min(y1 - y0, height - y0) / height),
        )
        regions.append(
            ComponentRegion(
                region_id=f"d{index:04d}",
                box=normalized,
                source=RegionSource.DETECTOR,
                label=None,
            )
        )
        _ = score
    return regions
