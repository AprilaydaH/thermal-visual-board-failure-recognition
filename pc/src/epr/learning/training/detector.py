"""Training for the class-agnostic component detector.

Boards are few and large, components are many and small, so training samples random tiles
rather than whole boards. Evaluation runs on whole boards, because that is the unit the
technician actually inspects and tile-level numbers would hide components cut by a tile edge.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from epr.learning.dataset_builder.wacv import ComponentInstance
from epr.recognition.classification.cnn import IMAGENET_FREE_MEAN, IMAGENET_FREE_STD
from epr.recognition.detection.detector import (
    OUTPUT_STRIDE,
    ComponentDetector,
    DetectorConfig,
    detect,
)
from epr.recognition.detection.heatmap import average_precision, render_targets
from epr.recognition.runtime import describe_runtime

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectorTrainingConfig:
    epochs: int = 40
    tiles_per_epoch: int = 1024
    batch_size: int = 16
    learning_rate: float = 1.5e-3
    weight_decay: float = 1e-4
    size_weight: float = 0.1
    offset_weight: float = 1.0
    augment: bool = True


@dataclass
class DetectionMetrics:
    average_precision: float = 0.0
    recall: float = 0.0
    precision: float = 0.0

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0


@dataclass
class DetectorResult:
    train_losses: list[float] = field(default_factory=list)
    metrics: list[DetectionMetrics] = field(default_factory=list)

    @property
    def best(self) -> DetectionMetrics:
        return max(self.metrics, key=lambda m: m.average_precision, default=DetectionMetrics())


@dataclass(frozen=True)
class BoardSample:
    board_id: str
    image: np.ndarray
    boxes: np.ndarray


def load_boards(instances: Sequence[ComponentInstance]) -> list[BoardSample]:
    """Group annotations by board scan and read each image once.

    The whole dataset is a few dozen modest photographs, so it is held in memory rather than
    cached to disk like the classification crops.
    """
    grouped: dict[tuple[str, str], list[ComponentInstance]] = defaultdict(list)
    for instance in instances:
        grouped[(instance.board_id, str(instance.image_path))].append(instance)

    boards = []
    for (board_id, image_path), members in sorted(grouped.items()):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("could not read %s", image_path)
            continue
        boxes = np.array([instance.box for instance in members], dtype=np.float32)
        boards.append(BoardSample(board_id=board_id, image=image, boxes=boxes))

    if not boards:
        raise ValueError("no readable boards")
    return boards


class TileDataset(Dataset):
    """Random tiles from random boards, with the boxes they contain."""

    def __init__(
        self,
        boards: Sequence[BoardSample],
        *,
        tile_size: int = 256,
        length: int = 1024,
        augment: bool = True,
        seed: int = 0,
    ) -> None:
        if not boards:
            raise ValueError("no boards supplied")
        self.boards = list(boards)
        self.tile_size = tile_size
        self.length = length
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        rng = np.random.default_rng(self.seed * 1_000_003 + index)
        tile, boxes = self._sample_tile(rng)
        if self.augment:
            tile, boxes = self._augment(tile, boxes, rng)

        targets = render_targets(boxes, (self.tile_size, self.tile_size), OUTPUT_STRIDE)
        image = torch.from_numpy(np.ascontiguousarray(tile)).to(torch.float32).div_(255.0)
        image = (image.permute(2, 0, 1) - IMAGENET_FREE_MEAN) / IMAGENET_FREE_STD

        return {"image": image, **{k: torch.from_numpy(v) for k, v in targets.items()}}

    def _sample_tile(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Prefer tiles that contain at least one component.

        Board photographs have wide empty margins; sampling uniformly would spend most of the
        budget teaching the model that background is background, which it learns immediately.
        """
        for _ in range(8):
            board = self.boards[int(rng.integers(len(self.boards)))]
            tile, boxes = self._crop(board, rng, anchored=True)
            if len(boxes):
                return tile, boxes

        board = self.boards[int(rng.integers(len(self.boards)))]
        return self._crop(board, rng, anchored=False)

    def _crop(
        self, board: BoardSample, rng: np.random.Generator, *, anchored: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width = board.image.shape[:2]
        size = self.tile_size

        if anchored and len(board.boxes):
            box = board.boxes[int(rng.integers(len(board.boxes)))]
            centre_x = (box[0] + box[2]) / 2
            centre_y = (box[1] + box[3]) / 2
            left = int(centre_x - size / 2 + rng.integers(-size // 4, size // 4 + 1))
            top = int(centre_y - size / 2 + rng.integers(-size // 4, size // 4 + 1))
        else:
            left = int(rng.integers(0, max(width - size, 0) + 1))
            top = int(rng.integers(0, max(height - size, 0) + 1))

        left = int(np.clip(left, 0, max(width - size, 0)))
        top = int(np.clip(top, 0, max(height - size, 0)))

        tile = board.image[top : top + size, left : left + size]
        if tile.shape[0] != size or tile.shape[1] != size:
            tile = cv2.copyMakeBorder(
                tile, 0, size - tile.shape[0], 0, size - tile.shape[1], cv2.BORDER_REPLICATE
            )

        boxes = board.boxes - np.array([left, top, left, top], dtype=np.float32)
        return tile, _clip_boxes(boxes, size)

    def _augment(
        self, tile: np.ndarray, boxes: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        size = self.tile_size
        if rng.random() < 0.5:
            tile = tile[:, ::-1]
            boxes = (
                np.stack(
                    [size - boxes[:, 2], boxes[:, 1], size - boxes[:, 0], boxes[:, 3]], axis=-1
                )
                if len(boxes)
                else boxes
            )
        if rng.random() < 0.5:
            tile = tile[::-1, :]
            boxes = (
                np.stack(
                    [boxes[:, 0], size - boxes[:, 3], boxes[:, 2], size - boxes[:, 1]], axis=-1
                )
                if len(boxes)
                else boxes
            )
        if rng.random() < 0.3:
            scale = 1.0 + float(rng.normal(0.0, 0.1))
            tile = np.clip(tile.astype(np.float32) * scale, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(tile), boxes


def _clip_boxes(boxes: np.ndarray, size: int, min_visible: float = 0.4) -> np.ndarray:
    """Keep boxes that are mostly inside the tile, clipped to it.

    A component sliced by the tile edge would otherwise be labelled at the wrong size and
    teach the regressor to shrink.
    """
    if not len(boxes):
        return np.zeros((0, 4), dtype=np.float32)

    original = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    clipped = boxes.copy()
    clipped[:, 0::2] = clipped[:, 0::2].clip(0, size)
    clipped[:, 1::2] = clipped[:, 1::2].clip(0, size)
    area = (clipped[:, 2] - clipped[:, 0]) * (clipped[:, 3] - clipped[:, 1])

    keep = (area > 0) & (area >= min_visible * np.maximum(original, 1e-6))
    return clipped[keep].astype(np.float32)


def focal_loss(logits: Tensor, target: Tensor) -> Tensor:
    """CenterNet's penalty-reduced focal loss.

    Cells near a centre are not plain negatives: the ``(1 - target)`` term discounts them, so
    a prediction one cell off the peak is not punished as hard as one on empty laminate.
    """
    predicted = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    positive = target.ge(1.0).float()
    negative = 1.0 - positive

    positive_loss = -torch.log(predicted) * (1 - predicted).pow(2) * positive
    negative_loss = -torch.log(1 - predicted) * predicted.pow(2) * (1 - target).pow(4) * negative

    count = positive.sum()
    total = positive_loss.sum() + negative_loss.sum()
    return total / count if count > 0 else negative_loss.sum()


def masked_l1(predicted: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    count = mask.sum()
    if count == 0:
        return predicted.sum() * 0.0
    return (F.l1_loss(predicted * mask, target * mask, reduction="sum")) / count


def train_detector(
    model: ComponentDetector,
    train_boards: Sequence[BoardSample],
    validation_boards: Sequence[BoardSample],
    config: DetectorTrainingConfig | None = None,
    *,
    device: torch.device | None = None,
    seed: int = 0,
) -> DetectorResult:
    config = config or DetectorTrainingConfig()
    device = device or torch.device("cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    result = DetectorResult()
    for epoch in range(config.epochs):
        dataset = TileDataset(
            train_boards,
            tile_size=model.config.tile_size,
            length=config.tiles_per_epoch,
            augment=config.augment,
            seed=seed + epoch,
        )
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

        model.train()
        total = 0.0
        for batch in loader:
            images = batch["image"].to(device)
            heatmap = batch["heatmap"].to(device)
            size = batch["size"].to(device)
            offset = batch["offset"].to(device)
            mask = batch["mask"].to(device)

            output = model(images)
            loss = (
                focal_loss(output.heatmap, heatmap)
                + config.size_weight * masked_l1(output.size, size, mask)
                + config.offset_weight * masked_l1(output.offset, offset, mask)
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss) * images.shape[0]

        scheduler.step()
        result.train_losses.append(total / len(dataset))

        metrics = evaluate_detector(model, validation_boards, device=device)
        result.metrics.append(metrics)
        logger.info(
            "epoch %d/%d loss %.4f AP50 %.3f recall %.3f precision %.3f",
            epoch + 1,
            config.epochs,
            result.train_losses[-1],
            metrics.average_precision,
            metrics.recall,
            metrics.precision,
        )

    return result


def evaluate_detector(
    model: ComponentDetector,
    boards: Sequence[BoardSample],
    *,
    device: torch.device | None = None,
    iou_threshold: float = 0.5,
    score_threshold: float | None = None,
) -> DetectionMetrics:
    """Average per-board AP, recall and precision at a fixed IoU."""
    scores = []
    for board in boards:
        detection = detect(model, board.image, device=device, score_threshold=score_threshold)
        scores.append(
            average_precision(
                detection.boxes, detection.scores, board.boxes, iou_threshold=iou_threshold
            )
        )

    if not scores:
        return DetectionMetrics()
    ap, recall, precision = (float(np.mean(column)) for column in zip(*scores, strict=True))
    return DetectionMetrics(average_precision=ap, recall=recall, precision=precision)


def save_detector(
    path: Path | str,
    model: ComponentDetector,
    *,
    dataset_fingerprint: str,
    result: DetectorResult,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    best = result.best
    torch.save(
        {
            "state_dict": model.state_dict(),
            "detector_config": asdict(model.config),
            "dataset_fingerprint": dataset_fingerprint,
            "average_precision": best.average_precision,
            "recall": best.recall,
            "precision": best.precision,
            "runtime": describe_runtime(),
        },
        path,
    )
    return path


def load_detector(path: Path | str, *, device: torch.device | None = None) -> ComponentDetector:
    payload = torch.load(path, map_location=device or "cpu", weights_only=False)
    saved = dict(payload["detector_config"])
    saved["widths"] = tuple(saved["widths"])
    model = ComponentDetector(DetectorConfig(**saved))
    model.load_state_dict(payload["state_dict"])
    return model.to(device) if device else model
