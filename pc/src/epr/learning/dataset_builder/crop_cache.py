"""Turns annotated boards into a crop cache for training.

Each board is one large photograph holding hundreds of components, so crops are extracted
once into a memory-mapped array rather than re-decoding the photograph on every access.

The manifest records a fingerprint of the exact instances that went in. That is the training
dataset version the plan requires every model to carry.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from epr.learning.dataset_builder.wacv import ComponentInstance

logger = logging.getLogger(__name__)

CROPS_FILE = "crops.npy"
LABELS_FILE = "labels.npy"
BOARDS_FILE = "boards.npy"
MANIFEST_FILE = "manifest.json"

DEFAULT_MARGIN = 0.12


@dataclass(frozen=True)
class CropCacheManifest:
    crop_size: int
    margin: float
    label_space: list[str]
    label_counts: dict[str, int]
    board_ids: list[str]
    instance_count: int
    fingerprint: str

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> CropCacheManifest:
        return cls(**json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class CropCache:
    crops: np.ndarray
    labels: np.ndarray
    board_index: np.ndarray
    manifest: CropCacheManifest

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    @property
    def label_space(self) -> tuple[str, ...]:
        return tuple(self.manifest.label_space)

    @classmethod
    def load(cls, directory: Path | str) -> CropCache:
        directory = Path(directory)
        return cls(
            crops=np.load(directory / CROPS_FILE, mmap_mode="r"),
            labels=np.load(directory / LABELS_FILE),
            board_index=np.load(directory / BOARDS_FILE),
            manifest=CropCacheManifest.load(directory / MANIFEST_FILE),
        )


def fingerprint(instances: Sequence[ComponentInstance], crop_size: int, margin: float) -> str:
    digest = hashlib.sha256()
    digest.update(f"{crop_size}:{margin:.4f}".encode())
    for instance in sorted(
        instances, key=lambda i: (i.board_id, i.label, i.box, str(i.image_path))
    ):
        digest.update(f"{instance.board_id}|{instance.label}|{instance.box}".encode())
    return digest.hexdigest()[:16]


def extract_square_crop(
    image: np.ndarray, box: tuple[int, int, int, int], size: int, margin: float = DEFAULT_MARGIN
) -> np.ndarray:
    """Crop a component, keeping its aspect ratio.

    The box is squared around its centre before resizing, so an elongated 0805 chip does not
    get stretched into the same shape as a square QFN. Edges are replicated where the square
    runs past the board photograph.
    """
    height, width = image.shape[:2]
    x0, y0, x1, y1 = box
    centre_x, centre_y = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) * (1.0 + margin) / 2

    left, top = int(round(centre_x - half)), int(round(centre_y - half))
    right, bottom = int(round(centre_x + half)), int(round(centre_y + half))

    crop = image[max(top, 0) : min(bottom, height), max(left, 0) : min(right, width)]
    if crop.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)

    crop = cv2.copyMakeBorder(
        crop,
        max(-top, 0),
        max(bottom - height, 0),
        max(-left, 0),
        max(right - width, 0),
        cv2.BORDER_REPLICATE,
    )
    interpolation = cv2.INTER_AREA if crop.shape[0] >= size else cv2.INTER_CUBIC
    return cv2.resize(crop, (size, size), interpolation=interpolation)


def build_crop_cache(
    instances: Sequence[ComponentInstance],
    directory: Path | str,
    *,
    label_space: Sequence[str],
    crop_size: int = 64,
    margin: float = DEFAULT_MARGIN,
) -> CropCache:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    label_to_index = {label: index for index, label in enumerate(label_space)}
    kept = [instance for instance in instances if instance.label in label_to_index]
    if not kept:
        raise ValueError("no instance matches the label space")

    board_ids = sorted({instance.board_id for instance in kept})
    board_to_index = {board: index for index, board in enumerate(board_ids)}

    crops = np.lib.format.open_memmap(
        directory / CROPS_FILE,
        mode="w+",
        dtype=np.uint8,
        shape=(len(kept), crop_size, crop_size, 3),
    )
    labels = np.zeros(len(kept), dtype=np.int64)
    boards = np.zeros(len(kept), dtype=np.int64)

    by_image: dict[Path, list[tuple[int, ComponentInstance]]] = defaultdict(list)
    for position, instance in enumerate(kept):
        by_image[instance.image_path].append((position, instance))

    written = 0
    for image_path, entries in by_image.items():
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("could not read %s, skipping %d crops", image_path, len(entries))
            continue
        for position, instance in entries:
            crops[position] = extract_square_crop(image, instance.box, crop_size, margin)
            labels[position] = label_to_index[instance.label]
            boards[position] = board_to_index[instance.board_id]
            written += 1

    crops.flush()
    np.save(directory / LABELS_FILE, labels)
    np.save(directory / BOARDS_FILE, boards)

    manifest = CropCacheManifest(
        crop_size=crop_size,
        margin=margin,
        label_space=list(label_space),
        label_counts=dict(Counter(instance.label for instance in kept)),
        board_ids=board_ids,
        instance_count=written,
        fingerprint=fingerprint(kept, crop_size, margin),
    )
    manifest.save(directory / MANIFEST_FILE)
    logger.info("cached %d crops from %d boards in %s", written, len(board_ids), directory)

    return CropCache.load(directory)
