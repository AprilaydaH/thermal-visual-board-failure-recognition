"""Pretraining for the RGB stream of the component CNN.

Public PCB photographs have no 850 nm channel, so only the RGB stream is trained here. The
NIR stream stays randomly initialized until real captures exist. Feeding zeros to the NIR
stream to reuse the full model would teach the fused head that NIR carries no information,
which is the opposite of what the design intends.

The checkpoint holds the stream weights plus the dataset fingerprint and runtime facts, which
is what the model registry entry needs.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from epr.recognition.classification.cnn import (
    IMAGENET_FREE_MEAN,
    IMAGENET_FREE_STD,
    CnnConfig,
    ComponentCnn,
    StreamEncoder,
)
from epr.recognition.runtime import describe_runtime

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PretrainConfig:
    epochs: int = 15
    batch_size: int = 128
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    balance_classes: bool = True
    augment: bool = True


@dataclass
class PretrainResult:
    train_losses: list[float] = field(default_factory=list)
    validation_accuracy: list[float] = field(default_factory=list)
    balanced_accuracy: float = 0.0

    @property
    def best_accuracy(self) -> float:
        return max(self.validation_accuracy) if self.validation_accuracy else 0.0


class CropDataset(Dataset):
    """Crops from a cache, optionally with light augmentation.

    Flips and small rotations only. Colour jitter is deliberately mild: on a real board the
    solder mask colour and the marking contrast carry information the model should keep.
    """

    def __init__(
        self,
        crops: np.ndarray,
        labels: np.ndarray,
        *,
        augment: bool = False,
        generator: np.random.Generator | None = None,
    ) -> None:
        if len(crops) != len(labels):
            raise ValueError("crops and labels must have the same length")
        self.crops = crops
        self.labels = labels
        self.augment = augment
        self._rng = generator or np.random.default_rng(0)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        # A copy, not a view: the cache is a read-only memory map.
        crop = np.array(self.crops[index])
        if self.augment:
            crop = self._augment(crop)

        tensor = torch.from_numpy(np.ascontiguousarray(crop)).to(torch.float32).div_(255.0)
        tensor = tensor.permute(2, 0, 1)
        tensor = (tensor - IMAGENET_FREE_MEAN) / IMAGENET_FREE_STD
        return tensor, int(self.labels[index])

    def _augment(self, crop: np.ndarray) -> np.ndarray:
        if self._rng.random() < 0.5:
            crop = crop[:, ::-1]
        if self._rng.random() < 0.5:
            crop = crop[::-1, :]
        rotations = int(self._rng.integers(0, 4))
        if rotations:
            crop = np.rot90(crop, rotations)
        if self._rng.random() < 0.3:
            scale = 1.0 + float(self._rng.normal(0.0, 0.08))
            crop = np.clip(crop.astype(np.float32) * scale, 0, 255).astype(np.uint8)
        return crop


class RgbPretrainModel(nn.Module):
    """The RGB stream with a classification head, trainable on its own."""

    def __init__(self, config: CnnConfig, n_classes: int) -> None:
        super().__init__()
        self.config = config
        self.stream = StreamEncoder(config.rgb_channels, config.widths, config.embedding_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(config.embedding_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dim, n_classes),
        )

    def forward(self, rgb: Tensor) -> Tensor:
        return self.head(self.stream(rgb))


def class_weights(labels: np.ndarray, n_classes: int) -> Tensor:
    """Inverse-frequency weights. The component distribution is extremely skewed: without
    this, a model that answers "capacitor" for everything already scores well."""
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    weights = np.where(counts > 0, counts.sum() / np.maximum(counts, 1), 0.0)
    return torch.as_tensor(weights / weights[weights > 0].mean(), dtype=torch.float32)


def train_rgb_stream(
    model: RgbPretrainModel,
    train: CropDataset,
    validation: CropDataset,
    config: PretrainConfig | None = None,
    *,
    device: torch.device | None = None,
) -> PretrainResult:
    config = config or PretrainConfig()
    device = device or torch.device("cpu")
    model = model.to(device)

    n_classes = model.head[-1].out_features
    weights = class_weights(train.labels, n_classes).to(device) if config.balance_classes else None
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True, drop_last=False)
    validation_loader = DataLoader(validation, batch_size=config.batch_size)

    result = PretrainResult()
    for epoch in range(config.epochs):
        model.train()
        total = 0.0
        for crops, labels in train_loader:
            crops, labels = crops.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(crops), labels)
            loss.backward()
            optimizer.step()
            total += float(loss) * crops.shape[0]
        scheduler.step()

        result.train_losses.append(total / len(train))
        accuracy, balanced = evaluate(model, validation_loader, n_classes, device)
        result.validation_accuracy.append(accuracy)
        result.balanced_accuracy = balanced
        logger.info(
            "epoch %d/%d loss %.4f validation accuracy %.3f balanced %.3f",
            epoch + 1,
            config.epochs,
            result.train_losses[-1],
            accuracy,
            balanced,
        )

    return result


@torch.inference_mode()
def evaluate(
    model: nn.Module, loader: DataLoader, n_classes: int, device: torch.device
) -> tuple[float, float]:
    """Plain and balanced accuracy.

    Balanced accuracy is the one to watch: with this class distribution the plain number is
    dominated by resistors and capacitors.
    """
    model.eval()
    correct = np.zeros(n_classes)
    totals = np.zeros(n_classes)
    for crops, labels in loader:
        predicted = model(crops.to(device)).argmax(dim=-1).cpu().numpy()
        actual = labels.numpy()
        for label, prediction in zip(actual, predicted, strict=True):
            totals[label] += 1
            correct[label] += label == prediction

    seen = totals > 0
    accuracy = float(correct.sum() / max(totals.sum(), 1))
    balanced = float((correct[seen] / totals[seen]).mean()) if seen.any() else 0.0
    return accuracy, balanced


def save_checkpoint(
    path: Path | str,
    model: RgbPretrainModel,
    *,
    label_space: list[str],
    dataset_fingerprint: str,
    result: PretrainResult,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stream_state_dict": model.stream.state_dict(),
            "cnn_config": asdict(model.config),
            "label_space": label_space,
            "dataset_fingerprint": dataset_fingerprint,
            "validation_accuracy": result.best_accuracy,
            "balanced_accuracy": result.balanced_accuracy,
            "runtime": describe_runtime(),
        },
        path,
    )
    return path


def load_pretrained_rgb_stream(cnn: ComponentCnn, checkpoint: Path | str) -> dict:
    """Copy pretrained weights into the RGB stream of a full component CNN.

    The NIR stream and the fused heads are left untouched; they are trained later on real
    captures.
    """
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved = payload["cnn_config"]
    for key in ("widths", "embedding_dim", "rgb_channels"):
        expected = tuple(saved[key]) if isinstance(saved[key], list) else saved[key]
        actual = getattr(cnn.config, key)
        if expected != actual:
            raise ValueError(f"checkpoint {key} is {expected}, model expects {actual}")

    cnn.rgb_stream.load_state_dict(payload["stream_state_dict"])
    return payload
