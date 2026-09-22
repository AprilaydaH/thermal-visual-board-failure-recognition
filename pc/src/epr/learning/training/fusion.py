"""Supervised training for the LNN fusion model.

This is level 2 of the plan's learning structure: a candidate model trained in a separate
process from reviewed corrections. It deliberately does not touch a production model; only the
validation and registry steps may promote what this produces.

Losses are weighted per head so that the class head, which is the one the technician reads
first, dominates the package and thermal-condition heads.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from epr.recognition.fusion.model import LnnFusionModel


@dataclass(frozen=True)
class FusionBatch:
    """One training set. Features are (N, T, F) and labels are per sequence."""

    features: Tensor
    dt: Tensor
    component_class: Tensor
    package: Tensor
    thermal_condition: Tensor
    unknown: Tensor

    def __post_init__(self) -> None:
        count = self.features.shape[0]
        for name in ("dt", "component_class", "package", "thermal_condition", "unknown"):
            if getattr(self, name).shape[0] != count:
                raise ValueError(f"{name} has a different length than features")

    def tensors(self) -> tuple[Tensor, ...]:
        return (
            self.features,
            self.dt,
            self.component_class,
            self.package,
            self.thermal_condition,
            self.unknown,
        )


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 40
    batch_size: int = 16
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    class_weight: float = 1.0
    package_weight: float = 0.5
    thermal_weight: float = 0.5
    unknown_weight: float = 0.5


@dataclass
class TrainingResult:
    losses: list[float] = field(default_factory=list)
    class_accuracy: float = 0.0

    @property
    def final_loss(self) -> float:
        return self.losses[-1] if self.losses else float("nan")

    @property
    def improved(self) -> bool:
        return len(self.losses) > 1 and self.losses[-1] < self.losses[0]


def train_fusion(
    model: LnnFusionModel,
    batch: FusionBatch,
    config: TrainingConfig | None = None,
    *,
    device: torch.device | None = None,
) -> TrainingResult:
    config = config or TrainingConfig()
    device = device or torch.device("cpu")
    model = model.to(device).train()

    loader = DataLoader(
        TensorDataset(*(tensor.to(device) for tensor in batch.tensors())),
        batch_size=config.batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    cross_entropy = nn.CrossEntropyLoss()
    binary = nn.BCEWithLogitsLoss()

    result = TrainingResult()
    for _ in range(config.epochs):
        epoch_loss = 0.0
        for features, dt, target_class, target_package, target_thermal, target_unknown in loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(features, dt).last()
            loss = (
                config.class_weight * cross_entropy(output.class_logits, target_class)
                + config.package_weight * cross_entropy(output.package_logits, target_package)
                + config.thermal_weight * cross_entropy(output.thermal_logits, target_thermal)
                + config.unknown_weight * binary(output.unknown_logit, target_unknown)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            epoch_loss += float(loss) * features.shape[0]
        result.losses.append(epoch_loss / len(loader.dataset))

    result.class_accuracy = _class_accuracy(model, batch, device)
    return result


@torch.inference_mode()
def _class_accuracy(model: LnnFusionModel, batch: FusionBatch, device: torch.device) -> float:
    model.eval()
    output = model(batch.features.to(device), batch.dt.to(device)).last()
    predicted = output.class_logits.argmax(dim=-1).cpu()
    return float((predicted == batch.component_class).float().mean())
