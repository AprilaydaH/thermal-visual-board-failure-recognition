"""LNN fusion model.

Consumes the per-component feature sequence and produces, for every time step, the outputs
the plan asks of the LNN: refined component class, package prediction, thermal condition and
an unknown/ambiguous signal. Emitting them per step rather than only at the end lets the user
interface show confidence settling as the board heats up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor, nn

from epr.recognition.fusion.features import FeatureLayout
from epr.recognition.lnn.cfc import CfcConfig, LiquidNetwork

THERMAL_CONDITIONS = ("normal", "elevated", "hot", "abnormal_rise")


@dataclass(frozen=True)
class FusionConfig:
    layout: FeatureLayout
    n_classes: int
    n_packages: int
    n_thermal_conditions: int = len(THERMAL_CONDITIONS)
    hidden_size: int = 128
    backbone_units: int = 128
    backbone_layers: int = 2
    layers: int = 1
    dropout: float = 0.1

    def cfc(self) -> CfcConfig:
        return CfcConfig(
            input_size=self.layout.total,
            hidden_size=self.hidden_size,
            backbone_units=self.backbone_units,
            backbone_layers=self.backbone_layers,
            layers=self.layers,
            dropout=self.dropout,
        )


class FusionOutput(NamedTuple):
    class_logits: Tensor
    package_logits: Tensor
    thermal_logits: Tensor
    unknown_logit: Tensor

    def last(self) -> FusionOutput:
        """The prediction after the final observed time step."""
        return FusionOutput(*(tensor[:, -1] for tensor in self))


class LnnFusionModel(nn.Module):
    def __init__(self, config: FusionConfig) -> None:
        super().__init__()
        self.config = config
        self.lnn = LiquidNetwork(config.cfc())
        hidden = config.hidden_size
        self.class_head = nn.Linear(hidden, config.n_classes)
        self.package_head = nn.Linear(hidden, config.n_packages)
        self.thermal_head = nn.Linear(hidden, config.n_thermal_conditions)
        self.unknown_head = nn.Linear(hidden, 1)

    @property
    def feature_dim(self) -> int:
        return self.config.layout.total

    def forward(self, x: Tensor, dt: Tensor) -> FusionOutput:
        """``x`` is (B, T, F) and ``dt`` is (B, T). Every output is (B, T, ...)."""
        sequence, _ = self.lnn(x, dt)
        return FusionOutput(
            class_logits=self.class_head(sequence),
            package_logits=self.package_head(sequence),
            thermal_logits=self.thermal_head(sequence),
            unknown_logit=self.unknown_head(sequence).squeeze(-1),
        )


class FusionStepModule(nn.Module):
    """Single-step view of the fusion model for streaming inference.

    The sequence model unrolls its time loop when traced, which fixes the sequence length in
    the exported graph. A live inspection instead wants to advance one frame set at a time and
    carry the state, so this wrapper takes and returns the hidden state explicitly.
    """

    def __init__(self, model: LnnFusionModel) -> None:
        super().__init__()
        self.model = model

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    @property
    def layers(self) -> int:
        return self.model.config.layers

    def initial_state(self, batch_size: int, device: torch.device | None = None) -> Tensor:
        return torch.zeros(self.layers, batch_size, self.hidden_size, device=device)

    def forward(self, x: Tensor, dt: Tensor, state: Tensor) -> tuple[Tensor, ...]:
        """``x`` is (B, F), ``dt`` is (B,), ``state`` is (layers, B, H)."""
        signal = x
        next_states = []
        for index, cell in enumerate(self.model.lnn.cells):
            signal = cell(signal, state[index], dt)
            next_states.append(signal)
        normalized = self.model.lnn.norm(signal)
        return (
            self.model.class_head(normalized),
            self.model.package_head(normalized),
            self.model.thermal_head(normalized),
            self.model.unknown_head(normalized).squeeze(-1),
            torch.stack(next_states, dim=0),
        )
