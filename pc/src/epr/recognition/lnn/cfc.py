"""Liquid neural network: closed-form continuous-time (CfC) cells.

CfC is the closed-form solution of the liquid time-constant ODE, so a step costs no more than
a GRU while the state still evolves in continuous time. That matters here because frame sets
are not guaranteed to arrive on a fixed interval: the elapsed time between captures enters the
update explicitly, instead of being assumed constant as in a plain RNN.

One step, following Hasani et al.:

    z      = backbone([x, h])
    gate   = sigmoid(a(z) * dt + b(z))
    h_next = (1 - gate) * f(z) + gate * g(z)

A large ``dt`` saturates the gate, which is what makes a long pause between captures behave
differently from a rapid burst.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

MIN_TIME_STEP_S = 1e-3


@dataclass(frozen=True)
class CfcConfig:
    input_size: int
    hidden_size: int = 128
    backbone_units: int = 128
    backbone_layers: int = 2
    layers: int = 1
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.backbone_layers < 1:
            raise ValueError("backbone_layers must be at least 1")
        if self.layers < 1:
            raise ValueError("layers must be at least 1")


class CfcCell(nn.Module):
    def __init__(self, config: CfcConfig) -> None:
        super().__init__()
        self.config = config

        layers: list[nn.Module] = []
        width = config.input_size + config.hidden_size
        for _ in range(config.backbone_layers):
            layers += [nn.Linear(width, config.backbone_units), nn.SiLU()]
            width = config.backbone_units
        if config.dropout > 0:
            layers.append(nn.Dropout(config.dropout))
        self.backbone = nn.Sequential(*layers)

        self.candidate = nn.Linear(config.backbone_units, config.hidden_size)
        self.steady_state = nn.Linear(config.backbone_units, config.hidden_size)
        self.time_scale = nn.Linear(config.backbone_units, config.hidden_size)
        self.time_bias = nn.Linear(config.backbone_units, config.hidden_size)

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    def forward(self, x: Tensor, hidden: Tensor, dt: Tensor) -> Tensor:
        """One step. ``x`` is (B, F), ``hidden`` is (B, H), ``dt`` is (B,) or (B, 1) seconds."""
        z = self.backbone(torch.cat([x, hidden], dim=-1))
        if dt.dim() == 1:
            dt = dt.unsqueeze(-1)
        dt = dt.clamp_min(MIN_TIME_STEP_S)

        gate = torch.sigmoid(self.time_scale(z) * dt + self.time_bias(z))
        candidate = torch.tanh(self.candidate(z))
        steady_state = torch.tanh(self.steady_state(z))
        return (1.0 - gate) * candidate + gate * steady_state

    def initial_hidden(self, batch_size: int, device: torch.device | None = None) -> Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device)


class LiquidNetwork(nn.Module):
    """Stacked CfC cells applied over a sequence of per-component feature vectors."""

    def __init__(self, config: CfcConfig) -> None:
        super().__init__()
        self.config = config
        cells = []
        for index in range(config.layers):
            layer_config = CfcConfig(
                input_size=config.input_size if index == 0 else config.hidden_size,
                hidden_size=config.hidden_size,
                backbone_units=config.backbone_units,
                backbone_layers=config.backbone_layers,
                layers=1,
                dropout=config.dropout,
            )
            cells.append(CfcCell(layer_config))
        self.cells = nn.ModuleList(cells)
        self.norm = nn.LayerNorm(config.hidden_size)

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    def forward(
        self, x: Tensor, dt: Tensor, hidden: list[Tensor] | None = None
    ) -> tuple[Tensor, list[Tensor]]:
        """Run a sequence.

        ``x`` is (B, T, F) and ``dt`` is (B, T) seconds since the previous step. Returns the
        per-step output (B, T, H) and the final hidden state of every layer, so inference can
        continue from where a live inspection left off.
        """
        if x.dim() != 3:
            raise ValueError(f"expected (batch, time, features), got shape {tuple(x.shape)}")
        batch_size, steps, _ = x.shape
        # Shapes become traced values during ONNX export, where comparing them is meaningless.
        if not torch.jit.is_tracing() and dt.shape != (batch_size, steps):
            raise ValueError(f"dt must be {(batch_size, steps)}, got {tuple(dt.shape)}")

        states = (
            [cell.initial_hidden(batch_size, x.device) for cell in self.cells]
            if hidden is None
            else list(hidden)
        )

        outputs = []
        for step in range(steps):
            signal = x[:, step]
            step_dt = dt[:, step]
            for index, cell in enumerate(self.cells):
                states[index] = cell(signal, states[index], step_dt)
                signal = states[index]
            outputs.append(self.norm(signal))
        return torch.stack(outputs, dim=1), states
