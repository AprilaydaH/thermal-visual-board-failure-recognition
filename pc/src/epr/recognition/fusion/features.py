"""LNN input vector.

Per detected component and time step, the plan defines

    x_t = [visual features, NIR features, OCR features,
           T_max, T_mean, dT, dT/dt, ambient temperature]

``FeatureLayout`` is the single place that fixes this order and the block widths. Models store
their layout, so a model trained against one layout cannot silently be fed another.

Temperatures are scaled into roughly unit range before they reach the network; the raw values
stay available in the metrics objects for the user interface and reports.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from epr.processing.thermal.metrics import THERMAL_FEATURE_DIM, ThermalMetrics
from epr.recognition.ocr.features import OCR_FEATURE_DIM

TEMPERATURE_SCALE = 100.0
HEATING_RATE_SCALE = 5.0
AMBIENT_DIM = 1


@dataclass(frozen=True)
class FeatureLayout:
    visual_dim: int
    nir_dim: int
    ocr_dim: int = OCR_FEATURE_DIM
    thermal_dim: int = THERMAL_FEATURE_DIM
    ambient_dim: int = AMBIENT_DIM

    @property
    def total(self) -> int:
        return self.visual_dim + self.nir_dim + self.ocr_dim + self.thermal_dim + self.ambient_dim

    @property
    def blocks(self) -> dict[str, slice]:
        """Where each block sits in the vector. Used by tests and by diagnostics."""
        offsets = {}
        start = 0
        for name, size in (
            ("visual", self.visual_dim),
            ("nir", self.nir_dim),
            ("ocr", self.ocr_dim),
            ("thermal", self.thermal_dim),
            ("ambient", self.ambient_dim),
        ):
            offsets[name] = slice(start, start + size)
            start += size
        return offsets


def scale_thermal(metrics: ThermalMetrics) -> np.ndarray:
    return np.array(
        [
            metrics.t_max_c / TEMPERATURE_SCALE,
            metrics.t_mean_c / TEMPERATURE_SCALE,
            metrics.delta_t_c / TEMPERATURE_SCALE,
            metrics.heating_rate_c_s / HEATING_RATE_SCALE,
        ],
        dtype=np.float32,
    )


def build_step_features(
    visual: Tensor,
    nir: Tensor,
    ocr: np.ndarray,
    metrics: ThermalMetrics,
    ambient_c: float,
    layout: FeatureLayout,
) -> Tensor:
    """Assemble one x_t. ``visual`` and ``nir`` are CNN embeddings of shape (D,)."""
    if visual.shape[-1] != layout.visual_dim or nir.shape[-1] != layout.nir_dim:
        raise ValueError("embedding width does not match the feature layout")

    device, dtype = visual.device, visual.dtype
    parts = [
        visual,
        nir,
        torch.as_tensor(ocr, device=device, dtype=dtype),
        torch.as_tensor(scale_thermal(metrics), device=device, dtype=dtype),
        torch.tensor([ambient_c / TEMPERATURE_SCALE], device=device, dtype=dtype),
    ]
    return torch.cat(parts, dim=-1)


def build_sequence_features(
    visual: Tensor,
    nir: Tensor,
    ocr: Sequence[np.ndarray],
    metrics: Sequence[ThermalMetrics],
    ambient_c: Sequence[float],
    layout: FeatureLayout,
) -> Tensor:
    """Assemble (T, F) for one component. ``visual`` and ``nir`` are (T, D)."""
    steps = visual.shape[0]
    if not (len(ocr) == len(metrics) == len(ambient_c) == steps):
        raise ValueError("visual, NIR, OCR, thermal and ambient inputs must have equal length")

    rows = [
        build_step_features(
            visual[step], nir[step], ocr[step], metrics[step], ambient_c[step], layout
        )
        for step in range(steps)
    ]
    return torch.stack(rows, dim=0)
