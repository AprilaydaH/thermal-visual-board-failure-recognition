"""Thermal feature extraction.

Produces the four thermal terms of the LNN input vector for one component region:
T_max, T_mean, dT and dT/dt. The heating rate needs two frame sets, so it is computed by
``sequence_metrics`` from the capture timestamps rather than guessed from a single frame.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from epr.core.domain_models.frame_set import Channel, FrameSet
from epr.core.domain_models.region import BoundingBox

KELVIN_OFFSET = 273.15
CENTIKELVIN = 100.0

THERMAL_FEATURE_DIM = 4


def raw_to_celsius(raw: np.ndarray) -> np.ndarray:
    """Convert radiometric centikelvin, as sent by the head, to degrees Celsius."""
    return raw.astype(np.float32) / CENTIKELVIN - KELVIN_OFFSET


def celsius_to_raw(celsius: np.ndarray) -> np.ndarray:
    return np.clip(np.round((celsius + KELVIN_OFFSET) * CENTIKELVIN), 0, 65535).astype("<u2")


@dataclass(frozen=True)
class ThermalMetrics:
    t_max_c: float
    t_mean_c: float
    delta_t_c: float
    heating_rate_c_s: float

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.t_max_c, self.t_mean_c, self.delta_t_c, self.heating_rate_c_s],
            dtype=np.float32,
        )


def component_metrics(
    frame_set: FrameSet,
    box: BoundingBox,
    *,
    previous: ThermalMetrics | None = None,
    elapsed_s: float = 0.0,
) -> ThermalMetrics:
    thermal = frame_set.array(Channel.THERMAL)
    height, width = thermal.shape
    rows, columns = box.slices(width, height)
    celsius = raw_to_celsius(thermal[rows, columns])

    ambient_c = frame_set.metadata.environment.ambient_temperature_c
    t_max = float(celsius.max())
    t_mean = float(celsius.mean())

    if previous is None or elapsed_s <= 0.0:
        heating_rate = 0.0
    else:
        heating_rate = (t_max - previous.t_max_c) / elapsed_s

    return ThermalMetrics(
        t_max_c=t_max,
        t_mean_c=t_mean,
        delta_t_c=t_max - ambient_c,
        heating_rate_c_s=heating_rate,
    )


def sequence_metrics(
    frame_sets: Sequence[FrameSet], box: BoundingBox
) -> list[ThermalMetrics]:
    """Metrics for one region across a capture sequence, with the heating rate filled in."""
    metrics: list[ThermalMetrics] = []
    previous: ThermalMetrics | None = None
    previous_timestamp_ns: int | None = None

    for frame_set in frame_sets:
        timestamp_ns = frame_set.metadata.timestamp_ns
        elapsed_s = (
            0.0 if previous_timestamp_ns is None else (timestamp_ns - previous_timestamp_ns) / 1e9
        )
        current = component_metrics(frame_set, box, previous=previous, elapsed_s=elapsed_s)
        metrics.append(current)
        previous, previous_timestamp_ns = current, timestamp_ns

    return metrics


def elapsed_seconds(frame_sets: Sequence[FrameSet]) -> np.ndarray:
    """Time since the previous frame set, in seconds. The first step reports zero."""
    timestamps = np.array([fs.metadata.timestamp_ns for fs in frame_sets], dtype=np.float64)
    deltas = np.diff(timestamps, prepend=timestamps[:1]) / 1e9
    return deltas.astype(np.float32)
