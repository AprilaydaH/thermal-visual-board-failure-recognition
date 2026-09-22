"""Unknown and ambiguous component rejection.

A repair bench sees parts the model was never trained on, so a confident wrong label is worse
than an honest "unknown". Three independent signals are combined:

- the learned unknown head of the fusion model,
- the calibrated maximum class probability,
- the normalized predictive entropy, which catches the case of two classes splitting the mass.

Temperature scaling is a stored calibration value, fitted on a validation set. It defaults to
1.0, meaning uncalibrated, and a fitted value belongs in the model registry entry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor


class UnknownReason(str, Enum):
    ACCEPTED = "accepted"
    UNKNOWN_HEAD = "unknown_head"
    LOW_CONFIDENCE = "low_confidence"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class UnknownDetectorConfig:
    temperature: float = 1.0
    min_confidence: float = 0.55
    max_entropy_ratio: float = 0.70
    max_unknown_probability: float = 0.50

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")


@dataclass(frozen=True)
class UnknownDecision:
    is_unknown: bool
    reason: UnknownReason
    confidence: float
    entropy_ratio: float
    unknown_probability: float
    predicted_index: int


class UnknownDetector:
    def __init__(self, config: UnknownDetectorConfig | None = None) -> None:
        self.config = config or UnknownDetectorConfig()

    def decide(self, class_logits: Tensor, unknown_logit: Tensor) -> UnknownDecision:
        """Judge one component. ``class_logits`` is (C,) and ``unknown_logit`` is a scalar."""
        if class_logits.dim() != 1:
            raise ValueError(f"expected one component, got shape {tuple(class_logits.shape)}")

        probabilities = torch.softmax(class_logits / self.config.temperature, dim=-1)
        confidence, predicted = probabilities.max(dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
        entropy_ratio = float(entropy / math.log(class_logits.shape[-1]))
        unknown_probability = float(torch.sigmoid(unknown_logit))

        reason = UnknownReason.ACCEPTED
        if unknown_probability > self.config.max_unknown_probability:
            reason = UnknownReason.UNKNOWN_HEAD
        elif float(confidence) < self.config.min_confidence:
            reason = UnknownReason.LOW_CONFIDENCE
        elif entropy_ratio > self.config.max_entropy_ratio:
            reason = UnknownReason.AMBIGUOUS

        return UnknownDecision(
            is_unknown=reason is not UnknownReason.ACCEPTED,
            reason=reason,
            confidence=float(confidence),
            entropy_ratio=entropy_ratio,
            unknown_probability=unknown_probability,
            predicted_index=int(predicted),
        )
