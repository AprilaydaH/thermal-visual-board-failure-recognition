"""Recognition pipeline.

Runs the plan's inspection workflow steps 6 to 10 for a sequence of frame sets: crop each
component, extract visual, NIR, OCR and thermal features, fuse them over time with the LNN and
report a prediction with its uncertainty.

Registration caveat: the pipeline addresses every channel with the same normalized box, which
assumes the channels are already registered. That holds for the simulator and it will not hold
for the real head until the calibration and registration work of gate G3 is done. Until then a
thermal hotspot can be attributed to a neighbouring component, which the plan names as the
highest risk in the project.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from epr.core.domain_models.frame_set import Channel, FrameSet
from epr.core.domain_models.region import ComponentRegion
from epr.processing.crops import extract_crop
from epr.processing.thermal.metrics import ThermalMetrics, elapsed_seconds, sequence_metrics
from epr.recognition.classification.cnn import ComponentCnn, crops_to_tensor
from epr.recognition.fusion.features import build_sequence_features
from epr.recognition.fusion.model import THERMAL_CONDITIONS, LnnFusionModel
from epr.recognition.ocr.features import MarkingReading, encode_marking
from epr.recognition.unknown_detection.detector import UnknownDecision, UnknownDetector


@dataclass(frozen=True)
class LabelSpace:
    component_classes: tuple[str, ...]
    packages: tuple[str, ...]
    thermal_conditions: tuple[str, ...] = THERMAL_CONDITIONS


@dataclass(frozen=True)
class ComponentPrediction:
    region_id: str
    component_class: str
    package: str
    thermal_condition: str
    thermal: ThermalMetrics
    decision: UnknownDecision

    @property
    def confidence(self) -> float:
        return self.decision.confidence

    @property
    def is_unknown(self) -> bool:
        return self.decision.is_unknown


class RecognitionPipeline:
    def __init__(
        self,
        cnn: ComponentCnn,
        fusion: LnnFusionModel,
        labels: LabelSpace,
        *,
        detector: UnknownDetector | None = None,
        device: torch.device | None = None,
    ) -> None:
        layout = fusion.config.layout
        if cnn.embedding_dim != layout.visual_dim or cnn.embedding_dim != layout.nir_dim:
            raise ValueError(
                f"CNN embedding of {cnn.embedding_dim} does not match the fusion layout "
                f"({layout.visual_dim} visual, {layout.nir_dim} NIR)"
            )
        if len(labels.component_classes) != fusion.config.n_classes:
            raise ValueError("label space and fusion model disagree on the number of classes")

        self.device = device or torch.device("cpu")
        self.cnn = cnn.to(self.device).eval()
        self.fusion = fusion.to(self.device).eval()
        self.labels = labels
        self.detector = detector or UnknownDetector()

    @torch.inference_mode()
    def run(
        self,
        frame_sets: Sequence[FrameSet],
        regions: Sequence[ComponentRegion],
        markings: Mapping[str, MarkingReading | str] | None = None,
    ) -> list[ComponentPrediction]:
        if not frame_sets:
            raise ValueError("at least one frame set is required")
        if not regions:
            return []

        steps = len(frame_sets)
        crop_size = self.cnn.config.crop_size
        markings = markings or {}

        rgb_crops: list[np.ndarray] = []
        nir_crops: list[np.ndarray] = []
        for region in regions:
            for frame_set in frame_sets:
                rgb_crops.append(extract_crop(frame_set.array(Channel.RGB), region.box, crop_size))
                nir_crops.append(extract_crop(frame_set.array(Channel.NIR), region.box, crop_size))

        embedding = self.cnn(
            crops_to_tensor(rgb_crops, device=self.device),
            crops_to_tensor(nir_crops, device=self.device),
        )
        visual = embedding.rgb_embedding.view(len(regions), steps, -1)
        nir = embedding.nir_embedding.view(len(regions), steps, -1)

        ambient = [frame_set.metadata.environment.ambient_temperature_c for frame_set in frame_sets]
        layout = self.fusion.config.layout
        thermal_per_region = [sequence_metrics(frame_sets, region.box) for region in regions]

        sequences = []
        for index, region in enumerate(regions):
            ocr = [encode_marking(markings.get(region.region_id))] * steps
            sequences.append(
                build_sequence_features(
                    visual[index], nir[index], ocr, thermal_per_region[index], ambient, layout
                )
            )

        features = torch.stack(sequences, dim=0)
        dt = torch.as_tensor(elapsed_seconds(frame_sets), device=self.device)
        dt = dt.unsqueeze(0).expand(len(regions), steps).contiguous()

        output = self.fusion(features, dt).last()
        return [
            self._prediction(region, thermal_per_region[index][-1], output, index)
            for index, region in enumerate(regions)
        ]

    def _prediction(
        self,
        region: ComponentRegion,
        thermal: ThermalMetrics,
        output,
        index: int,
    ) -> ComponentPrediction:
        decision = self.detector.decide(output.class_logits[index], output.unknown_logit[index])
        package_index = int(output.package_logits[index].argmax())
        condition_index = int(output.thermal_logits[index].argmax())
        component_class = (
            "unknown"
            if decision.is_unknown
            else self.labels.component_classes[decision.predicted_index]
        )
        return ComponentPrediction(
            region_id=region.region_id,
            component_class=component_class,
            package=self.labels.packages[package_index],
            thermal_condition=self.labels.thermal_conditions[condition_index],
            thermal=thermal,
            decision=decision,
        )
