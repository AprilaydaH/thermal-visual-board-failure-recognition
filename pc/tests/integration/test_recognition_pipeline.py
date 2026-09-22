"""Simulated frame sets through the whole recognition pipeline."""

import pytest
import torch

from epr.device.simulator import SimulatedDevice
from epr.recognition.classification import CnnConfig, ComponentCnn
from epr.recognition.fusion import FeatureLayout, FusionConfig, LnnFusionModel
from epr.recognition.ocr import MarkingReading
from epr.recognition.pipeline import LabelSpace, RecognitionPipeline

EMBEDDING = 8
CLASSES = ("resistor", "capacitor", "transistor", "integrated_circuit", "connector")
PACKAGES = ("0805", "SOT23", "SOIC8", "QFP64", "HEADER")


@pytest.fixture
def device():
    return SimulatedDevice(rgb_size=(160, 120), thermal_size=(80, 60), frame_interval_s=5.0)


@pytest.fixture
def pipeline():
    torch.manual_seed(0)
    cnn = ComponentCnn(
        CnnConfig(
            crop_size=32,
            widths=(8, 16),
            embedding_dim=EMBEDDING,
            n_classes=len(CLASSES),
            n_packages=len(PACKAGES),
        )
    )
    fusion = LnnFusionModel(
        FusionConfig(
            layout=FeatureLayout(visual_dim=EMBEDDING, nir_dim=EMBEDDING),
            n_classes=len(CLASSES),
            n_packages=len(PACKAGES),
            hidden_size=32,
            backbone_units=32,
            dropout=0.0,
        )
    )
    return RecognitionPipeline(cnn, fusion, LabelSpace(CLASSES, PACKAGES))


def test_every_region_gets_a_prediction(device, pipeline):
    frame_sets = list(device.capture_many(4))
    regions = device.component_regions()

    predictions = pipeline.run(frame_sets, regions)

    assert [p.region_id for p in predictions] == [r.region_id for r in regions]
    for prediction in predictions:
        assert 0.0 <= prediction.confidence <= 1.0
        assert prediction.package in PACKAGES
        assert prediction.thermal_condition in LabelSpace(CLASSES, PACKAGES).thermal_conditions
        assert prediction.component_class in (*CLASSES, "unknown")


def test_thermal_metrics_follow_the_heated_component(device, pipeline):
    predictions = pipeline.run(list(device.capture_many(4)), device.component_regions())
    by_region = {prediction.region_id: prediction for prediction in predictions}

    assert by_region["U1"].thermal.t_max_c > by_region["C1"].thermal.t_max_c
    assert by_region["U1"].thermal.delta_t_c > 10.0
    assert by_region["C1"].thermal.heating_rate_c_s < by_region["U1"].thermal.heating_rate_c_s


def test_markings_reach_the_feature_vector(device, pipeline):
    frame_sets = list(device.capture_many(2))
    regions = device.component_regions()

    without = pipeline.run(frame_sets, regions)
    with_marking = pipeline.run(
        frame_sets, regions, markings={"U1": MarkingReading("STM32N657", confidence=0.9)}
    )

    assert without[0].decision.confidence != with_marking[0].decision.confidence


def test_a_single_frame_set_still_works(device, pipeline):
    """Recognition must not require a heating sequence to produce a first answer."""
    predictions = pipeline.run([device.capture()], device.component_regions())
    assert len(predictions) == len(device.component_regions())
    assert all(p.thermal.heating_rate_c_s == 0.0 for p in predictions)


def test_no_regions_means_no_predictions(device, pipeline):
    assert pipeline.run([device.capture()], []) == []


def test_empty_capture_is_rejected(device, pipeline):
    with pytest.raises(ValueError, match="at least one frame set"):
        pipeline.run([], device.component_regions())


def test_mismatched_embedding_and_layout_is_rejected():
    cnn = ComponentCnn(CnnConfig(crop_size=32, widths=(8, 16), embedding_dim=EMBEDDING))
    fusion = LnnFusionModel(
        FusionConfig(
            layout=FeatureLayout(visual_dim=EMBEDDING + 1, nir_dim=EMBEDDING),
            n_classes=len(CLASSES),
            n_packages=len(PACKAGES),
        )
    )
    with pytest.raises(ValueError, match="does not match the fusion layout"):
        RecognitionPipeline(cnn, fusion, LabelSpace(CLASSES, PACKAGES))
