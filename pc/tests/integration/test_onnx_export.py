"""Production inference runs on ONNX Runtime, so every model must survive the round trip."""

import numpy as np
import pytest
import torch

from epr.recognition.classification import CnnConfig, ComponentCnn
from epr.recognition.export import (
    export_component_cnn,
    export_fusion_sequence,
    export_fusion_step,
    max_absolute_difference,
    run_onnx,
)
from epr.recognition.fusion import FeatureLayout, FusionConfig, LnnFusionModel
from epr.recognition.fusion.model import FusionStepModule

TOLERANCE = 1e-4
EMBEDDING, CLASSES, PACKAGES = 8, 5, 4
STEPS = 4

pytest.importorskip("onnxruntime")


@pytest.fixture
def layout():
    return FeatureLayout(visual_dim=EMBEDDING, nir_dim=EMBEDDING)


@pytest.fixture
def cnn():
    torch.manual_seed(0)
    return ComponentCnn(
        CnnConfig(
            crop_size=32,
            widths=(8, 16),
            embedding_dim=EMBEDDING,
            n_classes=CLASSES,
            n_packages=PACKAGES,
        )
    ).eval()


@pytest.fixture
def fusion(layout):
    torch.manual_seed(0)
    return LnnFusionModel(
        FusionConfig(
            layout=layout,
            n_classes=CLASSES,
            n_packages=PACKAGES,
            hidden_size=24,
            backbone_units=24,
            dropout=0.0,
        )
    ).eval()


def test_cnn_matches_onnx_runtime(cnn, tmp_path):
    rgb = torch.randn(2, 3, cnn.config.crop_size, cnn.config.crop_size)
    nir = torch.randn(2, 1, cnn.config.crop_size, cnn.config.crop_size)

    path = export_component_cnn(cnn, tmp_path / "component_cnn.onnx")
    with torch.inference_mode():
        expected = cnn(rgb, nir)
    actual = run_onnx(path, {"rgb": rgb.numpy(), "nir": nir.numpy()})

    for reference, candidate in zip(expected, actual, strict=True):
        assert max_absolute_difference(reference, candidate) < TOLERANCE


def test_fusion_sequence_matches_onnx_runtime(fusion, layout, tmp_path):
    features = torch.randn(3, STEPS, layout.total)
    dt = torch.full((3, STEPS), 2.0)

    path = export_fusion_sequence(fusion, tmp_path / "fusion_sequence.onnx", STEPS)
    with torch.inference_mode():
        expected = fusion(features, dt)
    actual = run_onnx(path, {"features": features.numpy(), "dt": dt.numpy()})

    for reference, candidate in zip(expected, actual, strict=True):
        assert max_absolute_difference(reference, candidate) < TOLERANCE


def test_streaming_export_reproduces_the_sequence_model(fusion, layout, tmp_path):
    """Stepping frame set by frame set must give the same answer as the batched run."""
    features = torch.randn(1, STEPS, layout.total)
    dt = torch.full((1, STEPS), 3.0)
    with torch.inference_mode():
        expected = fusion(features, dt)

    path = export_fusion_step(fusion, tmp_path / "fusion_step.onnx")
    state = FusionStepModule(fusion).initial_state(1).numpy()

    for step in range(STEPS):
        outputs = run_onnx(
            path,
            {
                "features": features[:, step].numpy(),
                "dt": dt[:, step].numpy(),
                "state": state,
            },
        )
        *logits, state = outputs

    for reference, candidate in zip(expected.last(), logits, strict=True):
        assert max_absolute_difference(reference, candidate) < TOLERANCE


@pytest.mark.parametrize("batch", [1, 5])
def test_exported_graph_accepts_any_batch_size(cnn, tmp_path, batch):
    """The exporter specializes size-one dimensions, which silently pins the batch size."""
    path = export_component_cnn(cnn, tmp_path / "component_cnn.onnx")
    size = cnn.config.crop_size
    outputs = run_onnx(
        path,
        {
            "rgb": np.zeros((batch, 3, size, size), dtype=np.float32),
            "nir": np.zeros((batch, 1, size, size), dtype=np.float32),
        },
    )
    assert all(output.shape[0] == batch for output in outputs)


def test_exported_files_are_not_empty(cnn, tmp_path):
    path = export_component_cnn(cnn, tmp_path / "nested" / "cnn.onnx")
    assert path.stat().st_size > 0


def test_onnx_runtime_is_deterministic(fusion, layout, tmp_path):
    features = torch.randn(1, STEPS, layout.total)
    dt = torch.full((1, STEPS), 1.0)
    path = export_fusion_sequence(fusion, tmp_path / "fusion.onnx", STEPS)

    first = run_onnx(path, {"features": features.numpy(), "dt": dt.numpy()})
    second = run_onnx(path, {"features": features.numpy(), "dt": dt.numpy()})
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)
