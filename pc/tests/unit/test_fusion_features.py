import numpy as np
import pytest
import torch

from epr.processing.thermal.metrics import ThermalMetrics
from epr.recognition.fusion import FeatureLayout, build_sequence_features, build_step_features
from epr.recognition.ocr import OCR_FEATURE_DIM

VISUAL, NIR = 6, 4
AMBIENT_C = 22.0


@pytest.fixture
def layout():
    return FeatureLayout(visual_dim=VISUAL, nir_dim=NIR)


@pytest.fixture
def metrics():
    return ThermalMetrics(t_max_c=60.0, t_mean_c=40.0, delta_t_c=38.0, heating_rate_c_s=2.5)


def test_layout_matches_the_plan(layout):
    assert layout.total == VISUAL + NIR + OCR_FEATURE_DIM + 4 + 1
    blocks = layout.blocks
    assert list(blocks) == ["visual", "nir", "ocr", "thermal", "ambient"]
    assert blocks["visual"].start == 0
    assert blocks["ambient"].stop == layout.total


def test_blocks_land_where_the_layout_says(layout, metrics):
    visual = torch.full((VISUAL,), 1.0)
    nir = torch.full((NIR,), 2.0)
    ocr = np.full(OCR_FEATURE_DIM, 3.0, dtype=np.float32)

    features = build_step_features(visual, nir, ocr, metrics, AMBIENT_C, layout)
    blocks = layout.blocks

    assert features.shape == (layout.total,)
    assert torch.all(features[blocks["visual"]] == 1.0)
    assert torch.all(features[blocks["nir"]] == 2.0)
    assert torch.all(features[blocks["ocr"]] == 3.0)
    torch.testing.assert_close(
        features[blocks["thermal"]], torch.tensor([0.60, 0.40, 0.38, 0.50])
    )
    assert features[blocks["ambient"]].item() == pytest.approx(0.22)


def test_sequence_stacks_time_steps(layout, metrics):
    steps = 3
    features = build_sequence_features(
        torch.zeros(steps, VISUAL),
        torch.zeros(steps, NIR),
        [np.zeros(OCR_FEATURE_DIM, dtype=np.float32)] * steps,
        [metrics] * steps,
        [AMBIENT_C] * steps,
        layout,
    )
    assert features.shape == (steps, layout.total)


def test_embedding_width_must_match_the_layout(layout, metrics):
    with pytest.raises(ValueError, match="embedding width"):
        build_step_features(
            torch.zeros(VISUAL + 1),
            torch.zeros(NIR),
            np.zeros(OCR_FEATURE_DIM, dtype=np.float32),
            metrics,
            AMBIENT_C,
            layout,
        )


def test_ragged_sequence_inputs_are_rejected(layout, metrics):
    with pytest.raises(ValueError, match="equal length"):
        build_sequence_features(
            torch.zeros(3, VISUAL),
            torch.zeros(3, NIR),
            [np.zeros(OCR_FEATURE_DIM, dtype=np.float32)] * 2,
            [metrics] * 3,
            [AMBIENT_C] * 3,
            layout,
        )
