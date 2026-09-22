"""The LNN must actually learn from a feature sequence, not merely run."""

import math

import pytest
import torch

from epr.learning.training import FusionBatch, TrainingConfig, train_fusion
from epr.recognition.fusion import FeatureLayout, FusionConfig, LnnFusionModel
from epr.recognition.runtime import seed_everything

VISUAL, NIR = 8, 8
CLASSES, PACKAGES = 4, 3
SEQUENCES, STEPS = 96, 4


@pytest.fixture
def layout():
    return FeatureLayout(visual_dim=VISUAL, nir_dim=NIR)


@pytest.fixture
def batch(layout):
    """A learnable toy task: the class shifts the visual block, the rest is noise."""
    seed_everything(0)
    features = torch.randn(SEQUENCES, STEPS, layout.total) * 0.3
    component_class = torch.randint(0, CLASSES, (SEQUENCES,))
    visual = layout.blocks["visual"]
    for index, label in enumerate(component_class):
        features[index, :, visual] += float(label) * 1.5

    return FusionBatch(
        features=features,
        dt=torch.full((SEQUENCES, STEPS), 2.0),
        component_class=component_class,
        package=torch.randint(0, PACKAGES, (SEQUENCES,)),
        thermal_condition=torch.randint(0, 4, (SEQUENCES,)),
        unknown=torch.zeros(SEQUENCES),
    )


@pytest.fixture
def model(layout):
    seed_everything(0)
    return LnnFusionModel(
        FusionConfig(
            layout=layout,
            n_classes=CLASSES,
            n_packages=PACKAGES,
            hidden_size=32,
            backbone_units=32,
            dropout=0.0,
        )
    )


def test_training_reduces_the_loss_and_learns_the_class(model, batch):
    result = train_fusion(model, batch, TrainingConfig(epochs=30, batch_size=16))

    assert result.improved
    assert result.final_loss < result.losses[0] * 0.6
    assert result.class_accuracy > 0.9


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_training_runs_on_the_gpu(model, batch):
    result = train_fusion(
        model, batch, TrainingConfig(epochs=3), device=torch.device("cuda")
    )
    assert math.isfinite(result.final_loss)
    assert 0.0 <= result.class_accuracy <= 1.0


def test_mismatched_label_length_is_rejected(batch):
    with pytest.raises(ValueError, match="different length"):
        FusionBatch(
            features=batch.features,
            dt=batch.dt,
            component_class=batch.component_class[:-1],
            package=batch.package,
            thermal_condition=batch.thermal_condition,
            unknown=batch.unknown,
        )
