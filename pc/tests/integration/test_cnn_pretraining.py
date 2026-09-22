"""RGB-stream pretraining, and handing the result to the full component CNN."""

import numpy as np
import pytest
import torch

from epr.learning.training.cnn import (
    CropDataset,
    PretrainConfig,
    RgbPretrainModel,
    class_weights,
    load_pretrained_rgb_stream,
    save_checkpoint,
    train_rgb_stream,
)
from epr.recognition.classification import CnnConfig, ComponentCnn
from epr.recognition.runtime import seed_everything

CROP, CLASSES, PER_CLASS = 16, 3, 60


@pytest.fixture
def config():
    return CnnConfig(
        crop_size=CROP, widths=(8, 16), embedding_dim=12, n_classes=CLASSES, n_packages=3
    )


@pytest.fixture
def datasets():
    """A learnable toy task: each class is a distinct brightness plus noise."""
    seed_everything(0)
    rng = np.random.default_rng(0)
    crops, labels = [], []
    for label in range(CLASSES):
        base = 40 + label * 70
        block = rng.normal(base, 12, (PER_CLASS, CROP, CROP, 3))
        crops.append(np.clip(block, 0, 255).astype(np.uint8))
        labels.append(np.full(PER_CLASS, label))

    crops = np.concatenate(crops)
    labels = np.concatenate(labels)
    order = rng.permutation(len(labels))
    crops, labels = crops[order], labels[order]

    split = int(len(labels) * 0.7)
    return (
        CropDataset(crops[:split], labels[:split], augment=True),
        CropDataset(crops[split:], labels[split:]),
    )


def test_pretraining_learns_and_improves(config, datasets):
    train, validation = datasets
    seed_everything(0)
    model = RgbPretrainModel(config, CLASSES)

    result = train_rgb_stream(model, train, validation, PretrainConfig(epochs=8, batch_size=16))

    assert result.train_losses[-1] < result.train_losses[0]
    assert result.best_accuracy > 0.8
    assert result.balanced_accuracy > 0.8


def test_dataset_returns_normalized_tensors(datasets):
    train, _ = datasets
    tensor, label = train[0]
    assert tensor.shape == (3, CROP, CROP)
    assert tensor.dtype == torch.float32
    assert isinstance(label, int)


def test_augmentation_changes_crops_but_not_labels(datasets):
    train, validation = datasets
    labels = {train[index][1] for index in range(5)}
    assert labels == {int(train.labels[index]) for index in range(5)}

    plain = validation[0][0]
    torch.testing.assert_close(plain, validation[0][0])


def test_mismatched_crops_and_labels_are_rejected():
    with pytest.raises(ValueError, match="same length"):
        CropDataset(np.zeros((4, CROP, CROP, 3), np.uint8), np.zeros(3, np.int64))


def test_class_weights_favour_rare_classes():
    labels = np.array([0] * 90 + [1] * 10)
    weights = class_weights(labels, 2)
    assert weights[1] > weights[0]


def test_pretrained_stream_loads_into_the_component_cnn(config, datasets, tmp_path):
    train, validation = datasets
    seed_everything(0)
    model = RgbPretrainModel(config, CLASSES)
    result = train_rgb_stream(model, train, validation, PretrainConfig(epochs=2, batch_size=16))

    path = save_checkpoint(
        tmp_path / "rgb.pt",
        model,
        label_space=["a", "b", "c"],
        dataset_fingerprint="abc123",
        result=result,
    )

    cnn = ComponentCnn(config)
    before = cnn.rgb_stream.project.weight.clone()
    nir_before = cnn.nir_stream.project.weight.clone()
    payload = load_pretrained_rgb_stream(cnn, path)

    assert payload["dataset_fingerprint"] == "abc123"
    assert not torch.equal(cnn.rgb_stream.project.weight, before)
    # The NIR stream has no public data to learn from and must be left alone.
    assert torch.equal(cnn.nir_stream.project.weight, nir_before)


def test_checkpoint_with_a_different_geometry_is_rejected(config, datasets, tmp_path):
    train, validation = datasets
    model = RgbPretrainModel(config, CLASSES)
    result = train_rgb_stream(model, train, validation, PretrainConfig(epochs=1, batch_size=32))
    path = save_checkpoint(
        tmp_path / "rgb.pt",
        model,
        label_space=["a", "b", "c"],
        dataset_fingerprint="abc123",
        result=result,
    )

    wider = ComponentCnn(CnnConfig(crop_size=CROP, widths=(8, 16), embedding_dim=24))
    with pytest.raises(ValueError, match="embedding_dim"):
        load_pretrained_rgb_stream(wider, path)
