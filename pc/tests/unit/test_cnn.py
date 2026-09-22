import numpy as np
import pytest
import torch

from epr.recognition.classification import CnnConfig, ComponentCnn, crops_to_tensor

BATCH = 4


@pytest.fixture
def config():
    return CnnConfig(crop_size=32, widths=(8, 16), embedding_dim=12, n_classes=5, n_packages=3)


@pytest.fixture
def model(config):
    torch.manual_seed(0)
    return ComponentCnn(config).eval()


def test_two_streams_produce_separate_embeddings(model, config):
    output = model(
        torch.randn(BATCH, 3, config.crop_size, config.crop_size),
        torch.randn(BATCH, 1, config.crop_size, config.crop_size),
    )
    assert output.rgb_embedding.shape == (BATCH, config.embedding_dim)
    assert output.nir_embedding.shape == (BATCH, config.embedding_dim)
    assert output.class_logits.shape == (BATCH, config.n_classes)
    assert output.package_logits.shape == (BATCH, config.n_packages)


def test_nir_stream_changes_the_prediction(model, config):
    rgb = torch.randn(BATCH, 3, config.crop_size, config.crop_size)
    nir = torch.randn(BATCH, 1, config.crop_size, config.crop_size)
    assert not torch.allclose(
        model(rgb, nir).class_logits, model(rgb, torch.zeros_like(nir)).class_logits
    )


def test_colour_crops_become_a_normalized_nchw_tensor():
    crops = np.full((BATCH, 16, 16, 3), 255, dtype=np.uint8)
    tensor = crops_to_tensor(crops)
    assert tensor.shape == (BATCH, 3, 16, 16)
    assert tensor.dtype == torch.float32
    assert torch.allclose(tensor, torch.full_like(tensor, 2.0))


def test_mono_crops_gain_a_channel_axis():
    assert crops_to_tensor(np.zeros((BATCH, 16, 16), dtype=np.uint8)).shape == (BATCH, 1, 16, 16)


def test_unsupported_crop_shape_is_rejected():
    with pytest.raises(ValueError, match="expected"):
        crops_to_tensor(np.zeros((16, 16), dtype=np.uint8))


def test_gradients_reach_both_streams(model, config):
    output = model.train()(
        torch.randn(BATCH, 3, config.crop_size, config.crop_size),
        torch.randn(BATCH, 1, config.crop_size, config.crop_size),
    )
    output.class_logits.sum().backward()
    for stream in (model.rgb_stream, model.nir_stream):
        assert stream.project.weight.grad.abs().sum() > 0


def test_crop_too_small_for_the_stages_is_rejected():
    with pytest.raises(ValueError, match="crop_size"):
        CnnConfig(crop_size=4, widths=(8, 16, 32))
