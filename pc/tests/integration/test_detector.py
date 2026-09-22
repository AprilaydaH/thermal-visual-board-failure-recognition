"""The detector network, its decoding, and a training loop that has to actually find parts."""

import numpy as np
import pytest
import torch

from epr.core.domain_models.region import RegionSource
from epr.learning.training.detector import (
    BoardSample,
    DetectorTrainingConfig,
    TileDataset,
    evaluate_detector,
    focal_loss,
    load_detector,
    masked_l1,
    save_detector,
    train_detector,
)
from epr.recognition.detection import (
    OUTPUT_STRIDE,
    ComponentDetector,
    DetectorConfig,
    decode,
    detect,
    detect_regions,
    image_to_tensor,
    render_targets,
)
from epr.recognition.runtime import seed_everything

TILE = 64


@pytest.fixture
def config():
    return DetectorConfig(widths=(8, 16, 24, 32), head_width=16, tile_size=TILE)


def synthetic_board(seed: int = 0, size: int = 128, count: int = 12):
    """Bright rectangles on a dark laminate: crude, but it has centres and sizes to regress."""
    rng = np.random.default_rng(seed)
    image = rng.integers(20, 45, (size, size, 3), dtype=np.uint8)
    boxes = []
    for _ in range(count):
        width, height = int(rng.integers(8, 16)), int(rng.integers(8, 16))
        x0 = int(rng.integers(2, size - width - 2))
        y0 = int(rng.integers(2, size - height - 2))
        if any(
            x0 < b[2] + 4 and b[0] < x0 + width + 4 and y0 < b[3] + 4 and b[1] < y0 + height + 4
            for b in boxes
        ):
            continue
        image[y0 : y0 + height, x0 : x0 + width] = rng.integers(180, 255, 3, dtype=np.uint8)
        boxes.append([x0, y0, x0 + width, y0 + height])

    return BoardSample(
        board_id=f"synthetic{seed}",
        image=image,
        boxes=np.array(boxes, dtype=np.float32),
    )


def test_output_shapes_are_at_stride_four(config):
    model = ComponentDetector(config)
    output = model(torch.zeros(2, 3, TILE, TILE))

    cells = TILE // OUTPUT_STRIDE
    assert output.heatmap.shape == (2, 1, cells, cells)
    assert output.size.shape == (2, 2, cells, cells)
    assert output.offset.shape == (2, 2, cells, cells)


def test_the_detector_is_fully_convolutional(config):
    """The same weights must run on a training tile and on a whole board."""
    model = ComponentDetector(config).eval()
    with torch.inference_mode():
        assert model(torch.zeros(1, 3, 64, 64)).heatmap.shape[-2:] == (16, 16)
        assert model(torch.zeros(1, 3, 128, 96)).heatmap.shape[-2:] == (32, 24)


def test_untrained_heatmap_starts_near_the_component_prior(config):
    model = ComponentDetector(config).eval()
    with torch.inference_mode():
        scores = torch.sigmoid(model(torch.zeros(1, 3, TILE, TILE)).heatmap)
    assert scores.mean() < 0.1


def test_configuration_is_validated():
    with pytest.raises(ValueError, match="four encoder stages"):
        DetectorConfig(widths=(8, 16))
    with pytest.raises(ValueError, match="multiple of"):
        DetectorConfig(tile_size=100)


def test_decoding_recovers_the_boxes_that_were_encoded():
    """Render targets, feed them back as if perfectly predicted, and expect the boxes out."""
    boxes = np.array([[16.0, 24.0, 32.0, 40.0], [60.0, 60.0, 76.0, 84.0]], dtype=np.float32)
    targets = render_targets(boxes, (128, 128), OUTPUT_STRIDE)

    logits = torch.from_numpy(targets["heatmap"]).clamp(1e-4, 1 - 1e-4).logit().unsqueeze(0)
    output = type("Output", (), {})()
    output.heatmap = logits
    output.size = torch.from_numpy(targets["size"]).unsqueeze(0)
    output.offset = torch.from_numpy(targets["offset"]).unsqueeze(0)

    detections = decode(output, score_threshold=0.9, max_detections=10)[0]

    assert len(detections) == 2
    recovered = np.sort(detections.boxes, axis=0)
    np.testing.assert_allclose(recovered, np.sort(boxes, axis=0), atol=1e-3)


def test_padding_makes_any_image_size_acceptable():
    tensor = image_to_tensor(np.zeros((701, 560, 3), np.uint8))
    assert tensor.shape[-2] % 16 == 0
    assert tensor.shape[-1] % 16 == 0
    assert tensor.shape[-2] >= 701 and tensor.shape[-1] >= 560


def test_tiles_contain_components_and_matching_targets():
    dataset = TileDataset([synthetic_board()], tile_size=TILE, length=8, seed=1)
    with_components = sum(int(dataset[i]["mask"].sum()) > 0 for i in range(len(dataset)))

    assert with_components >= 7
    sample = dataset[0]
    assert sample["image"].shape == (3, TILE, TILE)
    assert sample["heatmap"].shape == (1, TILE // OUTPUT_STRIDE, TILE // OUTPUT_STRIDE)


def test_tile_dataset_needs_boards():
    with pytest.raises(ValueError, match="no boards"):
        TileDataset([], tile_size=TILE)


def test_boxes_cut_by_a_tile_edge_are_dropped():
    """A component half outside the tile must not teach the regressor a smaller size."""
    board = BoardSample(
        board_id="edge",
        image=np.zeros((64, 64, 3), np.uint8),
        boxes=np.array([[-10.0, 10.0, 4.0, 24.0]], dtype=np.float32),
    )
    dataset = TileDataset([board], tile_size=64, length=1, augment=False, seed=0)
    assert dataset[0]["mask"].sum() == 0


def test_focal_loss_rewards_correct_peaks():
    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, 4, 4] = 1.0

    confident = torch.full((1, 1, 8, 8), -6.0)
    confident[0, 0, 4, 4] = 6.0
    wrong = torch.full((1, 1, 8, 8), -6.0)
    wrong[0, 0, 1, 1] = 6.0

    assert focal_loss(confident, target) < focal_loss(torch.zeros(1, 1, 8, 8), target)
    assert focal_loss(confident, target) < focal_loss(wrong, target)


def test_regression_loss_only_counts_component_cells():
    predicted = torch.ones(1, 2, 4, 4)
    target = torch.zeros(1, 2, 4, 4)
    mask = torch.zeros(1, 1, 4, 4)

    assert masked_l1(predicted, target, mask) == 0.0
    mask[0, 0, 2, 2] = 1.0
    assert masked_l1(predicted, target, mask) > 0.0


@pytest.mark.slow
def test_training_finds_components_on_held_out_boards(config):
    seed_everything(0)
    train_boards = [synthetic_board(seed) for seed in range(6)]
    validation_boards = [synthetic_board(seed) for seed in range(100, 102)]

    model = ComponentDetector(config)
    before = evaluate_detector(model, validation_boards)
    result = train_detector(
        model,
        train_boards,
        validation_boards,
        DetectorTrainingConfig(epochs=12, tiles_per_epoch=96, batch_size=8, learning_rate=3e-3),
    )

    assert result.train_losses[-1] < result.train_losses[0]
    assert result.best.average_precision > before.average_precision
    assert result.best.recall > 0.5


def test_detection_produces_normalized_regions(config):
    board = synthetic_board()
    model = ComponentDetector(config)

    regions = detect_regions(model, board.image, score_threshold=0.0)

    assert regions, "a zero threshold should return the top peaks"
    for region in regions:
        assert region.source is RegionSource.DETECTOR
        assert 0.0 <= region.box.x <= 1.0
        assert region.box.x + region.box.width <= 1.0
        assert region.box.y + region.box.height <= 1.0
    assert len({region.region_id for region in regions}) == len(regions)


def test_a_high_threshold_returns_nothing_rather_than_failing(config):
    model = ComponentDetector(config)
    assert len(detect(model, synthetic_board().image, score_threshold=0.999)) == 0


def test_checkpoint_round_trip(config, tmp_path):
    model = ComponentDetector(config)
    board = synthetic_board()
    before = detect(model, board.image, score_threshold=0.0)

    path = save_detector(
        tmp_path / "detector.pt",
        model,
        dataset_fingerprint="wacv:test",
        result=train_detector(
            model,
            [board],
            [board],
            DetectorTrainingConfig(epochs=0, tiles_per_epoch=8, batch_size=4),
        ),
    )
    restored = load_detector(path)
    after = detect(restored, board.image, score_threshold=0.0)

    assert restored.config == config
    np.testing.assert_allclose(before.boxes, after.boxes, atol=1e-4)
