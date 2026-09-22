import numpy as np
import pytest

from epr.recognition.detection.heatmap import (
    average_precision,
    draw_gaussian,
    gaussian_radius,
    iou_matrix,
    non_maximum_suppression,
    render_targets,
)


def test_radius_grows_with_component_size():
    assert gaussian_radius(4, 4) < gaussian_radius(20, 20) < gaussian_radius(80, 80)


def test_radius_is_never_degenerate():
    assert gaussian_radius(0.5, 0.5) >= 1


def test_gaussian_peaks_at_the_centre():
    heatmap = np.zeros((20, 20), np.float32)
    draw_gaussian(heatmap, 10, 10, radius=3)

    assert heatmap[10, 10] == pytest.approx(1.0)
    assert heatmap[10, 13] < heatmap[10, 11] < heatmap[10, 10]
    assert heatmap[0, 0] == 0.0


def test_overlapping_gaussians_keep_the_maximum():
    heatmap = np.zeros((20, 20), np.float32)
    draw_gaussian(heatmap, 10, 10, radius=3)
    draw_gaussian(heatmap, 11, 10, radius=3)

    assert heatmap[10, 10] == pytest.approx(1.0)
    assert heatmap[10, 11] == pytest.approx(1.0)


def test_gaussian_at_the_edge_is_clipped_not_wrapped():
    heatmap = np.zeros((20, 20), np.float32)
    draw_gaussian(heatmap, 0, 0, radius=4)

    assert heatmap[0, 0] == pytest.approx(1.0)
    assert heatmap[19, 19] == 0.0


def test_targets_encode_centre_size_and_offset():
    boxes = np.array([[20.0, 40.0, 36.0, 56.0]])
    targets = render_targets(boxes, (128, 128), stride=4)

    assert targets["heatmap"].shape == (1, 32, 32)
    assert targets["mask"].sum() == 1

    row, column = np.argwhere(targets["mask"][0] == 1)[0]
    # Centre is (28, 48) in pixels, so cell (7, 12) at stride 4, with no sub-cell offset.
    assert (row, column) == (12, 7)
    assert targets["size"][:, row, column] == pytest.approx([4.0, 4.0])
    assert targets["offset"][:, row, column] == pytest.approx([0.0, 0.0])
    assert targets["heatmap"][0, row, column] == pytest.approx(1.0)


def test_sub_cell_offset_is_recorded():
    targets = render_targets(np.array([[10.0, 10.0, 22.0, 22.0]]), (64, 64), stride=4)
    row, column = np.argwhere(targets["mask"][0] == 1)[0]
    assert targets["offset"][:, row, column] == pytest.approx([0.0, 0.0])

    targets = render_targets(np.array([[11.0, 11.0, 23.0, 23.0]]), (64, 64), stride=4)
    row, column = np.argwhere(targets["mask"][0] == 1)[0]
    assert targets["offset"][0, row, column] == pytest.approx(0.25)


def test_degenerate_and_outside_boxes_are_skipped():
    targets = render_targets(
        np.array([[10.0, 10.0, 10.0, 20.0], [500.0, 500.0, 520.0, 520.0]]), (64, 64), stride=4
    )
    assert targets["mask"].sum() == 0


def test_no_boxes_gives_empty_targets():
    targets = render_targets(np.zeros((0, 4)), (64, 64), stride=4)
    assert targets["heatmap"].sum() == 0
    assert targets["mask"].sum() == 0


def test_iou_of_identical_and_disjoint_boxes():
    box = np.array([[0.0, 0.0, 10.0, 10.0]])
    assert iou_matrix(box, box)[0, 0] == pytest.approx(1.0)
    assert iou_matrix(box, np.array([[20.0, 20.0, 30.0, 30.0]]))[0, 0] == pytest.approx(0.0)
    # Half overlap: intersection 50, union 150.
    assert iou_matrix(box, np.array([[5.0, 0.0, 15.0, 10.0]]))[0, 0] == pytest.approx(1 / 3)


def test_iou_with_nothing_is_empty():
    assert iou_matrix(np.zeros((0, 4)), np.ones((2, 4))).shape == (0, 2)


def test_suppression_keeps_the_best_of_a_cluster():
    boxes = np.array(
        [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [50.0, 50.0, 60.0, 60.0]]
    )
    keep = non_maximum_suppression(boxes, np.array([0.7, 0.9, 0.8]), threshold=0.45)

    assert list(keep) == [1, 2]


def test_suppression_of_nothing():
    assert len(non_maximum_suppression(np.zeros((0, 4)), np.zeros(0))) == 0


def test_perfect_detections_score_one():
    truth = np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    ap, recall, precision = average_precision(truth, np.array([0.9, 0.8]), truth)

    assert ap == pytest.approx(1.0)
    assert recall == pytest.approx(1.0)
    assert precision == pytest.approx(1.0)


def test_duplicate_detections_are_false_positives():
    truth = np.array([[0.0, 0.0, 10.0, 10.0]])
    predicted = np.array([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]])
    ap, recall, precision = average_precision(predicted, np.array([0.9, 0.8]), truth)

    assert recall == pytest.approx(1.0)
    assert precision == pytest.approx(0.5)
    assert ap == pytest.approx(1.0)


def test_a_miss_lowers_recall():
    truth = np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    ap, recall, _ = average_precision(truth[:1], np.array([0.9]), truth)

    assert recall == pytest.approx(0.5)
    assert ap == pytest.approx(0.5)


def test_poorly_localized_boxes_do_not_match():
    truth = np.array([[0.0, 0.0, 10.0, 10.0]])
    shifted = np.array([[8.0, 8.0, 18.0, 18.0]])
    _, recall, precision = average_precision(shifted, np.array([0.9]), truth)

    assert recall == 0.0
    assert precision == 0.0
