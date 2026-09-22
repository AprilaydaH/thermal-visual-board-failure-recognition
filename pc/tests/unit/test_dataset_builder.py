import cv2
import numpy as np
import pytest

from epr.learning.dataset_builder.crop_cache import (
    CropCache,
    build_crop_cache,
    extract_square_crop,
    fingerprint,
)
from epr.learning.dataset_builder.wacv import (
    ComponentInstance,
    build_label_space,
    is_component,
    load_wacv,
    normalize_label,
    parse_annotation,
    physical_board_id,
    split_by_board,
)

ANNOTATION = """<annotation>
  <folder>{folder}</folder>
  <filename>{folder}.jpg</filename>
  <size><width>200</width><height>150</height><depth>3</depth></size>
  {objects}
</annotation>
"""

OBJECT = """<object>
    <name>{name}</name>
    <pose>Unspecified</pose><truncated>0</truncated><difficult>0</difficult>
    <bndbox><xmin>{x0}</xmin><ymin>{y0}</ymin><xmax>{x1}</xmax><ymax>{y1}</ymax></bndbox>
  </object>"""


def write_board(root, folder, names):
    directory = root / folder
    directory.mkdir(parents=True)
    cv2.imwrite(str(directory / f"{folder}.jpg"), np.full((150, 200, 3), 60, np.uint8))
    objects = "\n  ".join(
        OBJECT.format(name=name, x0=10 + 12 * i, y0=10, x1=30 + 12 * i, y1=40)
        for i, name in enumerate(names)
    )
    (directory / f"{folder}.xml").write_text(
        ANNOTATION.format(folder=folder, objects=objects), encoding="utf-8"
    )
    return directory


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("resistor R3", "resistor"),
        ("connector CNA", "connector"),
        ("capacitor unknown", "capacitor"),
        ("emi filter EF1", "emi_filter"),
        ("led D3 3.3v", "led"),
        ("ic MK64FN1M0VLL12", "ic"),
        ("resistor network", "resistor_network"),
        ('"electrolytic capacitor"', "electrolytic_capacitor"),
        ("heatsink", "heatsink"),
    ],
)
def test_label_normalization(raw, expected):
    assert normalize_label(raw) == expected


@pytest.mark.parametrize(
    "raw", ["text \"[jtag]\"", "pins unknown", "pads R19", "test point", "unknown unknown"]
)
def test_non_components_are_filtered(raw):
    assert not is_component(normalize_label(raw))


def test_both_sides_are_the_same_physical_board():
    assert physical_board_id("ACM-109_Top") == physical_board_id("ACM-109_Bottom")
    assert physical_board_id("ArduinoMega_Bottom") == "ArduinoMega"


def test_annotation_parsing(tmp_path):
    directory = write_board(tmp_path, "BoardA_Top", ["resistor R1", "text LABEL", "ic U2"])
    instances = parse_annotation(directory / "BoardA_Top.xml")

    assert [instance.label for instance in instances] == ["resistor", "ic"]
    assert all(instance.board_id == "BoardA" for instance in instances)
    assert instances[0].box == (10, 10, 30, 40)


def test_loading_a_tree_of_boards(tmp_path):
    write_board(tmp_path, "BoardA_Top", ["resistor R1", "capacitor C1"])
    write_board(tmp_path, "BoardB_Top", ["ic U1"])
    assert len(load_wacv(tmp_path)) == 3


def test_missing_dataset_directory_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_wacv(tmp_path / "absent")


def make_instances(board_count=4, per_board=10):
    return [
        ComponentInstance(
            board_id=f"board{board}",
            image_path=f"board{board}.jpg",
            label="resistor" if index % 2 else "capacitor",
            box=(0, 0, 10, 10),
        )
        for board in range(board_count)
        for index in range(per_board)
    ]


def test_split_keeps_boards_whole():
    train, validation = split_by_board(make_instances(), validation_fraction=0.25)
    train_boards = {instance.board_id for instance in train}
    validation_boards = {instance.board_id for instance in validation}
    assert not train_boards & validation_boards
    assert len(validation_boards) == 1


def test_split_needs_more_than_one_board():
    with pytest.raises(ValueError, match="at least two boards"):
        split_by_board(make_instances(board_count=1))


def test_label_space_drops_rare_classes():
    instances = make_instances(board_count=10, per_board=20)
    instances.append(instances[0].__class__(**{**instances[0].__dict__, "label": "buzzer"}))
    assert build_label_space(instances, min_instances=50) == ("capacitor", "resistor")


def test_label_space_needs_a_frequent_class():
    with pytest.raises(ValueError, match="min_instances"):
        build_label_space(make_instances(board_count=2, per_board=2), min_instances=50)


def test_square_crop_preserves_aspect_ratio():
    image = np.zeros((100, 100, 3), np.uint8)
    image[40:60, 10:90] = 255  # a wide component

    crop = extract_square_crop(image, (10, 40, 90, 60), size=32, margin=0.0)

    assert crop.shape == (32, 32, 3)
    # Squaring the box keeps the component a horizontal bar rather than filling the frame.
    bright_rows = (crop.max(axis=(1, 2)) > 127).sum()
    assert 0 < bright_rows < 32


def test_crop_at_the_image_edge_is_padded():
    image = np.full((50, 50, 3), 200, np.uint8)
    assert extract_square_crop(image, (0, 0, 10, 10), size=16).shape == (16, 16, 3)
    assert extract_square_crop(image, (45, 45, 50, 50), size=16).shape == (16, 16, 3)


def test_crop_cache_round_trip(tmp_path):
    write_board(tmp_path / "raw", "BoardA_Top", ["resistor R1", "capacitor C1"])
    write_board(tmp_path / "raw", "BoardB_Top", ["resistor R2", "capacitor C2"])
    instances = load_wacv(tmp_path / "raw")

    cache = build_crop_cache(
        instances, tmp_path / "cache", label_space=["capacitor", "resistor"], crop_size=16
    )

    assert len(cache) == 4
    assert cache.crops.shape == (4, 16, 16, 3)
    assert set(np.unique(cache.labels)) == {0, 1}
    assert cache.manifest.board_ids == ["BoardA", "BoardB"]

    reloaded = CropCache.load(tmp_path / "cache")
    assert reloaded.manifest.fingerprint == cache.manifest.fingerprint
    assert reloaded.label_space == ("capacitor", "resistor")


def test_instances_outside_the_label_space_are_dropped(tmp_path):
    write_board(tmp_path / "raw", "BoardA_Top", ["resistor R1", "capacitor C1"])
    instances = load_wacv(tmp_path / "raw")

    cache = build_crop_cache(
        instances, tmp_path / "cache", label_space=["resistor"], crop_size=16
    )
    assert len(cache) == 1


def test_empty_label_space_is_rejected(tmp_path):
    write_board(tmp_path / "raw", "BoardA_Top", ["resistor R1"])
    instances = load_wacv(tmp_path / "raw")
    with pytest.raises(ValueError, match="no instance matches"):
        build_crop_cache(instances, tmp_path / "cache", label_space=["buzzer"], crop_size=16)


def test_fingerprint_tracks_content_and_geometry():
    instances = make_instances(board_count=2, per_board=2)
    assert fingerprint(instances, 64, 0.1) == fingerprint(list(reversed(instances)), 64, 0.1)
    assert fingerprint(instances, 64, 0.1) != fingerprint(instances, 32, 0.1)
    assert fingerprint(instances, 64, 0.1) != fingerprint(instances[:-1], 64, 0.1)
