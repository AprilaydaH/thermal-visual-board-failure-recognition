"""Reads the WACV 2019 PCB component dataset.

https://sites.google.com/view/chiawen-kuo/home/pcb-component-detection

The annotations are PASCAL VOC XML, one file per board scan, with names of the form
``resistor R3``: a component type followed by its reference designator. Only the type is
useful as a label.

This dataset trains the RGB stream of the component CNN. It cannot train the LNN, which needs
synchronized NIR and radiometric thermal over time. Check the dataset licence before shipping
anything derived from it.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ElementTree
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Annotations that mark silkscreen, bare copper or an unidentifiable blob rather than a part.
# Matched on the leading token, because the annotation carries the literal silkscreen string:
# `text "[jtag]"`, `test point`, `pins unknown`.
NON_COMPONENT_HEADS = frozenset(
    {"text", "pins", "pads", "unknown", "test", "hole", "via", "component", "silkscreen"}
)

# A trailing `unknown` means the annotator could not read the designator, not that the
# component type is unknown. Dropping the token recovers several hundred real parts.
UNREADABLE_TOKENS = frozenset({"unknown", "unk", "?"})

# A reference designator is a short upper-case token: R10, CN15, U1A, but also CNA, which has
# no digits at all. Component types in this dataset are written in lower case, so case is the
# reliable discriminator rather than the presence of a number.
DESIGNATOR = re.compile(r"^[A-Z0-9][A-Z0-9/_-]{0,5}$")

# Board scans are named like ACM-109_Top and ACM-109_Bottom. Both sides are the same physical
# PCB, so they must land on the same side of a split.
SIDE_SUFFIX = re.compile(r"[_-](top|bottom|front|back)$", re.IGNORECASE)

MIN_BOX_PIXELS = 4


@dataclass(frozen=True)
class ComponentInstance:
    board_id: str
    image_path: Path
    label: str
    box: tuple[int, int, int, int]

    @property
    def width(self) -> int:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]


def normalize_label(name: str) -> str:
    """``resistor R3`` becomes ``resistor``, ``capacitor unknown`` becomes ``capacitor``."""
    cleaned = name.strip().strip("\"'").replace('"', " ").replace("'", " ")
    tokens = [token for token in re.split(r"[\s_]+", cleaned) if token]

    # Trailing designators, part numbers and values are stripped: `led D3 3.3v` and
    # `ic MK64FN1M0VLL12` are both wanted as their type. Type words such as `resistor_network`
    # carry no digits and survive.
    while len(tokens) > 1 and (
        DESIGNATOR.match(tokens[-1])
        or tokens[-1].lower() in UNREADABLE_TOKENS
        or re.search(r"\d", tokens[-1])
    ):
        tokens.pop()
    return "_".join(tokens).lower()


def is_component(label: str) -> bool:
    return bool(label) and label.split("_")[0] not in NON_COMPONENT_HEADS


def physical_board_id(scan_name: str) -> str:
    """Strip the side suffix so both scans of one PCB share an identifier."""
    return SIDE_SUFFIX.sub("", scan_name).strip()


def parse_annotation(xml_path: Path, *, image_dir: Path | None = None) -> list[ComponentInstance]:
    root = ElementTree.parse(xml_path).getroot()
    image_dir = image_dir or xml_path.parent

    filename = root.findtext("filename")
    image_path = image_dir / filename if filename else _guess_image(xml_path)
    if image_path is None or not image_path.exists():
        logger.warning("no image found for %s", xml_path)
        return []

    board_id = physical_board_id(xml_path.parent.name)
    instances = []
    for element in root.iter("object"):
        box = element.find("bndbox")
        name = element.findtext("name")
        if box is None or not name:
            continue

        label = normalize_label(name)
        if not is_component(label):
            continue

        try:
            x0, y0, x1, y1 = (
                int(float(box.findtext(tag))) for tag in ("xmin", "ymin", "xmax", "ymax")
            )
        except (TypeError, ValueError):
            logger.warning("unreadable bounding box in %s", xml_path)
            continue

        if x1 - x0 < MIN_BOX_PIXELS or y1 - y0 < MIN_BOX_PIXELS:
            continue

        instances.append(
            ComponentInstance(
                board_id=board_id,
                image_path=image_path,
                label=label,
                box=(x0, y0, x1, y1),
            )
        )
    return instances


def load_wacv(root: Path | str) -> list[ComponentInstance]:
    """Load every annotated component under the unpacked dataset directory."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {root}")

    instances: list[ComponentInstance] = []
    for xml_path in sorted(root.rglob("*.xml")):
        instances.extend(parse_annotation(xml_path))

    if not instances:
        raise ValueError(f"no annotations found under {root}")
    logger.info("loaded %d components from %d boards", len(instances), len(boards(instances)))
    return instances


def boards(instances: Iterable[ComponentInstance]) -> set[str]:
    return {instance.board_id for instance in instances}


def label_histogram(instances: Iterable[ComponentInstance]) -> Counter[str]:
    return Counter(instance.label for instance in instances)


def build_label_space(
    instances: Sequence[ComponentInstance], *, min_instances: int = 50, max_classes: int = 20
) -> tuple[str, ...]:
    """Derive the taxonomy from the data instead of hard-coding it.

    The distribution is extremely skewed, so rare types are dropped rather than left to
    contribute a handful of examples that the model cannot learn and the metrics cannot
    measure.
    """
    histogram = label_histogram(instances)
    kept = [label for label, count in histogram.most_common(max_classes) if count >= min_instances]
    if not kept:
        raise ValueError("no label reaches min_instances")
    return tuple(sorted(kept))


def split_by_board(
    instances: Sequence[ComponentInstance],
    *,
    validation_fraction: float = 0.25,
    seed: int = 0,
) -> tuple[list[ComponentInstance], list[ComponentInstance]]:
    """Split by physical PCB, never by image.

    Crops from one board are highly correlated: same illumination, same component stock, often
    the same part repeated. Splitting by image would report a score that says nothing about an
    unseen board.
    """
    import random

    board_ids = sorted(boards(instances))
    if len(board_ids) < 2:
        raise ValueError("need at least two boards to split")

    count = max(1, round(len(board_ids) * validation_fraction))
    validation_boards = set(random.Random(seed).sample(board_ids, count))

    train = [instance for instance in instances if instance.board_id not in validation_boards]
    validation = [instance for instance in instances if instance.board_id in validation_boards]
    if not train:
        raise ValueError("validation fraction leaves no training boards")
    return train, validation


def _guess_image(xml_path: Path) -> Path | None:
    for suffix in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"):
        candidate = xml_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    images = [
        path
        for path in xml_path.parent.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    ]
    return images[0] if len(images) == 1 else None
