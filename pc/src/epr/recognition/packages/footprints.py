"""Physical package dimensions read from KiCad footprint libraries.

A KiCad footprint states, in millimetres, exactly how large a part is. That is information no
photographic dataset provides and no amount of training recovers, and it separates the cases
that defeat appearance alone: an 0603 and an 0805 chip resistor look identical but are
1.6x0.8 mm and 2.0x1.25 mm.

Three outlines are available per footprint, in decreasing order of usefulness here:

- ``F.Fab`` is the component body, which is what a camera looking down sees.
- ``F.CrtYd`` is the courtyard, the body plus assembly clearance, so it overstates the part.
- the pads are copper, which for a chip part extend beyond the body at the terminals.

The source is recorded on every entry so a consumer can reject the approximate ones.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from epr.recognition.packages import sexpr

logger = logging.getLogger(__name__)

BODY_LAYER = "F.Fab"
COURTYARD_LAYER = "F.CrtYd"

# Ordered longest-prefix-first: the first rule whose prefix matches the library name wins.
FAMILY_RULES: tuple[tuple[str, str], ...] = (
    ("Capacitor_Tantalum", "capacitor"),
    ("Capacitor", "capacitor"),
    ("Resistor", "resistor"),
    ("Inductor", "inductor"),
    ("Diode", "diode"),
    ("LED", "led"),
    ("Crystal", "crystal"),
    ("Oscillator", "crystal"),
    ("Button_Switch", "switch"),
    ("Switch", "switch"),
    ("Connector", "connector"),
    ("Fuse", "fuse"),
    ("Relay", "relay"),
    ("Transformer", "transformer"),
    ("Potentiometer", "potentiometer"),
    ("Buzzer", "buzzer"),
    ("Battery", "battery"),
    ("Heatsink", "heatsink"),
    ("Sensor", "sensor"),
    ("Package_BGA", "bga"),
    ("Package_CSP", "bga"),
    ("Package_QFP", "qfp"),
    ("Package_DFN_QFN", "qfn"),
    ("Package_LGA", "lga"),
    ("Package_SON", "qfn"),
    ("Package_SO", "soic"),
    ("Package_DIP", "dip"),
    ("Package_TO_SOT_SMD", "sot"),
    ("Package_TO_SOT_THT", "to"),
    ("Package_SIP", "sip"),
    ("Module", "module"),
    ("MountingHole", "mounting"),
    ("TestPoint", "test_point"),
)

# R_0805_2012Metric, C_0603_1608Metric: the imperial code is the first four-digit group.
IMPERIAL_CODE = re.compile(r"^[A-Z]{1,3}_(\d{4})_\d{4}Metric", re.IGNORECASE)
# A token carrying dimensions or pitch rather than a package name.
DIMENSION_TOKEN = re.compile(
    r"(\d+(\.\d+)?x\d+(\.\d+)?mm|^P\d|\dmm$|Handsolder|Pad\d)", re.IGNORECASE
)


@dataclass(frozen=True)
class Footprint:
    name: str
    library: str
    family: str
    package: str
    length_mm: float
    width_mm: float
    source: str
    pad_count: int
    mounting: str

    @property
    def area_mm2(self) -> float:
        return self.length_mm * self.width_mm

    @property
    def aspect(self) -> float:
        return self.length_mm / self.width_mm if self.width_mm else 0.0


def family_for_library(library: str) -> str:
    for prefix, family in FAMILY_RULES:
        if library.startswith(prefix):
            return family
    return "other"


def package_for_name(name: str) -> str:
    """``R_0805_2012Metric`` gives ``0805``; ``TQFP-64_10x10mm_P0.5mm`` gives ``TQFP-64``."""
    imperial = IMPERIAL_CODE.match(name)
    if imperial:
        return imperial.group(1)

    tokens: list[str] = []
    for token in name.split("_"):
        if not token or DIMENSION_TOKEN.search(token):
            break
        tokens.append(token)
    return "_".join(tokens) if tokens else name


def _layer_of(node: sexpr.Node) -> str | None:
    return sexpr.atom(node, "layer")


def _points_on_layer(footprint: sexpr.Node, layer: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []

    for node in sexpr.walk(footprint):
        if not node or not isinstance(node[0], str):
            continue
        tag = node[0]
        if tag not in {"fp_line", "fp_rect", "fp_poly", "fp_circle", "fp_arc"}:
            continue
        if _layer_of(node) != layer:
            continue

        if tag in {"fp_line", "fp_rect"}:
            for key in ("start", "end"):
                point = sexpr.floats(sexpr.first(node, key))
                if point:
                    points.append(point)
        elif tag == "fp_arc":
            for key in ("start", "mid", "end"):
                point = sexpr.floats(sexpr.first(node, key))
                if point:
                    points.append(point)
        elif tag == "fp_circle":
            centre = sexpr.floats(sexpr.first(node, "center"))
            edge = sexpr.floats(sexpr.first(node, "end"))
            if centre and edge:
                radius = max(abs(edge[0] - centre[0]), abs(edge[1] - centre[1]))
                points.extend(
                    [
                        (centre[0] - radius, centre[1] - radius),
                        (centre[0] + radius, centre[1] + radius),
                    ]
                )
        elif tag == "fp_poly":
            pts = sexpr.first(node, "pts")
            if pts:
                for xy in sexpr.tagged(pts, "xy"):
                    point = sexpr.floats(xy)
                    if point:
                        points.append(point)

    return points


def _pad_points(footprint: sexpr.Node) -> tuple[list[tuple[float, float]], int]:
    points: list[tuple[float, float]] = []
    count = 0

    for pad in sexpr.tagged(footprint, "pad"):
        count += 1
        at = sexpr.floats(sexpr.first(pad, "at"))
        size = sexpr.floats(sexpr.first(pad, "size"))
        if not at or not size:
            continue

        rotation = 0.0
        at_node = sexpr.first(pad, "at")
        if at_node is not None and len(at_node) > 3:
            try:
                rotation = float(at_node[3])  # type: ignore[arg-type]
            except (TypeError, ValueError):
                rotation = 0.0

        half_x, half_y = size[0] / 2, size[1] / 2
        if round(rotation / 90) % 2:
            half_x, half_y = half_y, half_x

        points.append((at[0] - half_x, at[1] - half_y))
        points.append((at[0] + half_x, at[1] + half_y))

    return points, count


def _extent(points: Iterable[tuple[float, float]]) -> tuple[float, float] | None:
    points = list(points)
    if len(points) < 2:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    width, height = max(xs) - min(xs), max(ys) - min(ys)
    if width <= 0 or height <= 0:
        return None
    return max(width, height), min(width, height)


def parse_footprint(path: Path, *, library: str | None = None) -> Footprint | None:
    """Read one ``.kicad_mod``. Returns ``None`` if it carries no usable outline."""
    try:
        document = sexpr.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError) as error:
        logger.debug("cannot read %s: %s", path, error)
        return None

    node = next((child for child in document if isinstance(child, list) and child), None)
    if node is None or node[0] != "footprint":
        return None

    name = node[1] if len(node) > 1 and isinstance(node[1], str) else path.stem
    library = library or path.parent.name.removesuffix(".pretty")

    pad_points, pad_count = _pad_points(node)
    candidates = (
        ("fab", _extent(_points_on_layer(node, BODY_LAYER))),
        ("courtyard", _extent(_points_on_layer(node, COURTYARD_LAYER))),
        ("pads", _extent(pad_points)),
    )
    source, extent = next(((s, e) for s, e in candidates if e), ("none", None))
    if extent is None:
        return None

    attr = sexpr.first(node, "attr")
    mounting = "other"
    if attr:
        tokens = {value for value in attr[1:] if isinstance(value, str)}
        if "smd" in tokens:
            mounting = "smd"
        elif "through_hole" in tokens:
            mounting = "tht"

    return Footprint(
        name=name,
        library=library,
        family=family_for_library(library),
        package=package_for_name(name),
        length_mm=round(extent[0], 4),
        width_mm=round(extent[1], 4),
        source=source,
        pad_count=pad_count,
        mounting=mounting,
    )


def scan_footprints(root: Path | str) -> Iterator[Footprint]:
    """Walk a KiCad footprint directory, yielding every readable footprint."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"footprint directory not found: {root}")

    for path in sorted(root.rglob("*.kicad_mod")):
        footprint = parse_footprint(path)
        if footprint is not None:
            yield footprint


def default_kicad_root() -> Path | None:
    """Best guess at the installed footprint directory on this machine."""
    bases = [
        Path.home() / "AppData/Local/Programs/KiCad",
        Path("C:/Program Files/KiCad"),
        Path("/usr/share/kicad"),
        Path("/Applications/KiCad/KiCad.app/Contents/SharedSupport"),
    ]
    for base in bases:
        if not base.exists():
            continue
        for candidate in sorted(base.rglob("footprints"), reverse=True):
            if candidate.is_dir() and any(candidate.glob("*.pretty")):
                return candidate
    return None
