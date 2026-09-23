"""Hit-test a click against known component regions."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from epr.core.domain_models.region import BoundingBox, ComponentRegion, RegionSource


def region_contains(box: BoundingBox, x: float, y: float) -> bool:
    return box.x <= x <= box.x + box.width and box.y <= y <= box.y + box.height


def find_region_at(
    regions: Sequence[ComponentRegion], x: float, y: float
) -> ComponentRegion | None:
    """Innermost region under the click (smallest area wins when boxes overlap)."""
    hits = [region for region in regions if region_contains(region.box, x, y)]
    if not hits:
        return None
    return min(hits, key=lambda region: region.box.width * region.box.height)


def save_regions(path: Path | str, regions: Sequence[ComponentRegion]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "regions": [region.model_dump(mode="json") for region in regions],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_regions(path: Path | str) -> list[ComponentRegion]:
    path = Path(path)
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    regions = []
    for entry in payload.get("regions", []):
        source = entry.get("source", RegionSource.DETECTOR.value)
        regions.append(
            ComponentRegion(
                region_id=entry["region_id"],
                box=BoundingBox(**entry["box"]),
                source=RegionSource(source),
                label=entry.get("label"),
                package=entry.get("package"),
                component_class=entry.get("component_class"),
            )
        )
    return regions


def class_from_package(package: str | None) -> str | None:
    """Rough class for limit lookup when only a package family is known.

    Chip codes like 0805 are left unset so the package row in the limit table wins.
    """
    if not package:
        return None
    upper = package.upper().replace("-", "")
    if upper[:4].isdigit():
        return None
    if upper.startswith(("SOT", "TO")):
        return "transistor"
    if upper.startswith(("SOIC", "QFP", "LQFP", "TQFP", "BGA", "QFN", "DFN", "DIP")):
        return "ic"
    if upper.startswith(("HEADER", "CONN")):
        return "connector"
    return None
