"""Matching a measured component size against known packages.

The detector gives a box in pixels; the distance sensor and the optics turn that into
millimetres; this turns millimetres into ranked package candidates. No training is involved,
which is the point: the answer is a lookup against published mechanical data and can be shown
to a technician as such.

Sizes are compared orientation-free, as the longer and shorter body dimension, because a part
can sit on the board at any angle.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from epr.recognition.packages.footprints import Footprint, scan_footprints

# A package is a physical standard, so the tolerance is dominated by how well the rig measures,
# not by part variation. These defaults suit a macro rig with a distance sensor.
DEFAULT_ABSOLUTE_TOLERANCE_MM = 0.25
DEFAULT_RELATIVE_TOLERANCE = 0.10


@dataclass(frozen=True)
class PackageMatch:
    family: str
    package: str
    length_mm: float
    width_mm: float
    score: float
    footprint: str
    source: str

    @property
    def label(self) -> str:
        return f"{self.family}:{self.package}"


@dataclass(frozen=True)
class SizeQuery:
    length_mm: float
    width_mm: float
    uncertainty_mm: float = 0.0

    def __post_init__(self) -> None:
        if self.length_mm <= 0 or self.width_mm <= 0:
            raise ValueError("measured size must be positive")
        longer, shorter = max(self.length_mm, self.width_mm), min(self.length_mm, self.width_mm)
        object.__setattr__(self, "length_mm", longer)
        object.__setattr__(self, "width_mm", shorter)


class PackageLibrary:
    """Package dimensions, queryable by measured size."""

    def __init__(self, footprints: Iterable[Footprint]) -> None:
        self.footprints: tuple[Footprint, ...] = tuple(footprints)
        if not self.footprints:
            raise ValueError("package library is empty")

    def __len__(self) -> int:
        return len(self.footprints)

    @classmethod
    def from_kicad(cls, root: Path | str, *, body_only: bool = False) -> PackageLibrary:
        footprints = scan_footprints(root)
        if body_only:
            footprints = (f for f in footprints if f.source == "fab")
        return cls(footprints)

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "count": len(self.footprints),
            "footprints": [asdict(f) for f in self.footprints],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | str) -> PackageLibrary:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(Footprint(**entry) for entry in payload["footprints"])

    def families(self) -> Counter[str]:
        return Counter(f.family for f in self.footprints)

    def match(
        self,
        query: SizeQuery,
        *,
        families: Sequence[str] | None = None,
        absolute_tolerance_mm: float = DEFAULT_ABSOLUTE_TOLERANCE_MM,
        relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
        body_only: bool = False,
        limit: int = 5,
    ) -> list[PackageMatch]:
        """Ranked package candidates for a measured size.

        Candidates are collapsed to one entry per family and package, so a query does not
        return forty near-identical 0805 variants.
        """
        allowed = set(families) if families else None
        tolerance_length = self._tolerance(
            query.length_mm, query.uncertainty_mm, absolute_tolerance_mm, relative_tolerance
        )
        tolerance_width = self._tolerance(
            query.width_mm, query.uncertainty_mm, absolute_tolerance_mm, relative_tolerance
        )

        best: dict[tuple[str, str], PackageMatch] = {}
        for footprint in self.footprints:
            if allowed is not None and footprint.family not in allowed:
                continue
            if body_only and footprint.source != "fab":
                continue

            deviation = max(
                abs(footprint.length_mm - query.length_mm) / tolerance_length,
                abs(footprint.width_mm - query.width_mm) / tolerance_width,
            )
            score = math.exp(-0.5 * deviation**2)
            if score <= 1e-4:
                continue

            key = (footprint.family, footprint.package)
            current = best.get(key)
            if current is None or score > current.score:
                best[key] = PackageMatch(
                    family=footprint.family,
                    package=footprint.package,
                    length_mm=footprint.length_mm,
                    width_mm=footprint.width_mm,
                    score=score,
                    footprint=footprint.name,
                    source=footprint.source,
                )

        ranked = sorted(best.values(), key=lambda m: (-m.score, m.family, m.package))
        return ranked[:limit]

    @staticmethod
    def _tolerance(value: float, uncertainty_mm: float, absolute: float, relative: float) -> float:
        return max(absolute, relative * value) + uncertainty_mm
