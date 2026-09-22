"""A package prior for a detected region, from its physical size.

This is the bridge between the detector and the package library: measure the region in
millimetres, look the size up, and return ranked candidates. It is deliberately not a learned
model. Appearance and physical size fail in different places, so keeping them separate means
the fusion model can weigh them, and a technician can be shown why a part was called an 0805.
"""

from __future__ import annotations

from dataclasses import dataclass

from epr.core.domain_models.frame_set import FrameSetMetadata
from epr.core.domain_models.region import ComponentRegion
from epr.processing.scale import Scale, box_size_mm, size_uncertainty_mm
from epr.recognition.packages.library import PackageLibrary, PackageMatch, SizeQuery

# The distance sensor is specified in millimetres; this is a plausible bench figure until the
# real sensor is characterized, and it only widens the matching tolerance.
DEFAULT_DISTANCE_UNCERTAINTY_MM = 2.0


@dataclass(frozen=True)
class PackagePrior:
    region_id: str
    length_mm: float
    width_mm: float
    uncertainty_mm: float
    candidates: tuple[PackageMatch, ...]

    @property
    def best(self) -> PackageMatch | None:
        return self.candidates[0] if self.candidates else None

    @property
    def is_confident(self) -> bool:
        """One candidate clearly ahead of the next.

        Size alone cannot separate packages that share dimensions, so a prior is only worth
        acting on when the runner-up is well behind.
        """
        if len(self.candidates) < 2:
            return bool(self.candidates)
        return (
            self.candidates[0].score > 0.5
            and self.candidates[1].score < 0.5 * self.candidates[0].score
        )


def measure_region(
    region: ComponentRegion,
    metadata: FrameSetMetadata,
    scale: Scale,
    *,
    distance_uncertainty_mm: float = DEFAULT_DISTANCE_UNCERTAINTY_MM,
) -> SizeQuery:
    """Physical size of a region, using the distance reported with the frame set."""
    descriptor = metadata.rgb.image
    distance_mm = metadata.geometry.distance_mm

    mm_per_pixel = scale.mm_per_pixel(distance_mm)
    length_mm, width_mm = box_size_mm(region.box, descriptor.width, descriptor.height, mm_per_pixel)
    return SizeQuery(
        length_mm=length_mm,
        width_mm=width_mm,
        uncertainty_mm=size_uncertainty_mm(length_mm, distance_mm, distance_uncertainty_mm),
    )


def package_prior(
    region: ComponentRegion,
    metadata: FrameSetMetadata,
    scale: Scale,
    library: PackageLibrary,
    *,
    distance_uncertainty_mm: float = DEFAULT_DISTANCE_UNCERTAINTY_MM,
    limit: int = 5,
    **match_options: object,
) -> PackagePrior:
    query = measure_region(region, metadata, scale, distance_uncertainty_mm=distance_uncertainty_mm)
    candidates = library.match(query, limit=limit, **match_options)  # type: ignore[arg-type]
    return PackagePrior(
        region_id=region.region_id,
        length_mm=query.length_mm,
        width_mm=query.width_mm,
        uncertainty_mm=query.uncertainty_mm,
        candidates=tuple(candidates),
    )
