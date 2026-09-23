"""One click on an inspection: temperature, optional part identity, thermal verdict."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from epr.core.domain_models.frame_set import FrameSet
from epr.core.domain_models.region import ComponentRegion
from epr.processing.regions import class_from_package, find_region_at
from epr.processing.registration import ThermalProbe, probe
from epr.processing.thermal.assess import ThermalAssessment, assess_thermal
from epr.processing.thermal.limits import ThermalLimitTable
from epr.processing.thermal.metrics import ThermalMetrics, component_metrics


@dataclass(frozen=True)
class InspectionHit:
    mark: ThermalProbe
    region: ComponentRegion | None
    metrics: ThermalMetrics
    assessment: ThermalAssessment | None

    @property
    def part_label(self) -> str:
        if self.region is None:
            return "unknown part"
        name = self.region.label or self.region.region_id
        if self.region.package:
            return f"{name} ({self.region.package})"
        return name


def inspect_click(
    frame_set: FrameSet,
    x: float,
    y: float,
    *,
    regions: Sequence[ComponentRegion] = (),
    previous: FrameSet | None = None,
    table: ThermalLimitTable | None = None,
) -> InspectionHit:
    """Probe thermal at the click, identify a known region, and assess the heat."""
    mark = probe(frame_set, x, y)
    region = find_region_at(regions, x, y)

    box = region.box if region is not None else mark.region
    prior_metrics = None
    elapsed = 0.0
    if previous is not None:
        elapsed = max(
            0.0,
            (frame_set.metadata.timestamp_ns - previous.metadata.timestamp_ns) / 1e9,
        )
        prior_metrics = component_metrics(previous, box)

    metrics = component_metrics(frame_set, box, previous=prior_metrics, elapsed_s=elapsed)

    package = region.package if region is not None else None
    component_class = class_from_package(package)
    assessment = assess_thermal(
        metrics, package=package, component_class=component_class, table=table
    )
    return InspectionHit(
        mark=mark, region=region, metrics=metrics, assessment=assessment
    )
