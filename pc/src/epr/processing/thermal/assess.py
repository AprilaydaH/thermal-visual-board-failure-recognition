"""Compare measured component heat to stored operating limits.

Recognition names the part; this module only answers whether the observed rise and rate
fit the limits for that identity. The result is a ThermalCondition plus the numbers a
technician can check by hand.
"""

from __future__ import annotations

from dataclasses import dataclass

from epr.core.domain_models.component import ComponentClass, ThermalCondition
from epr.processing.thermal.limits import ThermalLimits, ThermalLimitTable
from epr.processing.thermal.metrics import ThermalMetrics


@dataclass(frozen=True)
class ThermalAssessment:
    condition: ThermalCondition
    delta_t_c: float
    heating_rate_c_s: float
    limits: ThermalLimits
    reason: str

    @property
    def is_abnormal(self) -> bool:
        return self.condition in {
            ThermalCondition.HOT,
            ThermalCondition.ABNORMAL_RISE,
            ThermalCondition.UNSTABLE,
        }


def assess_thermal(
    metrics: ThermalMetrics,
    *,
    package: str | None = None,
    component_class: ComponentClass | str | None = None,
    table: ThermalLimitTable | None = None,
) -> ThermalAssessment:
    """Rule-based verdict from ΔT and heating rate against the limit table."""
    table = table or ThermalLimitTable.builtin()
    limits = table.lookup(package=package, component_class=component_class)
    delta = float(metrics.delta_t_c)
    rate = float(metrics.heating_rate_c_s)

    if rate > limits.heating_rate_max_c_s and delta > limits.delta_normal_c:
        return ThermalAssessment(
            condition=ThermalCondition.ABNORMAL_RISE,
            delta_t_c=delta,
            heating_rate_c_s=rate,
            limits=limits,
            reason=(
                f"heating rate {rate:.2f} C/s exceeds {limits.heating_rate_max_c_s:.2f} C/s "
                f"for {limits.key}"
            ),
        )

    if delta <= limits.delta_normal_c:
        condition = ThermalCondition.NORMAL
        reason = f"ΔT {delta:.1f} C within normal ≤ {limits.delta_normal_c:.1f} C ({limits.key})"
    elif delta <= limits.delta_elevated_c:
        condition = ThermalCondition.ELEVATED
        reason = (
            f"ΔT {delta:.1f} C elevated "
            f"({limits.delta_normal_c:.1f}–{limits.delta_elevated_c:.1f} C for {limits.key})"
        )
    elif delta <= limits.delta_hot_c:
        condition = ThermalCondition.HOT
        reason = (
            f"ΔT {delta:.1f} C hot "
            f"({limits.delta_elevated_c:.1f}–{limits.delta_hot_c:.1f} C for {limits.key})"
        )
    else:
        condition = ThermalCondition.ABNORMAL_RISE
        reason = f"ΔT {delta:.1f} C above hot limit {limits.delta_hot_c:.1f} C for {limits.key}"

    return ThermalAssessment(
        condition=condition,
        delta_t_c=delta,
        heating_rate_c_s=rate,
        limits=limits,
        reason=reason,
    )
