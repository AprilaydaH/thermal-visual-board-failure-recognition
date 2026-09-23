"""Thermal operating limits looked up by package or component class.

These are **candidate bench defaults**, not datasheet guarantees. A technician or a later
datasheet import replaces a row; the assessor only compares measured heat to whatever is
stored here. That separation is deliberate: recognition names the part, this table says what
heat is allowed, and the verdict is explainable.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from epr.core.domain_models.component import ComponentClass

# Strip trailing pin counts so QFP64 and TQFP-64 both hit the QFP family.
_PACKAGE_FAMILY = re.compile(
    r"^(?P<head>QFP|LQFP|TQFP|SOIC|SOP|SSOP|TSOP|SOT|TO|BGA|DFN|QFN|DIP|HEADER|"
    r"CONN|RESISTOR|CAPACITOR|\d{4})",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ThermalLimits:
    """Allowed temperature rise above ambient for one identity."""

    key: str
    delta_normal_c: float
    delta_elevated_c: float
    delta_hot_c: float
    heating_rate_max_c_s: float
    source: str = "default"

    def __post_init__(self) -> None:
        if not (0 <= self.delta_normal_c <= self.delta_elevated_c <= self.delta_hot_c):
            raise ValueError(
                f"limits for {self.key} must satisfy "
                "0 ≤ normal ≤ elevated ≤ hot"
            )
        if self.heating_rate_max_c_s < 0:
            raise ValueError("heating rate limit must be non-negative")


# Conservative stand-mounted inspection defaults. Tight for passives and connectors;
# looser for packages that normally dissipate heat after power-on.
DEFAULT_LIMITS: tuple[ThermalLimits, ...] = (
    ThermalLimits("0805", 5.0, 12.0, 25.0, 0.4, "prototype"),
    ThermalLimits("0603", 5.0, 12.0, 25.0, 0.4, "prototype"),
    ThermalLimits("0402", 5.0, 10.0, 20.0, 0.3, "prototype"),
    ThermalLimits("1206", 6.0, 15.0, 30.0, 0.5, "prototype"),
    ThermalLimits("SOT", 15.0, 30.0, 50.0, 1.5, "prototype"),
    ThermalLimits("SOT23", 15.0, 30.0, 50.0, 1.5, "prototype"),
    ThermalLimits("SOIC", 12.0, 25.0, 45.0, 1.2, "prototype"),
    ThermalLimits("SOIC8", 12.0, 25.0, 45.0, 1.2, "prototype"),
    ThermalLimits("QFP", 15.0, 30.0, 50.0, 1.5, "prototype"),
    ThermalLimits("QFP64", 15.0, 30.0, 50.0, 1.5, "prototype"),
    ThermalLimits("HEADER", 4.0, 8.0, 15.0, 0.3, "prototype"),
    ThermalLimits("resistor", 8.0, 20.0, 40.0, 0.8, "prototype"),
    ThermalLimits("capacitor", 5.0, 12.0, 25.0, 0.4, "prototype"),
    ThermalLimits("transistor", 15.0, 30.0, 55.0, 1.5, "prototype"),
    ThermalLimits("mosfet", 20.0, 40.0, 70.0, 2.0, "prototype"),
    ThermalLimits("ic", 15.0, 30.0, 50.0, 1.5, "prototype"),
    ThermalLimits("connector", 4.0, 8.0, 15.0, 0.3, "prototype"),
    ThermalLimits("default", 15.0, 30.0, 50.0, 1.0, "prototype"),
)


def package_family(package: str | None) -> str | None:
    if not package:
        return None
    cleaned = package.strip().upper().replace("-", "").replace("_", "")
    match = _PACKAGE_FAMILY.match(cleaned)
    if not match:
        return cleaned or None
    head = match.group("head").upper()
    if head.isdigit():
        return head  # 0805-style
    return head


class ThermalLimitTable:
    """Lookup table keyed by package code, package family, or component class."""

    def __init__(self, limits: Mapping[str, ThermalLimits] | None = None) -> None:
        if limits is None:
            self._by_key = {entry.key.lower(): entry for entry in DEFAULT_LIMITS}
        else:
            self._by_key = {key.lower(): value for key, value in limits.items()}
        if "default" not in self._by_key:
            raise ValueError("limit table needs a 'default' row")

    def __len__(self) -> int:
        return len(self._by_key)

    @classmethod
    def builtin(cls) -> ThermalLimitTable:
        return cls()

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ordered = sorted(self._by_key.values(), key=lambda limit: limit.key)
        payload = {
            "version": 1,
            "limits": [asdict(limit) for limit in ordered],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | str) -> ThermalLimitTable:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        limits = {
            entry["key"]: ThermalLimits(**entry) for entry in payload["limits"]
        }
        return cls(limits)

    def lookup(
        self,
        *,
        package: str | None = None,
        component_class: ComponentClass | str | None = None,
    ) -> ThermalLimits:
        """Most specific match wins: exact package, then family, then class, then default."""
        candidates: list[str] = []
        if package:
            candidates.append(package)
            family = package_family(package)
            if family and family.lower() != package.lower():
                candidates.append(family)
        if component_class is not None:
            name = (
                component_class.value
                if isinstance(component_class, ComponentClass)
                else str(component_class)
            )
            candidates.append(name)
        candidates.append("default")

        for key in candidates:
            found = self._by_key.get(key.lower())
            if found is not None:
                return found
        return self._by_key["default"]
