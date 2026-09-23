"""Inspection record for one component on one board.

This is the object the technician sees and corrects. A detector produces a region, OCR
produces a string, thermal metrics produce temperatures; this record is what is stored after
those results are assembled. Technician labels attach here so immediate memory and the
training queue have a single place to read from.

The SQLite tables are not implemented yet. The record is the schema they will persist.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from epr.core.domain_models.region import ComponentRegion


class ComponentClass(str, Enum):
    RESISTOR = "resistor"
    CAPACITOR = "capacitor"
    INDUCTOR = "inductor"
    DIODE = "diode"
    TRANSISTOR = "transistor"
    MOSFET = "mosfet"
    IC = "ic"
    CONNECTOR = "connector"
    LED = "led"
    OTHER = "other"
    UNKNOWN = "unknown"


class ThermalCondition(str, Enum):
    NORMAL = "normal"
    ELEVATED = "elevated"
    HOT = "hot"
    ABNORMAL_RISE = "abnormal_rise"
    UNSTABLE = "unstable"
    UNKNOWN = "unknown"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ThermalSnapshot(_Model):
    """Temperatures belonging to one component at one moment, plus optional history."""

    t_max_c: float
    t_mean_c: float
    delta_t_c: float
    heating_rate_c_s: float = 0.0
    history_c: tuple[float, ...] = ()


class Marking(_Model):
    text: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ComponentRecord(_Model):
    """One component as stored with an inspection, independent of how it was found."""

    component_id: str = Field(min_length=1)
    board_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    frame_set_id: int = Field(ge=0)
    region: ComponentRegion
    predicted_class: ComponentClass = ComponentClass.UNKNOWN
    package: str | None = None
    marking: Marking = Field(default_factory=Marking)
    manufacturer: str | None = None
    part_number: str | None = None
    thermal: ThermalSnapshot | None = None
    thermal_condition: ThermalCondition = ThermalCondition.UNKNOWN
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    technician_label: str | None = None
    model_version: str | None = None

    @property
    def is_confirmed(self) -> bool:
        return self.technician_label is not None

    @property
    def display_class(self) -> str:
        """What the interface should show: the technician wins when they have spoken."""
        return self.technician_label or self.predicted_class.value
