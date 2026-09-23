"""Component inspection records, independent of how the region was found."""

from epr.core.domain_models.component import (
    ComponentClass,
    ComponentRecord,
    Marking,
    ThermalCondition,
    ThermalSnapshot,
)
from epr.core.domain_models.region import BoundingBox, ComponentRegion, RegionSource


def _region():
    return ComponentRegion(
        region_id="r1",
        box=BoundingBox(x=0.1, y=0.2, width=0.05, height=0.04),
        source=RegionSource.TECHNICIAN,
    )


def test_technician_label_overrides_the_prediction():
    record = ComponentRecord(
        component_id="c1",
        board_id="board-a",
        session_id="s1",
        frame_set_id=3,
        region=_region(),
        predicted_class=ComponentClass.RESISTOR,
        confidence=0.4,
        technician_label="capacitor",
        thermal=ThermalSnapshot(
            t_max_c=48.0, t_mean_c=40.0, delta_t_c=23.0, history_c=(31.2, 39.4, 48.0)
        ),
        thermal_condition=ThermalCondition.ABNORMAL_RISE,
        marking=Marking(text="1k", confidence=0.9),
    )

    assert record.is_confirmed
    assert record.display_class == "capacitor"
    assert record.thermal is not None
    assert record.thermal.history_c[-1] == 48.0


def test_unconfirmed_record_shows_the_predicted_class():
    record = ComponentRecord(
        component_id="c2",
        board_id="board-a",
        session_id="s1",
        frame_set_id=3,
        region=_region(),
    )

    assert not record.is_confirmed
    assert record.display_class == "unknown"
