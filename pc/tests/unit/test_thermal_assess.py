"""Thermal limit lookup and heat assessment."""

import pytest

from epr.core.domain_models.component import ThermalCondition
from epr.core.domain_models.region import BoundingBox, ComponentRegion, RegionSource
from epr.device.simulator import SimulatedDevice
from epr.processing.inspection import inspect_click
from epr.processing.regions import find_region_at
from epr.processing.thermal.assess import assess_thermal
from epr.processing.thermal.limits import ThermalLimits, ThermalLimitTable, package_family
from epr.processing.thermal.metrics import ThermalMetrics


def test_package_family_strips_pin_counts():
    assert package_family("QFP64") == "QFP"
    assert package_family("SOIC-8") == "SOIC"
    assert package_family("0805") == "0805"
    assert package_family("SOT23") == "SOT"


def test_exact_package_beats_family_and_default():
    table = ThermalLimitTable.builtin()
    exact = table.lookup(package="QFP64")
    assert exact.key == "QFP64"
    assert table.lookup(package="UnknownThing").key == "default"


def test_normal_heat_is_normal():
    result = assess_thermal(ThermalMetrics(30.0, 28.0, 8.0, 0.1), package="QFP64")
    assert result.condition is ThermalCondition.NORMAL


def test_fast_rise_is_abnormal():
    result = assess_thermal(ThermalMetrics(50.0, 40.0, 20.0, 5.0), package="QFP64")
    assert result.condition is ThermalCondition.ABNORMAL_RISE
    assert "heating rate" in result.reason


def test_over_hot_limit_is_abnormal():
    result = assess_thermal(ThermalMetrics(90.0, 80.0, 60.0, 0.2), package="0805")
    assert result.condition is ThermalCondition.ABNORMAL_RISE


def test_limit_table_round_trips(tmp_path):
    path = ThermalLimitTable.builtin().save(tmp_path / "limits.json")
    reloaded = ThermalLimitTable.load(path)
    assert reloaded.lookup(package="SOT23").delta_hot_c == 50.0


def test_invalid_limit_order_is_rejected():
    with pytest.raises(ValueError, match="normal"):
        ThermalLimits("bad", 20.0, 10.0, 30.0, 1.0)


def test_find_region_prefers_the_smaller_box():
    outer = ComponentRegion(
        region_id="board",
        box=BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0),
        source=RegionSource.TECHNICIAN,
    )
    inner = ComponentRegion(
        region_id="U1",
        box=BoundingBox(x=0.4, y=0.4, width=0.1, height=0.1),
        source=RegionSource.SIMULATED,
        package="QFP64",
    )
    assert find_region_at([outer, inner], 0.45, 0.45) is inner


def test_inspect_click_names_the_hot_package_and_flags_heat():
    device = SimulatedDevice(
        rgb_size=(80, 60), thermal_size=(20, 16), frame_interval_s=8.0
    )
    frames = list(device.capture_many(5))
    hit = inspect_click(
        frames[-1],
        0.45,
        0.42,
        regions=device.component_regions(),
        previous=frames[-2],
    )

    assert hit.region is not None
    assert hit.region.region_id == "U1"
    assert hit.region.component_class == "ic"
    assert "ic" in hit.part_label
    assert hit.assessment is not None
    assert hit.metrics.delta_t_c > 10
    assert hit.assessment.condition is not ThermalCondition.UNKNOWN
