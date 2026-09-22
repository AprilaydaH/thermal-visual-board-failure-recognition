import numpy as np
import pytest

from epr.device.simulator import SimulatedDevice
from epr.processing.thermal import (
    celsius_to_raw,
    component_metrics,
    elapsed_seconds,
    raw_to_celsius,
    sequence_metrics,
)

HOT = "U1"
COLD = "C1"


@pytest.fixture
def device():
    return SimulatedDevice(rgb_size=(128, 96), thermal_size=(64, 48), frame_interval_s=5.0)


@pytest.fixture
def regions(device):
    return {region.region_id: region for region in device.component_regions()}


def test_raw_and_celsius_round_trip():
    celsius = np.array([[-20.0, 0.0, 21.5, 120.0]], dtype=np.float32)
    np.testing.assert_allclose(raw_to_celsius(celsius_to_raw(celsius)), celsius, atol=0.01)


def test_heated_component_is_warmer_than_a_passive_one(device, regions):
    frame_sets = list(device.capture_many(4))
    hot = sequence_metrics(frame_sets, regions[HOT].box)[-1]
    cold = sequence_metrics(frame_sets, regions[COLD].box)[-1]
    assert hot.t_max_c > cold.t_max_c + 5.0
    assert hot.t_max_c >= hot.t_mean_c


def test_delta_t_is_measured_against_ambient(device, regions):
    frame_set = device.capture()
    metrics = component_metrics(frame_set, regions[HOT].box)
    ambient = frame_set.metadata.environment.ambient_temperature_c
    assert metrics.delta_t_c == pytest.approx(metrics.t_max_c - ambient)


def test_heating_rate_needs_two_frame_sets(device, regions):
    metrics = sequence_metrics(list(device.capture_many(4)), regions[HOT].box)
    assert metrics[0].heating_rate_c_s == 0.0
    assert metrics[1].heating_rate_c_s > 0.0
    # Heating follows an exponential approach, so the rate falls as the part saturates.
    assert metrics[1].heating_rate_c_s > metrics[-1].heating_rate_c_s


def test_elapsed_seconds_uses_capture_timestamps(device):
    deltas = elapsed_seconds(list(device.capture_many(3)))
    assert deltas[0] == 0.0
    np.testing.assert_allclose(deltas[1:], [5.0, 5.0], atol=1e-6)


def test_metrics_vector_has_four_terms(device, regions):
    metrics = component_metrics(device.capture(), regions[HOT].box)
    assert metrics.as_array().shape == (4,)
