"""G1: an RGB click maps onto a thermal pixel and a temperature."""

import numpy as np
import pytest

from epr.device.simulator import SimulatedDevice
from epr.processing.registration import Registration, box_around, point_to_pixel, probe


def test_identity_registration_keeps_the_normalized_point():
    assert Registration().to_thermal(0.25, 0.8) == (0.25, 0.8)


def test_homography_slot_can_shift_the_point():
    # Translate +0.1 in x in normalized space.
    matrix = np.array([[1.0, 0.0, 0.1], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert Registration(rgb_to_thermal=matrix).to_thermal(0.2, 0.5) == pytest.approx((0.3, 0.5))


def test_a_point_converts_to_a_pixel_inside_the_image():
    assert point_to_pixel(0.0, 0.0, 160, 120) == (0, 0)
    assert point_to_pixel(1.0, 1.0, 160, 120) == (159, 119)


def test_out_of_range_coordinates_are_rejected():
    with pytest.raises(ValueError, match="normalized"):
        point_to_pixel(-0.01, 0.5, 10, 10)


def test_click_on_the_hot_package_is_hotter_than_the_board():
    device = SimulatedDevice(
        rgb_size=(80, 60), thermal_size=(20, 16), seed=0, frame_interval_s=8.0
    )
    frame_set = list(device.capture_many(5))[-1]
    # U1 is centred near 0.45, 0.42 in the simulator.
    hot = probe(frame_set, 0.45, 0.42)
    cold = probe(frame_set, 0.08, 0.08)

    assert hot.temperature_c > cold.temperature_c + 5
    assert hot.region_max_c >= hot.temperature_c
    assert hot.thermal_pixel[0] < 20 and hot.thermal_pixel[1] < 16


def test_box_around_a_corner_stays_inside_the_frame():
    box = box_around(0.0, 1.0, size=0.1)
    assert box.x >= 0 and box.y + box.height <= 1.0
