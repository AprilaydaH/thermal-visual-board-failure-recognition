"""Raw recording and loading are inverses."""

import numpy as np
import pytest

from epr.core.domain_models import Channel
from epr.core.errors import StorageError
from epr.device.simulator import SimulatedDevice
from epr.storage.file_storage import RawLibrary, RawRecorder


def test_a_recorded_frame_set_loads_unchanged(tmp_path):
    original = SimulatedDevice(rgb_size=(64, 48), thermal_size=(16, 12), seed=1).capture()
    RawRecorder(tmp_path).record(original)

    loaded = RawLibrary(tmp_path).load(original.frame_set_id)

    assert loaded.metadata == original.metadata
    for channel in Channel:
        np.testing.assert_array_equal(loaded.array(channel), original.array(channel))


def test_missing_project_has_no_ids(tmp_path):
    assert RawLibrary(tmp_path).ids() == []


def test_incomplete_frame_set_is_rejected(tmp_path):
    original = SimulatedDevice(rgb_size=(32, 24), thermal_size=(8, 6)).capture()
    paths = RawRecorder(tmp_path).record(original)
    paths.nir.chmod(0o666)
    paths.nir.unlink()

    with pytest.raises(StorageError, match="incomplete"):
        RawLibrary(tmp_path).load(original.frame_set_id)
