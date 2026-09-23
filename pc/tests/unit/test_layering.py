"""The acquisition stack must install and run without the recognition requirements."""

import subprocess
import sys

ACQUISITION_MODULES = [
    "epr.core.domain_models",
    "epr.core.domain_models.component",
    "epr.core.paths",
    "epr.device.acquisition",
    "epr.device.simulator",
    "epr.storage.file_storage",
    "epr.processing.thermal",
    "epr.processing.registration",
    "epr.processing.inspection",
    "epr.apps.cli",
    "epr.apps.acquisition_service.__main__",
    "epr.apps.inspection.__main__",
]


def test_acquisition_path_does_not_pull_in_torch():
    imports = "; ".join(f"import {module}" for module in ACQUISITION_MODULES)
    result = subprocess.run(
        [sys.executable, "-c", f"{imports}; import sys; print('torch' in sys.modules)"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False"
