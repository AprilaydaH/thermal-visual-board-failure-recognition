"""Headless G1 path: simulate a capture and probe a click."""

import pytest

from epr.apps.inspection.__main__ import main


def test_inspect_prints_the_thermal_reading(tmp_path, capsys):
    status = main(["--project", str(tmp_path), "--simulate", "--click", "0.45,0.42"])

    output = capsys.readouterr().out
    assert status == 0
    assert "U1" in output
    assert "QFP64" in output
    assert "verdict" in output
    assert "thermal pixel" in output
    assert "ΔT" in output


def test_inspect_refuses_an_empty_project(tmp_path):
    with pytest.raises(SystemExit, match="no raw frame sets"):
        main(["--project", str(tmp_path), "--click", "0.5,0.5"])
