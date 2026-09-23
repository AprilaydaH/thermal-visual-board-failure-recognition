"""Headless G1 path: simulate a capture and probe a click."""

import pytest

from epr.apps.inspection.__main__ import main


def test_inspect_prints_the_thermal_reading(tmp_path, capsys):
    status = main(["--project", str(tmp_path), "--simulate", "--click", "0.45,0.42"])

    output = capsys.readouterr().out
    assert status == 0
    assert "thermal pixel" in output
    assert "T " in output
    assert "C" in output


def test_inspect_refuses_an_empty_project(tmp_path):
    with pytest.raises(SystemExit, match="no raw frame sets"):
        main(["--project", str(tmp_path), "--click", "0.5,0.5"])
