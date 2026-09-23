"""Application data location is independent of the working directory."""

from pathlib import Path

from epr.core.paths import application_home, data_path, find_checkout, user_home


def test_checkout_is_found_from_this_repository():
    root = find_checkout()
    assert root is not None
    assert (root / "pc" / "src" / "epr" / "__init__.py").is_file()


def test_data_path_sits_under_the_application_home():
    expected = application_home() / "data" / "models" / "detector.pt"
    assert data_path("models", "detector.pt") == expected


def test_epr_home_overrides_the_checkout(tmp_path, monkeypatch):
    monkeypatch.setenv("EPR_HOME", str(tmp_path))
    assert application_home() == tmp_path.resolve()
    assert data_path("projects") == tmp_path.resolve() / "data" / "projects"


def test_user_home_is_used_when_there_is_no_checkout(tmp_path, monkeypatch):
    monkeypatch.delenv("EPR_HOME", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("epr.core.paths.find_checkout", lambda start=None: None)
    assert application_home() == user_home()


def test_user_home_is_under_a_platform_data_directory():
    home = user_home()
    assert home.name == "epr"
    assert home.is_absolute()
    assert home != Path("/")
