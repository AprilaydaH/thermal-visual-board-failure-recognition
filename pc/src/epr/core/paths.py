"""Where the application stores data on this machine.

Commands default to a checkout's ``data/`` folder when you are working from the repository.
On a machine that only has the installed package, they use the platform user-data directory
unless ``EPR_HOME`` is set.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ENV_HOME = "EPR_HOME"


def find_checkout(start: Path | None = None) -> Path | None:
    """Walk up from ``start`` (or the working directory) looking for this repository."""
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        package = candidate / "pc" / "src" / "epr" / "__init__.py"
        if package.is_file() and (candidate / "data").is_dir():
            return candidate
    return None


def user_home() -> Path:
    """Platform user-data directory for an installed copy."""
    if sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return root / "epr"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "epr"
    xdg = os.environ.get("XDG_DATA_HOME")
    return Path(xdg) / "epr" if xdg else Path.home() / ".local" / "share" / "epr"


def application_home() -> Path:
    """Root used for ``data/``, models and projects."""
    if explicit := os.environ.get(ENV_HOME):
        return Path(explicit).expanduser().resolve()
    return find_checkout() or user_home()


def data_path(*parts: str | Path) -> Path:
    """A path under the application ``data/`` directory."""
    path = application_home() / "data"
    for part in parts:
        path /= part
    return path
