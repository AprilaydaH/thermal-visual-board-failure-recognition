"""Fetches public PCB datasets.

Only the WACV 2019 set is wired up: it is a direct download with no registration. FICS-PCB is
larger but sits behind a Trust-Hub account and a data agreement, so it is listed here for
reference and has to be fetched by hand.
"""

from __future__ import annotations

import logging
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

CHUNK = 1 << 20


@dataclass(frozen=True)
class DatasetSource:
    name: str
    url: str | None
    directory: str
    note: str


WACV_2019 = DatasetSource(
    name="pcb_wacv_2019",
    url="https://ripl.cc.gatech.edu/data/pcb_wacv_2019.zip",
    directory="pcb_wacv_2019",
    note="47 board scans, PASCAL VOC annotations. Cite Kuo et al., WACV 2019.",
)

FICS_PCB = DatasetSource(
    name="fics_pcb",
    url=None,
    directory="fics_pcb",
    note="https://www.trust-hub.org/#/data/fics-pcb, needs an account and a data agreement.",
)


def download_wacv(destination: Path | str, *, force: bool = False) -> Path:
    """Download and unpack the dataset. Returns the directory holding the board folders."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    unpacked = destination / WACV_2019.directory
    if unpacked.is_dir() and not force:
        logger.info("dataset already present at %s", unpacked)
        return unpacked

    archive = destination / f"{WACV_2019.name}.zip"
    if not archive.exists() or force:
        logger.info("downloading %s", WACV_2019.url)
        _download(WACV_2019.url, archive)

    logger.info("unpacking %s", archive)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(destination)

    if not unpacked.is_dir():
        raise FileNotFoundError(f"archive did not contain {WACV_2019.directory}")
    return unpacked


def _download(url: str, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url) as response, temporary.open("wb") as handle:
        total = int(response.headers.get("Content-Length", 0))
        received = 0
        while chunk := response.read(CHUNK):
            handle.write(chunk)
            received += len(chunk)
            if total:
                logger.debug("%.0f%% of %s", 100 * received / total, path.name)
    temporary.replace(path)
