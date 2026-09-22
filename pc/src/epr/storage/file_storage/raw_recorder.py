"""Writes raw frame sets into a project directory.

Raw data is immutable: files are made read-only after writing and an existing frame set is
never overwritten. Processed results are regenerated from these files, not the other way round.
"""

from __future__ import annotations

import stat
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tifffile

from epr.core.domain_models.frame_set import Channel, FrameSet
from epr.core.errors import StorageError

RAW_SUBDIRECTORIES = ("rgb", "nir", "thermal", "metadata")


@dataclass(frozen=True)
class RecordedPaths:
    rgb: Path
    nir: Path
    thermal: Path
    metadata: Path

    def __iter__(self):
        return iter((self.rgb, self.nir, self.thermal, self.metadata))


class RawRecorder:
    """Stores frame sets under ``<project>/raw/{rgb,nir,thermal,metadata}``."""

    def __init__(self, project_dir: Path | str) -> None:
        self.project_dir = Path(project_dir)
        self.raw_dir = self.project_dir / "raw"
        for name in RAW_SUBDIRECTORIES:
            (self.raw_dir / name).mkdir(parents=True, exist_ok=True)

    def record(self, frame_set: FrameSet) -> RecordedPaths:
        name = f"{frame_set.frame_set_id:012d}"
        paths = RecordedPaths(
            rgb=self.raw_dir / "rgb" / f"{name}.png",
            nir=self.raw_dir / "nir" / f"{name}.png",
            thermal=self.raw_dir / "thermal" / f"{name}.tiff",
            metadata=self.raw_dir / "metadata" / f"{name}.json",
        )
        for path in paths:
            if path.exists():
                raise StorageError(f"raw data already exists and is never overwritten: {path}")

        bgr = cv2.cvtColor(frame_set.array(Channel.RGB), cv2.COLOR_RGB2BGR)
        _write_image(paths.rgb, bgr)
        _write_image(paths.nir, frame_set.array(Channel.NIR))
        tifffile.imwrite(paths.thermal, frame_set.array(Channel.THERMAL))
        paths.metadata.write_text(
            frame_set.metadata.model_dump_json(indent=2), encoding="utf-8"
        )

        for path in paths:
            _make_read_only(path)
        return paths


def _write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise StorageError(f"could not write {path}")


def _make_read_only(path: Path) -> None:
    path.chmod(stat.S_IREAD)
