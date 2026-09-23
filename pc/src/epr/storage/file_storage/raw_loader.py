"""Reads immutable raw frame sets back from a project directory."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import tifffile

from epr.core.domain_models.frame_set import FrameSet, FrameSetMetadata
from epr.core.errors import StorageError
from epr.storage.file_storage.raw_recorder import RAW_SUBDIRECTORIES, RecordedPaths


class RawLibrary:
    """The read side of ``RawRecorder``. Files stay read-only."""

    def __init__(self, project_dir: Path | str) -> None:
        self.project_dir = Path(project_dir)
        self.raw_dir = self.project_dir / "raw"

    def ids(self) -> list[int]:
        metadata = self.raw_dir / "metadata"
        if not metadata.is_dir():
            return []
        found = []
        for path in metadata.glob("*.json"):
            try:
                found.append(int(path.stem))
            except ValueError:
                continue
        return sorted(found)

    def paths_for(self, frame_set_id: int) -> RecordedPaths:
        name = f"{frame_set_id:012d}"
        return RecordedPaths(
            rgb=self.raw_dir / "rgb" / f"{name}.png",
            nir=self.raw_dir / "nir" / f"{name}.png",
            thermal=self.raw_dir / "thermal" / f"{name}.tiff",
            metadata=self.raw_dir / "metadata" / f"{name}.json",
        )

    def load(self, frame_set_id: int) -> FrameSet:
        paths = self.paths_for(frame_set_id)
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise StorageError(f"incomplete raw frame set {frame_set_id}: {', '.join(missing)}")

        metadata = FrameSetMetadata.model_validate_json(
            paths.metadata.read_text(encoding="utf-8")
        )
        rgb_bgr = cv2.imread(str(paths.rgb), cv2.IMREAD_COLOR)
        nir = cv2.imread(str(paths.nir), cv2.IMREAD_GRAYSCALE)
        thermal = tifffile.imread(paths.thermal)
        if rgb_bgr is None or nir is None:
            raise StorageError(f"could not decode images for frame set {frame_set_id}")

        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        if thermal.dtype != np.dtype("<u2"):
            thermal = np.asarray(thermal, dtype="<u2")
        return FrameSet(metadata=metadata, rgb=rgb, nir=nir, thermal=thermal)

    def load_latest(self) -> FrameSet:
        ids = self.ids()
        if not ids:
            raise StorageError(f"no raw frame sets in {self.project_dir}")
        return self.load(ids[-1])


def recorded_layout_exists(project_dir: Path | str) -> bool:
    raw = Path(project_dir) / "raw"
    return all((raw / name).is_dir() for name in RAW_SUBDIRECTORIES)
