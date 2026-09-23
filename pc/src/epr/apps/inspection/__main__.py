"""G1 inspection: stored RGB + thermal, click maps to a temperature.

    epr inspect --simulate --click 0.45,0.40
    epr inspect --project data/projects/demo

The first form captures one simulated frame set and prints the thermal reading at a
normalized RGB point. The second opens the viewer on an existing project. That is the
milestone: a visible point on the board and the corresponding thermal region.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from epr.core.paths import data_path
from epr.device.simulator import SimulatedDevice
from epr.processing.registration import probe
from epr.storage.file_storage import RawRecorder
from epr.storage.file_storage.raw_loader import RawLibrary

logger = logging.getLogger("epr.inspect")


def _click(text: str) -> tuple[float, float]:
    try:
        x, y = (float(part) for part in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("click must look like 0.45,0.40") from error
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise argparse.ArgumentTypeError("click coordinates must be in [0, 1]")
    return x, y


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--project",
        type=Path,
        default=None,
        help="project directory (default: <EPR_HOME>/data/projects/g1)",
    )
    parser.add_argument("--frame", type=int, default=None, help="frame_set_id to open")
    parser.add_argument(
        "--click",
        type=_click,
        default=None,
        help="normalized RGB point, e.g. 0.45,0.40; prints the thermal reading and exits",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="capture one simulated frame set into the project if it is empty",
    )
    parser.add_argument("--gui", action="store_true", help="open the click-to-thermal window")
    return parser


def _ensure_frames(project: Path, simulate: bool) -> RawLibrary:
    library = RawLibrary(project)
    if library.ids():
        return library
    if not simulate:
        raise SystemExit(
            f"no raw frame sets in {project}. Run `epr acquire` or pass --simulate."
        )
    recorder = RawRecorder(project)
    device = SimulatedDevice(
        rgb_size=(640, 480), thermal_size=(160, 120), frame_interval_s=8.0
    )
    stored = None
    for frame_set in device.capture_many(5):
        recorder.record(frame_set)
        stored = frame_set
    logger.info("simulated frame set %d stored in %s", stored.frame_set_id, project)
    return RawLibrary(project)


def _load_sets(library: RawLibrary, frame_id: int | None):
    if frame_id is None:
        return [library.load(item) for item in library.ids()]
    return [library.load(frame_id)]


def format_probe(frame_set, mark) -> str:
    meta = frame_set.metadata
    return (
        f"frame set {frame_set.frame_set_id}  click {mark.rgb_xy[0]:.3f},{mark.rgb_xy[1]:.3f}\n"
        f"  RGB pixel      {mark.rgb_pixel} of {frame_set.rgb.shape[1]}x{frame_set.rgb.shape[0]}\n"
        f"  thermal pixel  {mark.thermal_pixel} of "
        f"{frame_set.thermal.shape[1]}x{frame_set.thermal.shape[0]}\n"
        f"  T              {mark.temperature_c:.2f} C  "
        f"(region mean {mark.region_mean_c:.2f}  max {mark.region_max_c:.2f})\n"
        f"  ambient        {meta.environment.ambient_temperature_c:.2f} C  "
        f"distance {meta.geometry.distance_mm:.1f} mm"
    )


def run_gui(frame_sets) -> int:
    from PySide6.QtWidgets import QApplication

    from epr.apps.inspection.window import InspectionWindow

    app = QApplication.instance() or QApplication(sys.argv)
    window = InspectionWindow(frame_sets)
    window.show()
    return int(app.exec())


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    project = args.project or data_path("projects", "g1")
    library = _ensure_frames(project, args.simulate)
    frame_sets = _load_sets(library, args.frame)

    target = frame_sets[0] if args.frame is not None else frame_sets[-1]

    if args.click is not None:
        mark = probe(target, *args.click)
        print(format_probe(target, mark))
        if not args.gui:
            return 0

    if args.click is None and not args.gui:
        # Headless default: the hot package, so G1 is demonstrable without a display.
        mark = probe(target, 0.45, 0.42)
        print(format_probe(target, mark))
        print("pass --gui to click on the RGB image")
        return 0

    return run_gui(frame_sets)


if __name__ == "__main__":
    sys.exit(main())
