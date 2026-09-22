"""Acquisition service.

Runs the whole data path against the simulated gadget: encoded packets in, validated frame
sets out, raw data on disk. It runs as its own process so that a user-interface crash cannot
corrupt a recording.

    python -m epr.apps.acquisition_service --project data/projects/demo --frames 5
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from epr.device.acquisition import FrameSetAssembler, assemble_stream
from epr.device.simulator import SimulatedDevice
from epr.device.transport import ChunkedByteStream
from epr.storage.file_storage import RawRecorder

logger = logging.getLogger("epr.acquisition")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", type=Path, required=True, help="project directory")
    parser.add_argument("--frames", type=int, default=5, help="number of frame sets")
    parser.add_argument("--seed", type=int, default=0, help="simulator seed")
    parser.add_argument("--rgb-width", type=int, default=1280)
    parser.add_argument("--rgb-height", type=int, default=960)
    parser.add_argument(
        "--frame-interval",
        type=float,
        default=1.0,
        help="simulated seconds between frame sets, drives the heating curve",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    device = SimulatedDevice(
        seed=args.seed,
        rgb_size=(args.rgb_width, args.rgb_height),
        frame_interval_s=args.frame_interval,
    )
    recorder = RawRecorder(args.project)
    assembler = FrameSetAssembler()
    stream = ChunkedByteStream(device.stream(args.frames))

    for frame_set in assemble_stream(stream, assembler):
        paths = recorder.record(frame_set)
        hotspot_c = float(frame_set.thermal.max()) / 100.0 - 273.15
        logger.info(
            "frame set %d stored, hotspot %.2f C, %s",
            frame_set.frame_set_id,
            hotspot_c,
            paths.rgb.parent.parent,
        )

    stats = assembler.stats
    logger.info(
        "completed %d, rejected %d, dropped packets %d",
        stats.completed_frame_sets,
        stats.rejected_frame_sets,
        stats.dropped_packets,
    )
    return 0 if stats.completed_frame_sets == args.frames else 1


if __name__ == "__main__":
    sys.exit(main())
