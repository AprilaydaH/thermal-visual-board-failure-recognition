"""Package dimension library, built from installed KiCad footprints.

    epr packages build
    epr packages match --size 2.0x1.25
    epr packages scale --distance 150 --pixels 48

`build` reads the footprint libraries and writes a table of physical package sizes. `match`
answers what a measured component could be. `scale` shows what a box in pixels means in
millimetres at a given working distance.

Nothing here is trained or learned: it is published mechanical data.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from epr.core.paths import data_path
from epr.processing.scale import CalibratedScale, ThinLensOptics
from epr.recognition.packages.footprints import default_kicad_root
from epr.recognition.packages.library import PackageLibrary, SizeQuery

logger = logging.getLogger("epr.packages")


def _size(text: str) -> tuple[float, float]:
    try:
        length, width = (float(part) for part in text.lower().replace("*", "x").split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("size must look like 2.0x1.25") from error
    return max(length, width), min(length, width)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="scan KiCad footprints into a size table")
    table = data_path("external", "package_library.json")
    build.add_argument("--kicad", type=Path, default=None, help="footprint directory")
    build.add_argument("--out", type=Path, default=table)
    build.add_argument("--body-only", action="store_true", help="keep only F.Fab outlines")

    match = commands.add_parser("match", help="look up a measured size")
    match.add_argument("--table", type=Path, default=table)
    match.add_argument("--size", type=_size, required=True, help="millimetres, e.g. 2.0x1.25")
    match.add_argument("--uncertainty", type=float, default=0.0)
    match.add_argument("--family", action="append", default=None)
    match.add_argument("--limit", type=int, default=8)
    match.add_argument("--body-only", action="store_true")

    scale = commands.add_parser("scale", help="convert pixels to millimetres")
    scale.add_argument("--distance", type=float, required=True, help="working distance in mm")
    scale.add_argument("--pixels", type=float, required=True, help="box size in pixels")
    scale.add_argument("--focal-length", type=float, default=25.0)
    scale.add_argument("--sensor-width", type=float, default=7.4)
    scale.add_argument("--image-width", type=int, default=2592)
    scale.add_argument("--calibrated-mm-per-px", type=float, default=None)
    scale.add_argument("--calibrated-at", type=float, default=None)
    return parser


def run_build(args: argparse.Namespace) -> int:
    root = args.kicad or default_kicad_root()
    if root is None:
        print("No KiCad footprint directory found. Pass --kicad explicitly.")
        return 1

    print(f"scanning {root}")
    library = PackageLibrary.from_kicad(root, body_only=args.body_only)
    path = library.save(args.out)

    families = library.families()
    sources = {}
    for footprint in library.footprints:
        sources[footprint.source] = sources.get(footprint.source, 0) + 1

    print(f"{len(library)} footprints written to {path}")
    print("  outline source: " + ", ".join(f"{k} {v}" for k, v in sorted(sources.items())))
    print(f"  families ({len(families)}):")
    for family, count in families.most_common():
        print(f"    {family:<14}{count:>6}")
    return 0


def run_match(args: argparse.Namespace) -> int:
    if not args.table.exists():
        print(f"No table at {args.table}. Run `build` first.")
        return 1

    library = PackageLibrary.load(args.table)
    length, width = args.size
    query = SizeQuery(length_mm=length, width_mm=width, uncertainty_mm=args.uncertainty)

    matches = library.match(query, families=args.family, limit=args.limit, body_only=args.body_only)
    print(
        f"measured {length:.2f} x {width:.2f} mm (+/- {args.uncertainty:.2f}) "
        f"against {len(library)} footprints"
    )
    if not matches:
        print("  no package of that size")
        return 0

    print(f"  {'score':>6}  {'family':<12} {'package':<16} {'size mm':<14} source  example")
    for match in matches:
        size = f"{match.length_mm:.2f}x{match.width_mm:.2f}"
        print(
            f"  {match.score:>6.3f}  {match.family:<12} {match.package:<16} {size:<14} "
            f"{match.source:<7} {match.footprint}"
        )
    return 0


def run_scale(args: argparse.Namespace) -> int:
    if args.calibrated_mm_per_px and args.calibrated_at:
        scale = CalibratedScale(args.calibrated_mm_per_px, args.calibrated_at)
        model = "calibrated"
    else:
        scale = ThinLensOptics(args.focal_length, args.sensor_width, args.image_width)
        model = "thin lens"

    mm_per_pixel = scale.mm_per_pixel(args.distance)
    print(f"{model} model at {args.distance:.1f} mm")
    print(f"  {mm_per_pixel * 1000:.2f} micrometres per pixel")
    print(f"  {args.pixels:.0f} px = {args.pixels * mm_per_pixel:.3f} mm")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    return {"build": run_build, "match": run_match, "scale": run_scale}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
