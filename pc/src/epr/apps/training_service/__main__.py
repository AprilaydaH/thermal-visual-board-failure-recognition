"""Training service.

Trains candidate models in a separate process, as the plan requires.

    epr train cnn --download --epochs 60
    epr train detector --epochs 40
    epr train eval

`cnn` pretrains the RGB stream of the component CNN on public board photographs. `detector`
trains the class-agnostic component detector on the same boxes.

Both produce candidates, never production models. Promotion needs the validation and approval
steps of learning level 3, which do not exist yet.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from epr.core.paths import data_path
from epr.learning.dataset_builder.crop_cache import CropCache, build_crop_cache
from epr.learning.dataset_builder.download import download_wacv
from epr.learning.dataset_builder.wacv import (
    boards,
    build_label_space,
    label_histogram,
    load_wacv,
    split_by_board,
)
from epr.learning.training.cnn import (
    CropDataset,
    PretrainConfig,
    RgbPretrainModel,
    save_checkpoint,
    train_rgb_stream,
)
from epr.learning.training.detector import (
    BoardSample,
    DetectorTrainingConfig,
    evaluate_detector,
    load_boards,
    load_detector,
    save_detector,
    train_detector,
)
from epr.recognition.classification.cnn import CnnConfig
from epr.recognition.detection.detector import ComponentDetector, DetectorConfig, detect
from epr.recognition.detection.heatmap import average_precision
from epr.recognition.runtime import seed_everything, select_device

logger = logging.getLogger("epr.training")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    external = data_path("external")
    models = data_path("models")

    for name in ("cnn", "detector"):
        command = commands.add_parser(name)
        command.add_argument("--data-dir", type=Path, default=external)
        command.add_argument("--download", action="store_true", help="fetch the dataset first")
        command.add_argument("--device", default="auto")
        command.add_argument("--seed", type=int, default=0)
        command.add_argument("--validation-fraction", type=float, default=0.25)
        command.add_argument("--epochs", type=int, default=60 if name == "cnn" else 40)

    cnn = commands.choices["cnn"]
    cnn.add_argument("--cache-dir", type=Path, default=None)
    cnn.add_argument("--checkpoint", type=Path, default=models / "rgb_stream.pt")
    cnn.add_argument("--crop-size", type=int, default=64)
    cnn.add_argument("--embedding-dim", type=int, default=128)
    cnn.add_argument("--batch-size", type=int, default=128)
    cnn.add_argument("--learning-rate", type=float, default=2e-3)
    cnn.add_argument("--min-instances", type=int, default=50)
    cnn.add_argument("--rebuild-cache", action="store_true")

    detector = commands.choices["detector"]
    detector.add_argument("--checkpoint", type=Path, default=models / "detector.pt")
    detector.add_argument("--tile-size", type=int, default=256)
    detector.add_argument("--batch-size", type=int, default=16)
    detector.add_argument("--learning-rate", type=float, default=1.5e-3)
    detector.add_argument("--tiles-per-epoch", type=int, default=1024)

    evaluate = commands.add_parser("eval", help="run a saved detector on the WACV boards")
    evaluate.add_argument("--data-dir", type=Path, default=external)
    evaluate.add_argument("--download", action="store_true", help="fetch the dataset first")
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--validation-fraction", type=float, default=0.25)
    evaluate.add_argument("--checkpoint", type=Path, default=models / "detector.pt")
    evaluate.add_argument("--out", type=Path, default=models / "detector_eval")
    return parser


def _dataset(args: argparse.Namespace):
    directory = download_wacv(args.data_dir) if args.download else args.data_dir / "pcb_wacv_2019"
    instances = load_wacv(directory)
    train, validation = split_by_board(
        instances, validation_fraction=args.validation_fraction, seed=args.seed
    )
    print(f"{len(instances)} components, {len(boards(instances))} physical boards")
    return instances, train, validation


def run_cnn(args: argparse.Namespace) -> int:
    device = select_device(args.device)
    instances, train_instances, validation_instances = _dataset(args)

    label_space = build_label_space(instances, min_instances=args.min_instances)
    histogram = label_histogram(instances)
    print(f"label space ({len(label_space)}): {', '.join(label_space)}")
    print("  " + ", ".join(f"{label} {histogram[label]}" for label in label_space))

    cache_root = args.cache_dir or args.data_dir / "cache" / f"crops{args.crop_size}"
    caches = {}
    for split, split_instances in (
        ("train", train_instances),
        ("validation", validation_instances),
    ):
        directory = cache_root / split
        if args.rebuild_cache or not (directory / "manifest.json").exists():
            caches[split] = build_crop_cache(
                split_instances, directory, label_space=label_space, crop_size=args.crop_size
            )
        else:
            caches[split] = CropCache.load(directory)

    train_cache, validation_cache = caches["train"], caches["validation"]
    print(
        f"train {len(train_cache)} crops from {len(train_cache.manifest.board_ids)} boards, "
        f"validation {len(validation_cache)} from {len(validation_cache.manifest.board_ids)}"
    )

    cnn_config = CnnConfig(
        crop_size=args.crop_size,
        embedding_dim=args.embedding_dim,
        n_classes=len(label_space),
    )
    model = RgbPretrainModel(cnn_config, len(label_space))
    result = train_rgb_stream(
        model,
        CropDataset(
            train_cache.crops,
            train_cache.labels,
            augment=True,
            generator=np.random.default_rng(args.seed),
        ),
        CropDataset(validation_cache.crops, validation_cache.labels),
        PretrainConfig(
            epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate
        ),
        device=device,
    )

    path = save_checkpoint(
        args.checkpoint,
        model,
        label_space=list(label_space),
        dataset_fingerprint=train_cache.manifest.fingerprint,
        result=result,
    )
    print(
        f"best validation accuracy {result.best_accuracy:.3f}, "
        f"balanced {result.balanced_accuracy:.3f}"
    )
    print(f"candidate RGB stream saved to {path}")
    return 0


def run_detector(args: argparse.Namespace) -> int:
    device = select_device(args.device)
    _, train_instances, validation_instances = _dataset(args)

    train_boards = load_boards(train_instances)
    validation_boards = load_boards(validation_instances)
    print(
        f"train {len(train_boards)} scans with "
        f"{sum(len(b.boxes) for b in train_boards)} boxes, "
        f"validation {len(validation_boards)} scans with "
        f"{sum(len(b.boxes) for b in validation_boards)} boxes"
    )

    model = ComponentDetector(DetectorConfig(tile_size=args.tile_size))
    result = train_detector(
        model,
        train_boards,
        validation_boards,
        DetectorTrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            tiles_per_epoch=args.tiles_per_epoch,
        ),
        device=device,
        seed=args.seed,
    )

    best = result.best
    path = save_detector(
        args.checkpoint,
        model,
        dataset_fingerprint=f"wacv:{len(train_instances)}",
        result=result,
    )
    print(
        f"best AP50 {best.average_precision:.3f}, recall {best.recall:.3f}, "
        f"precision {best.precision:.3f}, F1 {best.f1:.3f}"
    )
    print(f"candidate detector saved to {path}")
    return 0


def _draw_boxes(
    image: np.ndarray, boxes: np.ndarray, colour: tuple[int, int, int], thickness: int = 1
) -> None:
    for box in boxes:
        x0, y0, x1, y1 = (int(round(value)) for value in box)
        cv2.rectangle(image, (x0, y0), (x1, y1), colour, thickness)


def _evaluate_split(
    model, boards: Sequence[BoardSample], *, device, overlay_dir: Path, split: str
) -> None:
    overlay_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{split}: {len(boards)} scans")
    print(f"  {'scan':<28} {'gt':>4} {'det':>4} {'AP50':>6} {'rec':>6} {'prec':>6}")

    for board in boards:
        detection = detect(model, board.image, device=device)
        ap, recall, precision = average_precision(
            detection.boxes, detection.scores, board.boxes
        )
        name = board.scan_id or board.board_id
        print(
            f"  {name:<28} {len(board.boxes):>4} {len(detection.boxes):>4} "
            f"{ap:>6.3f} {recall:>6.3f} {precision:>6.3f}"
        )

        canvas = board.image.copy()
        _draw_boxes(canvas, board.boxes, (40, 180, 40), 1)
        _draw_boxes(canvas, detection.boxes, (0, 0, 220), 1)
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
        cv2.imwrite(str(overlay_dir / f"{split}_{safe}.jpg"), canvas)

    metrics = evaluate_detector(model, boards, device=device)
    print(
        f"  mean AP50 {metrics.average_precision:.3f}, recall {metrics.recall:.3f}, "
        f"precision {metrics.precision:.3f}, F1 {metrics.f1:.3f}"
    )


def run_eval(args: argparse.Namespace) -> int:
    if not args.checkpoint.exists():
        print(f"No detector checkpoint at {args.checkpoint}. Train with `detector` first.")
        return 1

    device = select_device(args.device)
    model = load_detector(args.checkpoint, device=device)
    _, train_instances, validation_instances = _dataset(args)

    print(f"loaded {args.checkpoint} on {device}")
    _evaluate_split(
        model,
        load_boards(validation_instances),
        device=device,
        overlay_dir=args.out,
        split="validation",
    )
    _evaluate_split(
        model,
        load_boards(train_instances),
        device=device,
        overlay_dir=args.out,
        split="train",
    )
    print(f"\noverlays written to {args.out}")
    print("green = annotated box, red = detector")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    seed_everything(args.seed)

    runners = {"cnn": run_cnn, "detector": run_detector, "eval": run_eval}
    status = runners[args.command](args)
    if args.command != "eval":
        print("This is a candidate. Promotion needs validation and approval, which are not built.")
    return status


if __name__ == "__main__":
    sys.exit(main())
