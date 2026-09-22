"""Training service.

Trains candidate models in a separate process, as the plan requires.

    python -m epr.apps.training_service cnn --download --epochs 60
    python -m epr.apps.training_service detector --epochs 40

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

import numpy as np

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
    DetectorTrainingConfig,
    load_boards,
    save_detector,
    train_detector,
)
from epr.recognition.classification.cnn import CnnConfig
from epr.recognition.detection.detector import ComponentDetector, DetectorConfig
from epr.recognition.runtime import seed_everything, select_device

logger = logging.getLogger("epr.training")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    for name in ("cnn", "detector"):
        command = commands.add_parser(name)
        command.add_argument("--data-dir", type=Path, default=Path("data/external"))
        command.add_argument("--download", action="store_true", help="fetch the dataset first")
        command.add_argument("--device", default="auto")
        command.add_argument("--seed", type=int, default=0)
        command.add_argument("--validation-fraction", type=float, default=0.25)
        command.add_argument("--epochs", type=int, default=60 if name == "cnn" else 40)

    cnn = commands.choices["cnn"]
    cnn.add_argument("--cache-dir", type=Path, default=None)
    cnn.add_argument("--checkpoint", type=Path, default=Path("data/models/rgb_stream.pt"))
    cnn.add_argument("--crop-size", type=int, default=64)
    cnn.add_argument("--embedding-dim", type=int, default=128)
    cnn.add_argument("--batch-size", type=int, default=128)
    cnn.add_argument("--learning-rate", type=float, default=2e-3)
    cnn.add_argument("--min-instances", type=int, default=50)
    cnn.add_argument("--rebuild-cache", action="store_true")

    detector = commands.choices["detector"]
    detector.add_argument("--checkpoint", type=Path, default=Path("data/models/detector.pt"))
    detector.add_argument("--tile-size", type=int, default=256)
    detector.add_argument("--batch-size", type=int, default=16)
    detector.add_argument("--learning-rate", type=float, default=1.5e-3)
    detector.add_argument("--tiles-per-epoch", type=int, default=1024)
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


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    seed_everything(args.seed)

    status = run_cnn(args) if args.command == "cnn" else run_detector(args)
    print("This is a candidate. Promotion needs validation and approval, which are not built.")
    return status


if __name__ == "__main__":
    sys.exit(main())
