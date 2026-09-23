"""Recognition demo.

Runs a simulated heating sequence through the CNN and the LNN and prints what the interface
would show per component. The models are untrained, so the classes are meaningless noise; what
this demonstrates is that the data path, the feature layout and the uncertainty reporting work
end to end. The thermal columns are real measurements and are meaningful now.

    epr demo --frames 6 --device auto
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from epr.device.simulator import SimulatedDevice
from epr.recognition.classification import CnnConfig, ComponentCnn
from epr.recognition.fusion import FeatureLayout, FusionConfig, LnnFusionModel
from epr.recognition.ocr import MarkingReading
from epr.recognition.pipeline import LabelSpace, RecognitionPipeline
from epr.recognition.runtime import describe_runtime, seed_everything, select_device

CLASSES = ("resistor", "capacitor", "inductor", "diode", "transistor", "integrated_circuit",
           "connector", "crystal")
PACKAGES = ("0805", "SOT23", "SOIC8", "QFP64", "BGA", "HEADER")
EMBEDDING = 64

HEADER = (
    f"{'region':<8}{'class':<20}{'package':<10}"
    f"{'condition':<14}{'conf':>6}{'T_max':>8}{'dT/dt':>8}"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=int, default=6, help="frame sets in the sequence")
    parser.add_argument("--frame-interval", type=float, default=5.0, help="simulated seconds")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    seed_everything(args.seed)
    device = select_device(args.device)

    runtime = describe_runtime()
    print(f"torch {runtime['torch']}, cuda {runtime['cuda_version']}, running on {device}")

    head = SimulatedDevice(seed=args.seed, frame_interval_s=args.frame_interval)
    frame_sets = list(head.capture_many(args.frames))
    regions = head.component_regions()

    layout = FeatureLayout(visual_dim=EMBEDDING, nir_dim=EMBEDDING)
    pipeline = RecognitionPipeline(
        ComponentCnn(
            CnnConfig(embedding_dim=EMBEDDING, n_classes=len(CLASSES), n_packages=len(PACKAGES))
        ),
        LnnFusionModel(
            FusionConfig(layout=layout, n_classes=len(CLASSES), n_packages=len(PACKAGES))
        ),
        LabelSpace(CLASSES, PACKAGES),
        device=device,
    )

    markings = {region.region_id: MarkingReading(region.region_id, 0.6) for region in regions}
    predictions = pipeline.run(frame_sets, regions, markings)

    span_s = (frame_sets[-1].metadata.timestamp_ns - frame_sets[0].metadata.timestamp_ns) / 1e9
    print(f"{len(regions)} components over {args.frames} frame sets spanning {span_s:.0f} s")
    print(f"feature vector: {layout.total} values per component and time step")
    print(HEADER)
    for prediction in predictions:
        print(
            f"{prediction.region_id:<8}"
            f"{prediction.component_class:<20}"
            f"{prediction.package:<10}"
            f"{prediction.thermal_condition:<14}"
            f"{prediction.confidence:>6.2f}"
            f"{prediction.thermal.t_max_c:>7.1f}C"
            f"{prediction.thermal.heating_rate_c_s:>7.2f}"
        )

    unknown = sum(prediction.is_unknown for prediction in predictions)
    print(f"{unknown} of {len(predictions)} reported unknown by the rejection rules")
    return 0


if __name__ == "__main__":
    sys.exit(main())
