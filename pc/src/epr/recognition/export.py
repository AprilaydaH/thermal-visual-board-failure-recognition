"""ONNX export.

Production inference targets ONNX Runtime, so every model that ships must survive the round
trip. Export uses the dynamo exporter; the older TorchScript path is deprecated in Torch 2.7.

Two forms of the fusion model are exported because they serve different jobs. The sequence
form unrolls its time loop into the graph, which fixes the number of time steps and suits
reprocessing a stored inspection. The step form takes and returns the hidden state, which is
what a live inspection needs as frame sets arrive one at a time.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from epr.recognition.classification.cnn import ComponentCnn
from epr.recognition.fusion.model import FusionStepModule, LnnFusionModel

# The batch dimension must be bounded: an unbounded one lets the exporter consider int64 max,
# which conflicts with the shapes derived inside the LNN loop.
MAX_BATCH = 4096

# The example input must not have a batch of one: the exporter specializes size-one dimensions
# and the exported graph then silently accepts only single-item batches.
EXAMPLE_BATCH = 2

CNN_OUTPUT_NAMES = ["rgb_embedding", "nir_embedding", "class_logits", "package_logits"]
FUSION_OUTPUT_NAMES = ["class_logits", "package_logits", "thermal_logits", "unknown_logit"]


def _batch_dim() -> torch.export.Dim:
    return torch.export.Dim("batch", min=1, max=MAX_BATCH)


def _prepare(path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def export_component_cnn(model: ComponentCnn, path: Path | str) -> Path:
    path = _prepare(path)
    size = model.config.crop_size
    batch = _batch_dim()

    torch.onnx.export(
        model.eval().cpu(),
        (
            torch.zeros(EXAMPLE_BATCH, model.config.rgb_channels, size, size),
            torch.zeros(EXAMPLE_BATCH, model.config.nir_channels, size, size),
        ),
        str(path),
        input_names=["rgb", "nir"],
        output_names=CNN_OUTPUT_NAMES,
        dynamic_shapes={"rgb": {0: batch}, "nir": {0: batch}},
        dynamo=True,
        verbose=False,
    )
    return path


def export_fusion_sequence(model: LnnFusionModel, path: Path | str, sequence_length: int) -> Path:
    """Export for a fixed number of time steps."""
    path = _prepare(path)
    batch = _batch_dim()

    torch.onnx.export(
        model.eval().cpu(),
        (
            torch.zeros(EXAMPLE_BATCH, sequence_length, model.feature_dim),
            torch.ones(EXAMPLE_BATCH, sequence_length),
        ),
        str(path),
        input_names=["features", "dt"],
        output_names=FUSION_OUTPUT_NAMES,
        dynamic_shapes={"x": {0: batch}, "dt": {0: batch}},
        dynamo=True,
        verbose=False,
    )
    return path


def export_fusion_step(model: LnnFusionModel, path: Path | str) -> Path:
    """Export the streaming form: one frame set in, hidden state in and out."""
    path = _prepare(path)
    step_module = FusionStepModule(model).eval().cpu()
    batch = _batch_dim()

    torch.onnx.export(
        step_module,
        (
            torch.zeros(EXAMPLE_BATCH, model.feature_dim),
            torch.ones(EXAMPLE_BATCH),
            step_module.initial_state(EXAMPLE_BATCH),
        ),
        str(path),
        input_names=["features", "dt", "state"],
        output_names=[*FUSION_OUTPUT_NAMES, "next_state"],
        dynamic_shapes={"x": {0: batch}, "dt": {0: batch}, "state": {1: batch}},
        dynamo=True,
        verbose=False,
    )
    return path


def run_onnx(path: Path | str, inputs: Mapping[str, np.ndarray]) -> list[np.ndarray]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run(None, {name: np.asarray(value) for name, value in inputs.items()})


def max_absolute_difference(reference: Tensor, candidate: np.ndarray) -> float:
    return float(np.abs(reference.detach().cpu().numpy() - candidate).max())
