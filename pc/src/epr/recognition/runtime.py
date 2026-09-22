"""Runtime helpers for the recognition stack."""

from __future__ import annotations

import random

import numpy as np
import torch


def select_device(preference: str = "auto") -> torch.device:
    """Resolve a device. ``auto`` picks CUDA when it is actually usable."""
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(preference)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and Torch.

    This does not make CUDA training bit-exact; reproducing a training run also needs the
    deterministic algorithm flags, which cost throughput and belong in the training service.
    """
    random.seed(seed)
    # The legacy global NumPy seed is set deliberately: dataset and augmentation code in the
    # wider ecosystem still draws from it, and a local Generator would not cover that.
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def describe_runtime() -> dict[str, str]:
    """Environment facts worth recording with every training run and model version."""
    info = {
        "torch": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda or "none",
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info
