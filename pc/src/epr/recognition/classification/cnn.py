"""Component CNN.

Two streams, because the plan feeds visual features and NIR features to the LNN as separate
blocks: RGB carries markings and colour, 850 nm NIR carries surface and material information.
Keeping the embeddings separate means a failing illumination channel degrades one block
instead of silently poisoning a single fused vector.

The network is trained from scratch. No pretrained download is required, so the recognition
stack works on an offline bench machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor, nn

IMAGENET_FREE_MEAN = 0.5
IMAGENET_FREE_STD = 0.25


@dataclass(frozen=True)
class CnnConfig:
    crop_size: int = 64
    widths: tuple[int, ...] = (32, 64, 128)
    embedding_dim: int = 128
    n_classes: int = 8
    n_packages: int = 6
    dropout: float = 0.1
    rgb_channels: int = 3
    nir_channels: int = 1

    def __post_init__(self) -> None:
        if self.crop_size < 2 ** len(self.widths):
            raise ValueError("crop_size is too small for the number of downsampling stages")


class ComponentCnnOutput(NamedTuple):
    rgb_embedding: Tensor
    nir_embedding: Tensor
    class_logits: Tensor
    package_logits: Tensor


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class StreamEncoder(nn.Module):
    """One modality: convolutional stages, global pooling, embedding.

    Exposed because the RGB stream is pretrained on public PCB photographs on its own, before
    any NIR or thermal data exists.
    """

    def __init__(self, in_channels: int, widths: tuple[int, ...], embedding_dim: int) -> None:
        super().__init__()
        stages: list[nn.Module] = []
        channels = in_channels
        for width in widths:
            stages.append(_ConvBlock(channels, width))
            channels = width
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Linear(channels, embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        features = self.pool(self.stages(x)).flatten(1)
        return self.project(features)


class ComponentCnn(nn.Module):
    def __init__(self, config: CnnConfig | None = None) -> None:
        super().__init__()
        self.config = config or CnnConfig()
        self.rgb_stream = StreamEncoder(
            self.config.rgb_channels, self.config.widths, self.config.embedding_dim
        )
        self.nir_stream = StreamEncoder(
            self.config.nir_channels, self.config.widths, self.config.embedding_dim
        )
        fused_dim = self.config.embedding_dim * 2
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(self.config.dropout),
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU(inplace=True),
        )
        self.class_head = nn.Linear(fused_dim, self.config.n_classes)
        self.package_head = nn.Linear(fused_dim, self.config.n_packages)

    @property
    def embedding_dim(self) -> int:
        return self.config.embedding_dim

    def forward(self, rgb: Tensor, nir: Tensor) -> ComponentCnnOutput:
        rgb_embedding = self.rgb_stream(rgb)
        nir_embedding = self.nir_stream(nir)
        fused = self.head(torch.cat([rgb_embedding, nir_embedding], dim=-1))
        return ComponentCnnOutput(
            rgb_embedding=rgb_embedding,
            nir_embedding=nir_embedding,
            class_logits=self.class_head(fused),
            package_logits=self.package_head(fused),
        )


def crops_to_tensor(
    crops: np.ndarray | list[np.ndarray], *, device: torch.device | None = None
) -> Tensor:
    """Convert uint8 crops to a normalized NCHW float tensor.

    Accepts (N, H, W) mono or (N, H, W, C) colour crops.
    """
    array = np.asarray(crops)
    if array.ndim == 3:
        array = array[..., None]
    if array.ndim != 4:
        raise ValueError(f"expected (N, H, W) or (N, H, W, C) crops, got shape {array.shape}")

    tensor = torch.from_numpy(np.ascontiguousarray(array)).to(torch.float32).div_(255.0)
    tensor = tensor.permute(0, 3, 1, 2).contiguous()
    tensor = (tensor - IMAGENET_FREE_MEAN) / IMAGENET_FREE_STD
    return tensor.to(device) if device is not None else tensor
