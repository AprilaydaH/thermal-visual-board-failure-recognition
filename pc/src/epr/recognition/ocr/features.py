"""Marking features for the fusion input.

The OCR model itself comes later. What the LNN needs from it is a fixed-width vector, so the
interface is fixed now: a reading plus its confidence becomes a hashed character n-gram
vector. Hashing uses CRC-32 rather than ``hash`` so the encoding is stable across processes
and therefore reproducible in stored datasets.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np

OCR_FEATURE_DIM = 32
_NGRAM_SIZES = (1, 2, 3)
_HASH_DIM = OCR_FEATURE_DIM - 2


@dataclass(frozen=True)
class MarkingReading:
    """One OCR result for a component region."""

    text: str = ""
    confidence: float = 0.0

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


def encode_marking(reading: MarkingReading | str | None) -> np.ndarray:
    """Encode a marking into a fixed-width feature vector.

    The last two elements are the OCR confidence and a flag that a marking was read at all,
    so the network can tell "no text" apart from "text that hashes to zeros".
    """
    if reading is None:
        reading = MarkingReading()
    elif isinstance(reading, str):
        reading = MarkingReading(text=reading, confidence=1.0)

    features = np.zeros(OCR_FEATURE_DIM, dtype=np.float32)
    text = _normalize(reading.text)
    if not text:
        return features

    for size in _NGRAM_SIZES:
        for start in range(len(text) - size + 1):
            ngram = text[start : start + size]
            bucket = zlib.crc32(ngram.encode("utf-8")) % _HASH_DIM
            features[bucket] += 1.0

    norm = float(np.linalg.norm(features[:_HASH_DIM]))
    if norm > 0:
        features[:_HASH_DIM] /= norm
    features[_HASH_DIM] = float(np.clip(reading.confidence, 0.0, 1.0))
    features[_HASH_DIM + 1] = 1.0
    return features


def _normalize(text: str) -> str:
    return "".join(character for character in text.upper() if character.isalnum())
