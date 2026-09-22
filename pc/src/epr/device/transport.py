"""Transport adapters.

USB bulk and Ethernet transports arrive with the hardware. Until then the acquisition path
consumes a non-seekable byte stream, exactly as it will from the device.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator


class ChunkedByteStream:
    """Presents an iterable of byte chunks as a ``read(size)`` stream."""

    def __init__(self, chunks: Iterable[bytes]) -> None:
        self._chunks: Iterator[bytes] = iter(chunks)
        self._buffer = bytearray()

    def read(self, size: int, /) -> bytes:
        while len(self._buffer) < size:
            chunk = next(self._chunks, None)
            if chunk is None:
                break
            self._buffer.extend(chunk)
        taken = bytes(self._buffer[:size])
        del self._buffer[:size]
        return taken
