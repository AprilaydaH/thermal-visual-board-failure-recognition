"""Multimodal electronic parts recognition, PC application."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("epr")
except PackageNotFoundError:
    __version__ = "0.1.0"
