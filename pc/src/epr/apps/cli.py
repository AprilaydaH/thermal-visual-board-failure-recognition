"""Installed command-line entry point.

    epr acquire --frames 5
    epr train detector --epochs 40
    epr train eval
    epr packages build
    epr demo

Each subcommand is an existing service. This module only dispatches, and it imports the
chosen service lazily so that ``epr acquire`` still runs on a machine without PyTorch.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence

COMMANDS = {
    "acquire": "epr.apps.acquisition_service.__main__",
    "demo": "epr.apps.recognition_demo.__main__",
    "train": "epr.apps.training_service.__main__",
    "packages": "epr.apps.package_library.__main__",
}

HELP = """Electronic parts recognition — PC application.

Usage:
  epr acquire [options]     record raw frame sets (simulator if no head)
  epr train <command>       train or evaluate a candidate model
  epr packages <command>    build or query the KiCad package table
  epr demo [options]        run the recognition pipeline on a simulated sequence
  epr --version

Set EPR_HOME to choose where data, models and projects are stored.
"""


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(HELP.strip())
        return 0
    if args[0] in ("-V", "--version"):
        from epr import __version__

        print(__version__)
        return 0

    command, rest = args[0], args[1:]
    module_name = COMMANDS.get(command)
    if module_name is None:
        print(f"unknown command: {command}\n", file=sys.stderr)
        print(HELP.strip(), file=sys.stderr)
        return 2

    module = importlib.import_module(module_name)
    return int(module.main(rest))


if __name__ == "__main__":
    sys.exit(main())
